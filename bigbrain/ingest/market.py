"""Market data ingestion: OHLCV bars in, indicator observations out.

Bars come from a CSV (``date,open,high,low,close,volume``, oldest first) or
from ``synthetic()`` when you want to exercise the brain without real data.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.net import HTTPStatusError, http_get


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0  # volume in the quote currency (USDT), when the source provides it


def parse_csv(text: str) -> list[Bar]:
    bars: list[Bar] = []
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.lower().strip(): c for c in reader.fieldnames or []}

    def col(name: str, *alts: str) -> str:
        for n in (name, *alts):
            if n in cols:
                return cols[n]
        raise ValueError(f"CSV is missing a '{name}' column; columns are {list(cols)}")

    d, o, h, l, c = col("date", "timestamp", "time"), col("open"), col("high"), col("low"), col("close", "adj close", "adj_close")
    v = cols.get("volume")
    for row in reader:
        try:
            bars.append(Bar(row[d], float(row[o]), float(row[h]), float(row[l]), float(row[c]), float(row[v]) if v and row[v] else 0.0))
        except (ValueError, TypeError):
            continue  # Stooq and others sometimes emit rows with missing numbers
    if bars and bars[0].date > bars[-1].date:
        bars.reverse()
    return bars


def load_csv(path: str | Path) -> list[Bar]:
    with open(path, newline="") as fh:
        return parse_csv(fh.read())


# ------------------------------------------------------------ online sources
BINANCE_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
BINANCE_INTERVALS = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")


def parse_binance_klines(payload: list) -> list[Bar]:
    """Binance kline rows: [open_time, open, high, low, close, volume, close_time, ...]."""
    bars = []
    for row in payload:
        ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
        quote = float(row[7]) if len(row) > 7 else 0.0
        bars.append(Bar(ts.strftime("%Y-%m-%d %H:%M"), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), quote))
    return bars


STABLE_OR_FIAT = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "GBP", "TRY", "BRL", "ARS", "UST", "USD1", "PYUSD", "AEUR", "XUSD", "USDE", "EURI", "JPY", "ZAR", "MXN", "PLN", "RON", "CZK", "UAH", "COP"}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def parse_top_usdt_pairs(tickers: list, n: int = 100) -> list[dict]:
    """Top-``n`` spot USDT pairs by 24h quote volume, skipping stablecoin/fiat pairs and leveraged tokens."""
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT") or sym.endswith(LEVERAGED_SUFFIXES):
            continue
        base = sym[:-4]
        if base in STABLE_OR_FIAT or not base:
            continue
        try:
            qv = float(t.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            continue
        if qv <= 0:
            continue
        rows.append({"symbol": sym, "quote_volume": qv, "last": float(t.get("lastPrice") or 0.0)})
    rows.sort(key=lambda r: -r["quote_volume"])
    return rows[:n]


def top_usdt_pairs(n: int = 100) -> list[dict]:
    last: Exception | None = None
    for host in BINANCE_HOSTS:
        try:
            tickers = json.loads(http_get(f"{host}/api/v3/ticker/24hr", headers={"Accept": "application/json"}).decode("utf-8"))
            return parse_top_usdt_pairs(tickers, n)
        except (HTTPStatusError, OSError) as exc:
            last = exc
    raise RuntimeError(f"Binance did not return tickers: {last}")


def fetch_binance(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 1000) -> list[Bar]:
    """Public candles from Binance (no account needed). Falls back to the public data mirror."""
    if interval not in BINANCE_INTERVALS:
        raise ValueError(f"interval must be one of {BINANCE_INTERVALS}")
    last: Exception | None = None
    for host in BINANCE_HOSTS:
        url = f"{host}/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={min(limit, 1000)}"
        try:
            return parse_binance_klines(json.loads(http_get(url, headers={"Accept": "application/json"}).decode("utf-8")))
        except (HTTPStatusError, OSError) as exc:
            last = exc
    raise RuntimeError(f"Binance did not return candles for {symbol}: {last}")


# ---------------------------------------------------------- Binance perps
FUTURES_HOST = "https://fapi.binance.com"


def fetch_binance_futures(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 1000) -> list[Bar]:
    """USDT-margined perpetual candles (public)."""
    if interval not in BINANCE_INTERVALS:
        raise ValueError(f"interval must be one of {BINANCE_INTERVALS}")
    url = f"{FUTURES_HOST}/fapi/v1/klines?symbol={symbol.upper()}&interval={interval}&limit={min(limit, 1500)}"
    return parse_binance_klines(json.loads(http_get(url, headers={"Accept": "application/json"}).decode("utf-8")))


def top_usdt_perps(n: int = 100) -> list[dict]:
    tickers = json.loads(http_get(f"{FUTURES_HOST}/fapi/v1/ticker/24hr", headers={"Accept": "application/json"}).decode("utf-8"))
    return parse_top_usdt_pairs(tickers, n)


def parse_funding(payload: list) -> dict[str, dict]:
    """{symbol: {"rate": last funding rate, "next": next funding time (ms)}} from /fapi/v1/premiumIndex."""
    out = {}
    for row in payload:
        try:
            out[row["symbol"]] = {"rate": float(row.get("lastFundingRate") or 0.0), "next": int(row.get("nextFundingTime") or 0)}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def funding_rates() -> dict[str, dict]:
    return parse_funding(json.loads(http_get(f"{FUTURES_HOST}/fapi/v1/premiumIndex", headers={"Accept": "application/json"}).decode("utf-8")))


def fetch_stooq(symbol: str) -> list[Bar]:
    """Free daily history from Stooq, e.g. 'aapl.us', 'spy.us', '^spx', 'btc.v'."""
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    text = http_get(url, headers={"Accept": "text/csv,*/*"}).decode("utf-8", errors="replace")
    if "No data" in text[:100] or "<html" in text[:200].lower():
        raise RuntimeError(f"Stooq has no daily data for {symbol!r} (try 'aapl.us' or 'btc.v')")
    return parse_csv(text)


def synthetic(symbol: str = "SYNTH", n: int = 400, seed: int = 7, drift: float = 0.0004, vol: float = 0.015) -> list[Bar]:
    """Geometric random walk with a regime switch, deterministic for a given seed."""
    rng = random.Random(seed)
    bars: list[Bar] = []
    price = 100.0
    for i in range(n):
        regime_drift = drift if i < n * 0.6 else -drift * 1.5
        ret = rng.gauss(regime_drift, vol)
        open_ = price
        close = price * math.exp(ret)
        high = max(open_, close) * (1 + abs(rng.gauss(0, vol / 3)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, vol / 3)))
        bars.append(Bar(f"D{i:04d}", open_, high, low, close, 1_000_000 * (1 + rng.random())))
        price = close
    return bars


def snapshot(symbol: str, bars: list[Bar]) -> ind.IndicatorSnapshot:
    closes = [b.close for b in bars]
    highs, lows = [b.high for b in bars], [b.low for b in bars]
    upper, _, lower = ind.bollinger(closes)
    _, _, hist = ind.macd(closes)
    bb_pos = None
    if upper[-1] is not None and lower[-1] is not None and upper[-1] != lower[-1]:
        bb_pos = (closes[-1] - lower[-1]) / (upper[-1] - lower[-1])
    return ind.IndicatorSnapshot(
        symbol=symbol,
        close=closes[-1],
        sma20=ind.sma(closes, 20)[-1],
        sma50=ind.sma(closes, 50)[-1],
        sma200=ind.sma(closes, 200)[-1],
        rsi14=ind.rsi(closes, 14)[-1],
        macd_hist=hist[-1],
        bb_position=bb_pos,
        atr14=ind.atr(highs, lows, closes, 14)[-1],
        vol20=ind.realized_volatility(closes, 20)[-1],
        drawdown=ind.max_drawdown(closes[-60:]) if len(closes) >= 2 else 0.0,
    )


def learn_bars(brain: Brain, symbol: str, bars: Iterable[Bar], source: str = "market data") -> list[str]:
    """Compute indicators on ``bars`` and teach the brain what they say. Returns cell titles learned."""
    bars = list(bars)
    if len(bars) < 30:
        raise ValueError("need at least 30 bars to say anything useful")
    snap = snapshot(symbol, bars)
    learned: list[str] = []
    for title, text in snap.observations():
        text = f"{text} (as of {bars[-1].date})"
        cell, _ = brain.learn("observation", title, text, source=source, extra_concepts=[symbol.lower()])
        learned.append(cell.title)
    return learned
