"""Market data ingestion: OHLCV bars in, indicator observations out.

Bars come from a CSV (``date,open,high,low,close,volume``, oldest first) or
from ``synthetic()`` when you want to exercise the brain without real data.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_csv(path: str | Path) -> list[Bar]:
    bars: list[Bar] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}
        def col(name: str, *alts: str) -> str:
            for n in (name, *alts):
                if n in cols:
                    return cols[n]
            raise ValueError(f"CSV is missing a '{name}' column; columns are {list(cols)}")
        d, o, h, l, c = col("date", "timestamp", "time"), col("open"), col("high"), col("low"), col("close", "adj close", "adj_close")
        v = cols.get("volume")
        for row in reader:
            bars.append(Bar(row[d], float(row[o]), float(row[h]), float(row[l]), float(row[c]), float(row[v]) if v else 0.0))
    if bars and bars[0].date > bars[-1].date:
        bars.reverse()
    return bars


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
