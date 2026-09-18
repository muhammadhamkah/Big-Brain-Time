"""Watch a market in real time and keep score.

The watcher polls a symbol at each candle close, recomputes the indicators,
and notices *events*: RSI crossing into oversold, a close outside a
Bollinger band, a moving-average cross, a MACD flip. Every event is
recorded as a *call* with an explicit hypothesis ("RSI oversold: price will
be higher N bars later"). Once N bars have passed the call is graded against
what the market did, and the running scorecard per signal becomes a lesson
cell, so the brain learns which signals mean something on this market at
this timeframe and which are noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import Bar, fetch_binance

INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}

# signal -> (hypothesis direction, description)
SIGNALS = {
    "rsi_oversold": ("up", "RSI(14) crossed below 30: mean reversion says price should be higher soon"),
    "rsi_overbought": ("down", "RSI(14) crossed above 70: mean reversion says price should be lower soon"),
    "below_lower_band": ("up", "close fell below the lower Bollinger band: a stretched move that mean reversion buys"),
    "above_upper_band": ("up", "close broke above the upper Bollinger band: a volatility breakout that trend traders buy"),
    "golden_cross": ("up", "20-period SMA crossed above the 50-period: trend following turns long"),
    "death_cross": ("down", "20-period SMA crossed below the 50-period: trend following turns flat or short"),
    "macd_bullish": ("up", "MACD histogram turned positive: momentum turning up"),
    "macd_bearish": ("down", "MACD histogram turned negative: momentum turning down"),
}


@dataclass
class Reading:
    bar_time: str
    close: float
    rsi: float | None
    sma20: float | None
    sma50: float | None
    upper: float | None
    lower: float | None
    macd_hist: float | None
    vol20: float | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def readings(bars: list[Bar]) -> list[Reading]:
    closes = [b.close for b in bars]
    rsi = ind.rsi(closes, 14)
    s20, s50 = ind.sma(closes, 20), ind.sma(closes, 50)
    upper, _, lower = ind.bollinger(closes)
    _, _, hist = ind.macd(closes)
    vol = ind.realized_volatility(closes, 20)
    return [
        Reading(bars[i].date, closes[i], rsi[i], s20[i], s50[i], upper[i], lower[i], hist[i], vol[i])
        for i in range(len(bars))
    ]


def detect(prev: Reading, cur: Reading) -> list[str]:
    """Signals that fired between two consecutive closed candles."""
    out = []
    if prev.rsi is not None and cur.rsi is not None:
        if prev.rsi >= 30 > cur.rsi:
            out.append("rsi_oversold")
        if prev.rsi <= 70 < cur.rsi:
            out.append("rsi_overbought")
    if None not in (prev.lower, cur.lower, prev.upper, cur.upper):
        if prev.close >= prev.lower and cur.close < cur.lower:
            out.append("below_lower_band")
        if prev.close <= prev.upper and cur.close > cur.upper:
            out.append("above_upper_band")
    if None not in (prev.sma20, prev.sma50, cur.sma20, cur.sma50):
        if prev.sma20 <= prev.sma50 and cur.sma20 > cur.sma50:
            out.append("golden_cross")
        if prev.sma20 >= prev.sma50 and cur.sma20 < cur.sma50:
            out.append("death_cross")
    if prev.macd_hist is not None and cur.macd_hist is not None:
        if prev.macd_hist <= 0 < cur.macd_hist:
            out.append("macd_bullish")
        if prev.macd_hist >= 0 > cur.macd_hist:
            out.append("macd_bearish")
    return out


class Watcher:
    def __init__(self, brain: Brain, symbol: str = "BTCUSDT", interval: str = "15m", horizon: int = 12, lookback: int = 300, fetch=fetch_binance) -> None:
        if interval not in INTERVAL_SECONDS:
            raise ValueError(f"interval must be one of {list(INTERVAL_SECONDS)}")
        self.brain, self.symbol, self.interval, self.horizon, self.lookback = brain, symbol.upper(), interval, horizon, lookback
        self.fetch = fetch
        self.key = f"watch:{self.symbol}:{self.interval}"

    # ----------------------------------------------------------------- tick
    def tick(self, bars: list[Bar] | None = None) -> dict:
        """One pass: fetch, detect new signals on the last closed candle, grade due calls."""
        if bars is None:
            bars = self.fetch(self.symbol, self.interval, self.lookback)
        if len(bars) < 60:
            raise ValueError("need at least 60 candles")
        bars = bars[:-1]  # the final candle from the exchange is still forming; only closed candles count
        rs = readings(bars)
        cur, prev = rs[-1], rs[-2]
        last_seen = self.brain.get_state(self.key, {}).get("bar_time")
        fired: list[str] = []
        if cur.bar_time != last_seen:
            fired = detect(prev, cur)
            for sig in fired:
                self._record_call(sig, cur)
            self.brain.set_state(self.key, {**cur.as_dict(), "symbol": self.symbol, "interval": self.interval, "updated": _now()})
        graded = self._grade(bars)
        return {"bar_time": cur.bar_time, "close": cur.close, "rsi": cur.rsi, "fired": fired, "graded": graded, "new_bar": cur.bar_time != last_seen}

    def _record_call(self, signal: str, cur: Reading) -> None:
        direction, description = SIGNALS[signal]
        self.brain.db.execute(
            "INSERT OR IGNORE INTO calls (symbol, interval, signal, hypothesis, bar_time, price, horizon) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.symbol, self.interval, signal, direction, cur.bar_time, cur.close, self.horizon),
        )
        self.brain.learn(
            "observation",
            f"{self.symbol} {self.interval}: {signal.replace('_', ' ')} at {cur.bar_time}",
            f"{self.symbol} on the {self.interval} chart closed at {cur.close:.2f} at {cur.bar_time} UTC and {description}. "
            f"RSI(14) {cur.rsi:.1f}." + (f" 20-bar realized volatility {cur.vol20:.0%} annualized." if cur.vol20 else "")
            + f" Hypothesis: price {direction} within {self.horizon} bars; the brain will grade this call.",
            source=f"watch:{self.symbol}:{self.interval}",
            extra_concepts=[self.symbol.lower(), "live signal"],
        )
        self.brain.db.commit()

    # ---------------------------------------------------------------- grade
    def _grade(self, bars: list[Bar]) -> list[dict]:
        index = {b.date: i for i, b in enumerate(bars)}
        pending = self.brain.db.execute(
            "SELECT id, signal, hypothesis, bar_time, price, horizon FROM calls WHERE symbol = ? AND interval = ? AND graded_at IS NULL",
            (self.symbol, self.interval),
        ).fetchall()
        graded = []
        for row in pending:
            i = index.get(row["bar_time"])
            if i is None or i + row["horizon"] >= len(bars):
                continue
            future = bars[i + row["horizon"]].close
            ret = future / row["price"] - 1.0
            hit = int((ret > 0) if row["hypothesis"] == "up" else (ret < 0))
            self.brain.db.execute(
                "UPDATE calls SET graded_at = ?, outcome_return = ?, hit = ? WHERE id = ?", (_now(), ret, hit, row["id"])
            )
            graded.append({"signal": row["signal"], "bar_time": row["bar_time"], "return": ret, "hit": bool(hit)})
        if graded:
            self.brain.db.commit()
            for sig in {g["signal"] for g in graded}:
                self._update_scorecard(sig)
        return graded

    def scorecard(self) -> list[dict]:
        rows = self.brain.db.execute(
            "SELECT signal, hypothesis, COUNT(*) AS n, SUM(hit) AS hits, AVG(outcome_return) AS avg_ret,"
            " SUM(CASE WHEN graded_at IS NULL THEN 1 ELSE 0 END) AS pending"
            " FROM calls WHERE symbol = ? AND interval = ? GROUP BY signal ORDER BY n DESC",
            (self.symbol, self.interval),
        ).fetchall()
        out = []
        for r in rows:
            graded = r["n"] - r["pending"]
            out.append({
                "signal": r["signal"], "hypothesis": r["hypothesis"], "fired": r["n"], "graded": graded,
                "hits": r["hits"] or 0, "hit_rate": (r["hits"] or 0) / graded if graded else None,
                "avg_return": r["avg_ret"], "pending": r["pending"],
            })
        return out

    def _update_scorecard(self, signal: str) -> None:
        card = next((c for c in self.scorecard() if c["signal"] == signal), None)
        if card is None or card["graded"] < 5:
            return  # too few to say anything; the lesson would be noise
        title = f"{self.symbol} {self.interval}: {signal.replace('_', ' ')} scorecard"
        self.brain.forget(title=title)
        rate, avg = card["hit_rate"], card["avg_return"] or 0.0
        verdict = "supported" if rate >= 0.6 and (avg > 0) == (card["hypothesis"] == "up") else "refuted" if rate <= 0.4 else "unproven"
        direction, description = SIGNALS[signal]
        text = (
            f"Live scorecard for {self.symbol} on the {self.interval} chart. The signal '{signal.replace('_', ' ')}' ({description}) "
            f"fired {card['fired']} times; {card['graded']} calls have been graded {card['hits']} hits, a {rate:.0%} hit rate, "
            f"with an average move of {avg:+.2%} over {self.horizon} bars against a hypothesis of '{direction}'. "
            f"Verdict so far: the hypothesis is {verdict} on this market and timeframe. "
            "This is walk-forward evidence from live candles, not a backtest, and it updates as more calls are graded; "
            "treat small samples with suspicion and size accordingly."
        )
        self.brain.learn("lesson", title, text, source=f"watch:{self.symbol}:{self.interval}", extra_concepts=[self.symbol.lower(), "live signal", "walk-forward"])

    # ----------------------------------------------------------------- loop
    def seconds_until_next_close(self, now: float | None = None) -> float:
        period = INTERVAL_SECONDS[self.interval]
        now = time.time() if now is None else now
        return period - (now % period) + 5  # a few seconds of grace for the exchange to finalize the candle

    def run(self, log=print, once: bool = False, sleep=time.sleep) -> None:
        while True:
            try:
                result = self.tick()
                stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                fired = ", ".join(result["fired"]) or "no new signal"
                graded = "".join(f"  graded {g['signal']}: {'hit' if g['hit'] else 'miss'} ({g['return']:+.2%})" for g in result["graded"])
                rsi = f"{result['rsi']:.1f}" if result["rsi"] is not None else "n/a"
                log(f"[{stamp}] {self.symbol} {self.interval} close {result['close']:.2f} rsi {rsi} | {fired}{graded}")
            except Exception as exc:
                log(f"watch error: {exc}")
            if once:
                return
            sleep(self.seconds_until_next_close())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def live_context(brain: Brain) -> list[str]:
    """One line per watched market, for the cortex to include in answers."""
    lines = []
    for r in brain.db.execute("SELECT key, value FROM state WHERE key LIKE 'watch:%'"):
        import json
        st = json.loads(r["value"])
        rsi = f"RSI {st['rsi']:.1f}" if st.get("rsi") is not None else ""
        band = ""
        if st.get("upper") and st.get("lower"):
            band = "above upper band" if st["close"] > st["upper"] else "below lower band" if st["close"] < st["lower"] else "inside bands"
        lines.append(f"{st['symbol']} {st['interval']} as of {st['bar_time']} UTC: close {st['close']:.2f}, {rsi}, {band}")
    return lines
