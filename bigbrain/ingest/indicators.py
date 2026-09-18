"""Technical indicators in pure Python.

Every function takes a list of floats (oldest first) and returns a list the
same length, using ``None`` where the indicator is not yet defined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Series = list


def sma(values: Sequence[float], period: int) -> Series:
    out: list[float | None] = [None] * len(values)
    window_sum = 0.0
    for i, v in enumerate(values):
        window_sum += v
        if i >= period:
            window_sum -= values[i - period]
        if i >= period - 1:
            out[i] = window_sum / period
    return out


def ema(values: Sequence[float], period: int) -> Series:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> Series:
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Series, Series, Series]:
    """Return (macd line, signal line, histogram)."""
    fast_e, slow_e = ema(values, fast), ema(values, slow)
    line: list[float | None] = [
        (f - s) if f is not None and s is not None else None for f, s in zip(fast_e, slow_e)
    ]
    defined = [v for v in line if v is not None]
    sig_defined = ema(defined, signal)
    offset = len(line) - len(defined)
    sig: list[float | None] = [None] * offset + sig_defined
    hist = [(m - s) if m is not None and s is not None else None for m, s in zip(line, sig)]
    return line, sig, hist


def bollinger(values: Sequence[float], period: int = 20, width: float = 2.0) -> tuple[Series, Series, Series]:
    """Return (upper, middle, lower)."""
    mid = sma(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = mid[i]
        sd = math.sqrt(sum((v - mean) ** 2 for v in window) / period)
        upper[i], lower[i] = mean + width * sd, mean - width * sd
    return upper, mid, lower


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> Series:
    trs: list[float] = []
    for i in range(len(close)):
        if i == 0:
            trs.append(high[i] - low[i])
        else:
            trs.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    return ema(trs, period)


def returns(values: Sequence[float]) -> list[float]:
    return [0.0] + [(values[i] / values[i - 1] - 1.0) if values[i - 1] else 0.0 for i in range(1, len(values))]


def realized_volatility(values: Sequence[float], period: int = 20, periods_per_year: int = 252) -> Series:
    rets = returns(values)
    out: list[float | None] = [None] * len(values)
    for i in range(period, len(values)):
        window = rets[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((r - mean) ** 2 for r in window) / (period - 1)
        out[i] = math.sqrt(var) * math.sqrt(periods_per_year)
    return out


def max_drawdown(equity: Sequence[float]) -> float:
    peak, worst = -math.inf, 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1.0)
    return worst


def sharpe(rets: Sequence[float], periods_per_year: int = 252) -> float:
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd < 1e-12:
        return 0.0
    return (mean / sd) * math.sqrt(periods_per_year)


@dataclass
class IndicatorSnapshot:
    """The latest reading of every indicator, plus a plain-English description."""

    symbol: str
    close: float
    sma20: float | None
    sma50: float | None
    sma200: float | None
    rsi14: float | None
    macd_hist: float | None
    bb_position: float | None  # 0 = lower band, 1 = upper band
    atr14: float | None
    vol20: float | None
    drawdown: float

    def observations(self) -> list[tuple[str, str]]:
        """Return (title, description) pairs describing what the indicators say."""
        obs: list[tuple[str, str]] = []
        s = self.symbol
        if self.sma50 is not None and self.sma200 is not None:
            if self.sma50 > self.sma200:
                obs.append((f"{s}: golden cross regime", f"{s} has its 50-day moving average above the 200-day (golden cross), a classic long-term uptrend signal used by trend following systems."))
            else:
                obs.append((f"{s}: death cross regime", f"{s} has its 50-day moving average below the 200-day (death cross), a classic downtrend regime signal."))
        if self.sma20 is not None:
            side = "above" if self.close > self.sma20 else "below"
            obs.append((f"{s}: price {side} 20-day SMA", f"{s} closed at {self.close:.2f}, {side} its 20-day simple moving average of {self.sma20:.2f}. Short-term momentum is {'positive' if side == 'above' else 'negative'}."))
        if self.rsi14 is not None:
            if self.rsi14 >= 70:
                obs.append((f"{s}: RSI overbought", f"{s} RSI(14) is {self.rsi14:.1f}, an overbought reading. Mean reversion traders look for exhaustion here; momentum traders see strength."))
            elif self.rsi14 <= 30:
                obs.append((f"{s}: RSI oversold", f"{s} RSI(14) is {self.rsi14:.1f}, an oversold reading. Mean reversion systems buy dips like this, typically with a stop loss under recent support."))
            else:
                obs.append((f"{s}: RSI neutral", f"{s} RSI(14) is {self.rsi14:.1f}, a neutral momentum reading with no overbought or oversold signal."))
        if self.macd_hist is not None:
            obs.append((f"{s}: MACD {'bullish' if self.macd_hist > 0 else 'bearish'}", f"{s} MACD histogram is {self.macd_hist:+.3f}, so the MACD line is {'above' if self.macd_hist > 0 else 'below'} its signal line: {'bullish' if self.macd_hist > 0 else 'bearish'} momentum."))
        if self.bb_position is not None:
            if self.bb_position >= 1:
                obs.append((f"{s}: above upper Bollinger band", f"{s} closed above its upper Bollinger band, a volatility breakout that mean reversion traders fade and breakout traders follow."))
            elif self.bb_position <= 0:
                obs.append((f"{s}: below lower Bollinger band", f"{s} closed below its lower Bollinger band, a stretched move that mean reversion traders buy."))
        if self.vol20 is not None:
            level = "high" if self.vol20 > 0.35 else "low" if self.vol20 < 0.12 else "moderate"
            obs.append((f"{s}: {level} volatility", f"{s} 20-day realized volatility is {self.vol20:.0%} annualized ({level}). Position sizing should scale inversely with volatility; ATR(14) is {self.atr14:.2f}." if self.atr14 is not None else f"{s} 20-day realized volatility is {self.vol20:.0%} annualized ({level})."))
        if self.drawdown < -0.10:
            obs.append((f"{s}: in drawdown", f"{s} is {self.drawdown:.1%} below its peak, a meaningful drawdown. Risk management rules often cut exposure in deep drawdowns."))
        return obs
