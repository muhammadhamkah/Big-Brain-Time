"""Trading strategies and a small backtester.

A strategy is a function from closes to a position series (1 = long, 0 =
flat). The backtester turns that into an equity curve and performance
metrics, and ``learn_backtests`` records what the brain learned as *lesson*
cells: "SMA crossover on X: Sharpe 0.9, max drawdown -12%". Lessons link to
the strategy concept cells and to the market observations for the same symbol,
so a question about "trend following in a bear market" can reach real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import Bar

Positions = list[int]
Strategy = Callable[[Sequence[float]], Positions]


def sma_crossover(closes: Sequence[float], fast: int = 20, slow: int = 50) -> Positions:
    f, s = ind.sma(closes, fast), ind.sma(closes, slow)
    return [1 if (a is not None and b is not None and a > b) else 0 for a, b in zip(f, s)]


def rsi_mean_reversion(closes: Sequence[float], period: int = 14, entry: float = 30.0, exit_: float = 55.0) -> Positions:
    r = ind.rsi(closes, period)
    pos, out = 0, []
    for v in r:
        if v is not None:
            if pos == 0 and v <= entry:
                pos = 1
            elif pos == 1 and v >= exit_:
                pos = 0
        out.append(pos)
    return out


def bollinger_breakout(closes: Sequence[float], period: int = 20, width: float = 2.0) -> Positions:
    upper, mid, _ = ind.bollinger(closes, period, width)
    pos, out = 0, []
    for c, u, m in zip(closes, upper, mid):
        if u is not None and m is not None:
            if pos == 0 and c > u:
                pos = 1
            elif pos == 1 and c < m:
                pos = 0
        out.append(pos)
    return out


def buy_and_hold(closes: Sequence[float]) -> Positions:
    return [1] * len(closes)


STRATEGIES: dict[str, tuple[Strategy, str]] = {
    "sma crossover": (sma_crossover, "Trend following: long when the 20-day SMA is above the 50-day SMA, flat otherwise."),
    "rsi mean reversion": (rsi_mean_reversion, "Mean reversion: buy when RSI(14) drops to 30 (oversold), exit when it recovers to 55."),
    "bollinger breakout": (bollinger_breakout, "Volatility breakout: buy a close above the upper Bollinger band, exit below the middle band."),
    "buy and hold": (buy_and_hold, "Benchmark: always long."),
}


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    bars: int
    total_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float
    exposure: float

    def describe(self, rationale: str) -> str:
        verdict = "strong" if self.sharpe > 1.0 else "positive" if self.sharpe > 0.3 else "weak" if self.sharpe > 0 else "negative"
        return (
            f"{rationale} Backtest on {self.symbol} over {self.bars} bars: total return {self.total_return:+.1%}, "
            f"Sharpe ratio {self.sharpe:.2f} ({verdict} risk-adjusted return), max drawdown {self.max_drawdown:.1%}, "
            f"{self.trades} trades with a {self.win_rate:.0%} win rate, {self.exposure:.0%} time in the market. "
            "Results include a 0.1% transaction cost per trade and no slippage; treat single-asset backtests as "
            "evidence, not proof, because of overfitting risk."
        )


def backtest(name: str, symbol: str, bars: Sequence[Bar], strategy: Strategy, cost: float = 0.001) -> BacktestResult:
    closes = [b.close for b in bars]
    positions = strategy(closes)
    rets = ind.returns(closes)
    equity, strat_rets = [1.0], []
    trades, wins, entry_price, prev_pos = 0, 0, None, 0
    for i in range(1, len(closes)):
        pos = positions[i - 1]  # trade on the next bar after the signal
        r = pos * rets[i]
        if pos != prev_pos:
            r -= cost
            if pos == 1:
                entry_price, trades = closes[i], trades + 1
            elif entry_price is not None:
                wins += closes[i] > entry_price
        prev_pos = pos
        strat_rets.append(r)
        equity.append(equity[-1] * (1 + r))
    if prev_pos == 1 and entry_price is not None:
        wins += closes[-1] > entry_price
    return BacktestResult(
        strategy=name,
        symbol=symbol,
        bars=len(bars),
        total_return=equity[-1] - 1.0,
        sharpe=ind.sharpe(strat_rets),
        max_drawdown=ind.max_drawdown(equity),
        trades=trades,
        win_rate=(wins / trades) if trades else 0.0,
        exposure=sum(positions) / len(positions) if positions else 0.0,
    )


def learn_backtests(brain: Brain, symbol: str, bars: Sequence[Bar], source: str = "backtest") -> list[BacktestResult]:
    """Run every built-in strategy on ``bars`` and store each result as a lesson cell."""
    results = []
    for name, (fn, rationale) in STRATEGIES.items():
        result = backtest(name, symbol, bars, fn)
        brain.learn(
            "lesson",
            f"{name} on {symbol}",
            result.describe(rationale),
            source=source,
            extra_concepts=[symbol.lower(), "backtesting"],
        )
        results.append(result)
    return results
