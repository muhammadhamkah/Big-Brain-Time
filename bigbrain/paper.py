"""Paper trading: every built-in strategy runs a virtual account on live candles.

Each (symbol, interval, strategy) has its own account that starts with
``STARTING_EQUITY``. At every closed candle the strategy's rule is
re-evaluated on the closed history; when its target position changes the
account trades at that candle's close, paying ``COST`` per side, exactly
the assumptions the backtester makes, so a live paper record and a backtest
over the same candles agree. Open positions are marked to market each
candle, so drawdown is measured on real equity, not on closed trades.

Every ``LESSON_EVERY`` closed trades the running record becomes a lesson
cell, which is the honest kind of evidence: out of sample by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import Bar
from bigbrain.ingest.strategies import STRATEGIES

STARTING_EQUITY = 10_000.0
COST = 0.001  # per side, same as the backtester
LESSON_EVERY = 10
MAX_CURVE = 5000  # equity points kept per account


@dataclass
class Account:
    strategy: str
    cash: float = STARTING_EQUITY
    qty: float = 0.0
    entry_price: float = 0.0
    entry_time: str = ""
    last_bar: str = ""
    peak: float = STARTING_EQUITY
    max_drawdown: float = 0.0
    curve: list = None  # [(bar_time, equity)]

    def __post_init__(self):
        if self.curve is None:
            self.curve = []

    @property
    def in_position(self) -> bool:
        return self.qty > 0

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        return cls(**d)


class PaperTrader:
    def __init__(self, brain: Brain, symbol: str, interval: str, strategies: dict | None = None) -> None:
        self.brain, self.symbol, self.interval = brain, symbol.upper(), interval
        self.strategies = strategies or STRATEGIES
        self.key = f"paper:{self.symbol}:{self.interval}"

    # ---------------------------------------------------------------- state
    def accounts(self) -> dict[str, Account]:
        stored = self.brain.get_state(self.key, {})
        return {name: Account.from_dict(stored[name]) if name in stored else Account(name) for name in self.strategies}

    def _save(self, accounts: dict[str, Account]) -> None:
        self.brain.set_state(self.key, {name: a.to_dict() for name, a in accounts.items()})

    # ----------------------------------------------------------------- step
    def step(self, bars: list[Bar]) -> list[dict]:
        """Advance every account to the last closed candle in ``bars``. Returns the trades made."""
        closes = [b.close for b in bars]
        last = bars[-1]
        accounts = self.accounts()
        events: list[dict] = []
        for name, (rule, _) in self.strategies.items():
            acct = accounts[name]
            if acct.last_bar == last.date:
                continue  # already processed this candle
            target = rule(closes)[-1]
            if target == 1 and not acct.in_position:
                acct.qty = acct.cash * (1 - COST) / last.close
                acct.cash = 0.0
                acct.entry_price, acct.entry_time = last.close, last.date
                self.brain.db.execute(
                    "INSERT OR IGNORE INTO paper_trades (symbol, interval, strategy, entry_time, entry_price, qty) VALUES (?, ?, ?, ?, ?, ?)",
                    (self.symbol, self.interval, name, last.date, last.close, acct.qty),
                )
                events.append({"strategy": name, "action": "buy", "price": last.close, "bar_time": last.date})
            elif target == 0 and acct.in_position:
                proceeds = acct.qty * last.close * (1 - COST)
                cost_basis = acct.qty * acct.entry_price / (1 - COST)
                pnl = proceeds - cost_basis
                ret = proceeds / cost_basis - 1.0
                acct.cash, acct.qty = proceeds, 0.0
                self.brain.db.execute(
                    "UPDATE paper_trades SET exit_time = ?, exit_price = ?, pnl = ?, ret = ? WHERE symbol = ? AND interval = ? AND strategy = ? AND entry_time = ?",
                    (last.date, last.close, pnl, ret, self.symbol, self.interval, name, acct.entry_time),
                )
                events.append({"strategy": name, "action": "sell", "price": last.close, "bar_time": last.date, "ret": ret})
                if self._closed_trades(name) % LESSON_EVERY == 0:
                    self._write_lesson(name, acct, last.close)
            eq = acct.equity(last.close)
            acct.peak = max(acct.peak, eq)
            acct.max_drawdown = min(acct.max_drawdown, eq / acct.peak - 1.0)
            acct.curve.append((last.date, round(eq, 2)))
            if len(acct.curve) > MAX_CURVE:
                acct.curve = acct.curve[-MAX_CURVE:]
            acct.last_bar = last.date
        self.brain.commit()
        self._save(accounts)
        return events

    # --------------------------------------------------------------- report
    def _closed_trades(self, strategy: str) -> int:
        return self.brain.db.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE symbol = ? AND interval = ? AND strategy = ? AND exit_time IS NOT NULL",
            (self.symbol, self.interval, strategy),
        ).fetchone()[0]

    def report(self, price: float | None = None) -> list[dict]:
        out = []
        for name, acct in self.accounts().items():
            last_price = price if price is not None else (acct.curve[-1][1] / acct.qty if acct.in_position and acct.curve else 0.0)
            eq = acct.equity(last_price) if acct.in_position else acct.cash
            row = self.brain.db.execute(
                "SELECT COUNT(*) AS n, SUM(CASE WHEN ret > 0 THEN 1 ELSE 0 END) AS wins, AVG(ret) AS avg_ret"
                " FROM paper_trades WHERE symbol = ? AND interval = ? AND strategy = ? AND exit_time IS NOT NULL",
                (self.symbol, self.interval, name),
            ).fetchone()
            out.append({
                "strategy": name, "equity": eq, "return": eq / STARTING_EQUITY - 1.0,
                "in_position": acct.in_position, "entry_price": acct.entry_price if acct.in_position else None,
                "trades": row["n"], "win_rate": (row["wins"] or 0) / row["n"] if row["n"] else None,
                "avg_trade": row["avg_ret"], "max_drawdown": acct.max_drawdown, "bars": len(acct.curve),
            })
        return out

    def _write_lesson(self, strategy: str, acct: Account, price: float) -> None:
        stats = next(r for r in self.report(price) if r["strategy"] == strategy)
        bh = next((r for r in self.report(price) if r["strategy"] == "buy and hold"), None)
        rets = [e for _, e in acct.curve]
        per_bar = ind.returns(rets) if len(rets) > 2 else []
        bars_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "30m": 17520, "1h": 8760, "4h": 2190, "1d": 365, "1w": 52}.get(self.interval, 365)
        sharpe = ind.sharpe(per_bar, bars_per_year) if per_bar else 0.0
        title = f"{self.symbol} {self.interval}: paper trading {strategy}"
        self.brain.forget(title=title)
        rationale = self.strategies[strategy][1]
        text = (
            f"Paper trading record for '{strategy}' on {self.symbol}, {self.interval} candles, live and out of sample. {rationale} "
            f"After {stats['trades']} closed trades over {stats['bars']} candles the account is at {stats['return']:+.1%} "
            f"(Sharpe {sharpe:.2f}), win rate {stats['win_rate']:.0%}, average trade {stats['avg_trade']:+.2%}, max drawdown {stats['max_drawdown']:.1%}, "
            f"with {COST:.1%} cost per side."
            + (f" Buy and hold over the same candles is at {bh['return']:+.1%}." if bh and bh['strategy'] != strategy else "")
            + " This is a forward test, the evidence that matters more than any backtest; it keeps updating as trades close."
        )
        self.brain.learn("lesson", title, text, source=f"paper:{self.symbol}:{self.interval}", extra_concepts=[self.symbol.lower(), "paper trading", "walk-forward"])
