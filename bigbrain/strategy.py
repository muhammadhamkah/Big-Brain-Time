"""A strategy book: the live trader running a lab rule instead of the playbook.

The lab found that daily trend following with volatility targeting on the majors is the one
rule that survived a fair test. This module trades a lab rule live with exactly the machinery
the playbook uses (chronological fills at live quotes, queued next-open fills in replay, real
funding, fees and slippage, atomic bookkeeping, the same wallet state the dashboard and
``bigbrain portfolio`` read) and none of the playbook's learning: no beliefs, no exit variants.
The rule decides; the book executes.

Parameters are re-chosen every ``refit_days`` on all the history the book can see, with the
same selection the lab's walk-forward used (the setting whose mean trade is the most standard
errors above zero), so what runs live is what was tested. A buy-and-hold control of the same
coins from the same start is kept beside it: the number the book has to beat.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from bigbrain import lab
from bigbrain.brain import Brain
from bigbrain.ingest.market import Bar
from bigbrain.trader import INTERVAL_SECONDS, MIN_NOTIONAL, Position, Trader, _shift

NO_STOP = 0.99  # a protective level that never fills, for rules that carry no stop
NEVER = 10 ** 9


class StrategyTrader(Trader):
    def __init__(self, brain: Brain, book: str, rule: str, symbols: list[str], interval: str = "1d", wallet: float = 1000.0, market: str = "perps",
                 grid: dict[str, list] | None = None, target_vol: float | None = None, refit_days: int = 30, lookback: int = 800, **fetchers) -> None:
        if rule not in lab.RULES:
            raise ValueError(f"unknown rule '{rule}'")
        if "funding" in lab.RULES[rule]["needs"]:
            raise ValueError(f"the {rule} rule needs funding history, which the live book does not fetch yet")
        universe = [{"symbol": s, "quote_volume": 0.0, "last": 0.0} for s in symbols]
        super().__init__(brain, book=book, interval=interval, wallet=wallet, top=len(symbols), lookback=lookback, market=market, frozen=True,
                         fetch_universe=lambda n: universe, **fetchers)
        self.rule, self.symbols = rule, list(symbols)
        defaults = lab.RULES[rule]["defaults"]
        self.grid = dict(grid) if grid else {k: [v] for k, v in defaults.items()}
        if target_vol is not None and "target_vol" in defaults:
            self.grid["target_vol"] = [target_vol]
        self.refit_days = refit_days
        cfg = brain.get_state(f"strategy:{book}") or {}
        self.params: dict | None = cfg.get("params")
        self.last_refit: str = cfg.get("last_refit", "")
        self.hold_base: dict[str, float] = cfg.get("hold_base") or {}
        self.hold_start: str = cfg.get("hold_start", "")
        self.closes: dict[str, float] = cfg.get("closes") or {}
        self._decisions: dict[str, dict] = {}
        self._save_config()

    # ------------------------------------------------------------ config
    def _save_config(self) -> None:
        self.brain.set_state(f"strategy:{self.book}", {"rule": self.rule, "symbols": self.symbols, "grid": self.grid, "params": self.params, "last_refit": self.last_refit,
                                                       "refit_days": self.refit_days, "hold_base": self.hold_base, "hold_start": self.hold_start, "closes": self.closes})

    # -------------------------------------------------------------- tick
    def _before_symbols(self, market: dict[str, list[Bar]]) -> None:
        closed = {s: bars[:-1] for s, bars in market.items() if s in self.symbols and len(bars) > 2}
        if not closed:
            return
        today = max(bars[-1].date for bars in closed.values())
        for s, bars in closed.items():
            self.closes[s] = bars[-1].close
        if not self.hold_base:  # the control starts when the book starts
            self.hold_base, self.hold_start = {s: bars[-1].close for s, bars in closed.items()}, today
        self._maybe_refit(closed, today)
        decs = lab.decisions_for(self.rule, closed, self.params or {}, {"interval": self.interval})
        self._decisions = {s: d[-1] for s, d in decs.items() if d}
        self._save_config()

    def _maybe_refit(self, closed: dict[str, list[Bar]], today: str) -> None:
        due = self.params is None or not self.last_refit or _shift(self.last_refit, self.refit_days * 86400) <= today
        if not due:
            return
        points = lab.grid_points(self.grid)
        if len(points) == 1:
            best_params, best = points[0], None
        else:
            costs = lab.Costs(fee=self.fees["taker"], spread_bps=1.0)
            rows = lab.grid_search(self.rule, closed, self.grid, costs, self.interval, workers=4)
            best_params, best = max(rows, key=lambda pr: lab._score(pr[1]))
        changed = best_params != self.params
        self.params, self.last_refit = best_params, today
        if changed or best is not None:
            note = f" (in-sample PF {min(best.profit_factor, 99):.2f} over {best.trades} trades)" if best is not None else ""
            self._log(f"    REFIT {self.rule}: {lab.describe_params(best_params)}{note}")
            self.brain._journal("refit", f"book {self.book}: {self.rule} {lab.describe_params(best_params)}{note}")

    # ------------------------------------------------------------ symbol
    def _step_symbol(self, symbol: str, bars: list[Bar], volume_24h: float) -> list[dict]:
        events: list[dict] = []
        quiet = SimpleNamespace(rsi=None, sma20=None, close=bars[-1].close)  # the exit rule is "never": readings are not consulted
        last_done = self.wallet.last_bar.get(symbol)
        dates = [b.date for b in bars]
        start = dates.index(last_done) + 1 if last_done in dates else len(bars) - 1
        for i in range(start, len(bars) - 1):  # missed candles: fills, stops and funding only, no new decisions
            if not self._has_work(symbol):
                break
            events += self._process_bar(symbol, bars[i], quiet, [], *self._liquidity(bars, i, volume_24h), catch_up=True)
        volume_24h, candle_qv = self._liquidity(bars, len(bars) - 1, volume_24h)
        events += self._process_bar(symbol, bars[-1], quiet, [], volume_24h, candle_qv)
        d = self._decisions.get(symbol)
        if d is not None and symbol in self.symbols:
            events += self._act(symbol, bars[-1], d, volume_24h, candle_qv)
        return events

    def _act(self, symbol: str, bar: Bar, d: dict, volume_24h: float, candle_qv: float) -> list[dict]:
        """Bring the position in line with the rule's decision: exit what it no longer wants, enter what it does."""
        events: list[dict] = []
        target, weight = int(d.get("target", 0)), float(d.get("weight", 1.0))
        key = f"{symbol}:{self.rule}"
        raw = self.wallet.positions.get(key)
        if raw and not raw.get("pending_exit") and (target == 0 or raw["side"] != target):
            pos = Position(**raw)
            q = self._fresh_quote(symbol)
            if q is not None:
                events.append(self._close(pos, bar, q["bid"] if pos.side == 1 else q["ask"], "signal", volume_24h, candle_qv, basis="quote"))
                del self.wallet.positions[key]
                raw = None
            else:
                pos.pending_exit = "signal"  # fills at the next candle's open, exactly as the lab filled it
                self.wallet.positions[key] = asdict(pos)
        held_side = raw["side"] if raw and not raw.get("pending_exit") else 0
        if target and weight > 0 and held_side != target and key not in self.wallet.pending:
            notional = self.wallet.equity() * min(weight, 1.0) / len(self.symbols)  # each coin gets an equal share, sized by the rule's weight
            if notional < MIN_NOTIONAL:
                return events
            stop = d.get("stop")
            stop_pct = target * (1 - stop / bar.close) if stop else NO_STOP
            order = {"symbol": symbol, "signal": self.rule, "side": target, "notional": notional, "stop_pct": max(stop_pct, 0.001), "risk_pct": stop_pct if stop else 0.0,
                     "explore": False, "variant": "rule", "max_bars": NEVER, "exit_rule": "never", "placed": bar.date, "signal_close": bar.close, "p_win": 0.5, "samples": 0,
                     "ctx": {"regime": "strategy", "vol_bucket": "mid", "vol20": 0.0, "atr_pct": 0.0, "rsi": None, "band_pos": None, "macd_hist": None, "hour_utc": None,
                             "weight": weight, "params": self.params}, "model_version": self.brain.RESULTS_VERSION}
            q = self._fresh_quote(symbol)
            if q is not None:
                price = q["ask"] if target == 1 else q["bid"]
                events.append(self._fill_entry(order, price, "quote", volume_24h, candle_qv, entry_time=bar.date, entered_at_open=False,
                                               last_funding_hour=_shift(bar.date, INTERVAL_SECONDS[self.interval])[:13] or bar.date[:13]))
            elif self.live:
                events.append({"symbol": symbol, "signal": self.rule, "action": "skip", "why": "no fresh quote at decision time"})
            else:
                self.wallet.pending[key] = order  # the exit (if any) fills first at the next open, then this
                events.append({"symbol": symbol, "signal": self.rule, "action": "queued", "notional": notional, "explore": False})
        return events

    # ------------------------------------------------------------ report
    def benchmark(self) -> dict:
        """Holding the same coins, equal weight, since the book started."""
        pairs = [(self.closes[s], self.hold_base[s]) for s in self.symbols if s in self.closes and self.hold_base.get(s)]
        if not pairs:
            return {"since": self.hold_start, "return": 0.0, "symbols": 0}
        return {"since": self.hold_start, "return": sum(c / b for c, b in pairs) / len(pairs) - 1, "symbols": len(pairs)}

    def report(self) -> dict:
        r = super().report()
        r.update({"rule": self.rule, "params": self.params, "last_refit": self.last_refit, "benchmark": self.benchmark(), "symbols": self.symbols})
        return r
