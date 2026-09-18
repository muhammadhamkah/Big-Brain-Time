"""The brain trades: many pairs, one wallet, decisions from its own evidence.

Every candle, for every pair in the universe, the trader sees which signals
fired and in what context (trend regime, volatility bucket, RSI, band
position). It asks the belief table what that signal has done in that
context before. Positive expectancy after costs: trade at full size.
Not enough evidence yet: explore at half size, because a brain that never
tries anything never learns. Evidence says it loses: stand aside, with an
occasional small re-test so a changed market can change the belief.

Costs are modelled honestly: Binance spot VIP 0 fees (0.1% taker each way,
market orders) plus slippage from a liquidity model (half the estimated
spread, wider for thin pairs, plus impact from the order's share of the
candle's volume). Stops fill at the stop or at the open if the candle gaps
through it. Spot means long only; bearish signals are reasons to exit.

Every closed trade goes through ``bigbrain.postmortem``: findings, belief
update, and a post-mortem cell when the trade is instructive.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from bigbrain import postmortem as pm
from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import Bar, fetch_binance, top_usdt_pairs
from bigbrain.watch import INTERVAL_SECONDS, Reading, detect, readings

# ---- costs (Binance spot, VIP 0) -------------------------------------------
MAKER_FEE = 0.001
TAKER_FEE = 0.001
MIN_NOTIONAL = 10.0  # Binance minimum order in USDT

# ---- risk ---------------------------------------------------------------------
RISK_PER_TRADE = 0.01  # fraction of equity lost if the stop is hit
EXPLORE_RISK = 0.005
MAX_POSITION_FRACTION = 0.25
EXPLORE_PROBABILITY = 0.7  # while a (signal, context) has too little evidence
RETEST_PROBABILITY = 0.1  # occasionally re-test a context the brain believes loses

# ---- long-only playbook: how each signal is managed -----------------------------
# stop in ATR multiples, maximum bars held, and the rule that closes it early
PLAYBOOK = {
    "rsi_oversold": {"stop_atr": 2.0, "max_bars": 16, "exit": "rsi_recovered"},
    "below_lower_band": {"stop_atr": 2.0, "max_bars": 16, "exit": "back_to_middle"},
    "above_upper_band": {"stop_atr": 2.0, "max_bars": 48, "exit": "lost_middle"},
    "golden_cross": {"stop_atr": 2.5, "max_bars": 96, "exit": "death_cross"},
    "macd_bullish": {"stop_atr": 2.0, "max_bars": 48, "exit": "macd_bearish"},
}


def slippage_bps(notional: float, quote_volume_24h: float, candle_quote_volume: float) -> float:
    """Half the estimated spread plus market impact, in basis points."""
    musd = max(quote_volume_24h, 1.0) / 1e6
    spread = min(30.0, max(1.0, 20.0 / math.sqrt(musd)))  # BTC ~1bp, a 20M pair ~4.5bp, a 1M pair 20bp
    participation = notional / max(candle_quote_volume, 1.0)
    impact = 100.0 * participation  # 1% of a candle's volume costs about 1bp
    return spread / 2 + min(impact, 50.0)


@dataclass
class Position:
    symbol: str
    signal: str
    entry_time: str
    entry_price: float
    qty: float
    notional: float  # cash spent, fees included
    stop: float
    max_bars: int
    exit_rule: str
    context: dict
    explore: bool
    bars_held: int = 0
    mfe: float = 0.0
    mae: float = 0.0
    mark: float = 0.0
    entry_fee: float = 0.0
    entry_slip: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.signal}"


@dataclass
class Wallet:
    cash: float
    start: float
    positions: dict = field(default_factory=dict)  # key -> Position as dict
    last_bar: dict = field(default_factory=dict)  # symbol -> last processed closed bar
    curve: list = field(default_factory=list)
    peak: float = 0.0
    max_drawdown: float = 0.0
    closed: int = 0

    def open_positions(self) -> list[Position]:
        return [Position(**p) for p in self.positions.values()]

    def equity(self) -> float:
        return self.cash + sum(p["qty"] * (p["mark"] or p["entry_price"]) for p in self.positions.values())


def context_for(rs: list[Reading], bars: list[Bar]) -> dict:
    cur = rs[-1]
    closes = [b.close for b in bars]
    sma200 = ind.sma(closes, 200)[-1]
    if cur.sma50 is not None and sma200 is not None:
        regime = "uptrend" if cur.sma50 > sma200 else "downtrend"
    elif cur.sma20 is not None and cur.sma50 is not None:
        regime = "uptrend" if cur.sma20 > cur.sma50 else "downtrend"
    else:
        regime = "unknown"
    vols = [r.vol20 for r in rs if r.vol20 is not None]
    median = sorted(vols)[len(vols) // 2] if vols else None
    if cur.vol20 is None or median is None:
        bucket = "mid"
    else:
        bucket = "high" if cur.vol20 > 1.5 * median else "low" if cur.vol20 < 0.75 * median else "mid"
    atr = ind.atr([b.high for b in bars], [b.low for b in bars], closes, 14)[-1]
    atr_pct = (atr / cur.close) if atr and cur.close else 0.0
    band_pos = None
    if cur.upper is not None and cur.lower is not None and cur.upper != cur.lower:
        band_pos = (cur.close - cur.lower) / (cur.upper - cur.lower)
    return {
        "regime": regime, "vol_bucket": bucket, "vol20": cur.vol20 or 0.0, "atr_pct": atr_pct, "rsi": cur.rsi,
        "band_pos": band_pos, "macd_hist": cur.macd_hist, "hour_utc": _hour(cur.bar_time),
    }


def _hour(bar_time: str) -> int | None:
    try:
        return int(bar_time[11:13])
    except (ValueError, IndexError):
        return None


def _coin_flip(seed: str, p: float) -> bool:
    h = int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < p


class Trader:
    def __init__(self, brain: Brain, book: str = "main", interval: str = "15m", wallet: float = 1000.0, top: int = 100, lookback: int = 300,
                 fetch_bars=fetch_binance, fetch_universe=top_usdt_pairs, workers: int = 10) -> None:
        if interval not in INTERVAL_SECONDS:
            raise ValueError(f"interval must be one of {list(INTERVAL_SECONDS)}")
        self.brain, self.book, self.interval, self.top, self.lookback = brain, book, interval, top, lookback
        self.fetch_bars, self.fetch_universe, self.workers = fetch_bars, fetch_universe, workers
        self.key = f"trader:{book}"
        stored = brain.get_state(self.key)
        self.wallet = Wallet(**stored) if stored else Wallet(cash=wallet, start=wallet, peak=wallet)

    # ------------------------------------------------------------------ tick
    def tick(self, market: dict[str, list[Bar]] | None = None, universe: list[dict] | None = None) -> dict:
        """One candle for every pair. ``market`` and ``universe`` can be injected (tests, replays)."""
        if universe is None:
            universe = self.fetch_universe(self.top)
        volumes = {u["symbol"]: u["quote_volume"] for u in universe}
        symbols = list(volumes) + [p.symbol for p in self.wallet.open_positions() if p.symbol not in volumes]
        if market is None:
            market = self._fetch_all(symbols)
        events: list[dict] = []
        for symbol in symbols:
            bars = market.get(symbol)
            if not bars or len(bars) < 60:
                continue
            bars = bars[:-1]  # the last candle is still forming
            if self.wallet.last_bar.get(symbol) == bars[-1].date:
                continue
            events += self._step_symbol(symbol, bars, volumes.get(symbol, 0.0))
            self.wallet.last_bar[symbol] = bars[-1].date
        self._mark_equity(market)
        self.brain.set_state(self.key, asdict(self.wallet))
        return {"events": events, "equity": self.wallet.equity(), "cash": self.wallet.cash, "open": len(self.wallet.positions), "symbols": len(symbols)}

    def _fetch_all(self, symbols: list[str]) -> dict[str, list[Bar]]:
        out: dict[str, list[Bar]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.fetch_bars, s, self.interval, self.lookback): s for s in symbols}
            for f in as_completed(futures):
                try:
                    out[futures[f]] = f.result()
                except Exception:
                    continue
        return out

    # ---------------------------------------------------------------- symbol
    def _step_symbol(self, symbol: str, bars: list[Bar], volume_24h: float) -> list[dict]:
        rs = readings(bars)
        cur, prev = rs[-1], rs[-2]
        fired = detect(prev, cur)
        ctx = context_for(rs, bars)
        candle_qv = sum(b.quote_volume for b in bars[-20:]) / 20 or sum(b.volume * b.close for b in bars[-20:]) / 20
        events = self._manage_exits(symbol, bars[-1], cur, fired, volume_24h, candle_qv)
        for signal in fired:
            if signal in PLAYBOOK:
                ev = self._consider_entry(symbol, signal, bars[-1], ctx, volume_24h, candle_qv)
                if ev:
                    events.append(ev)
        return events

    def _consider_entry(self, symbol: str, signal: str, bar: Bar, ctx: dict, volume_24h: float, candle_qv: float) -> dict | None:
        key = f"{symbol}:{signal}"
        if key in self.wallet.positions:
            return None
        b = pm.belief(self.brain, self.book, signal, ctx["regime"], ctx["vol_bucket"])
        seed = f"{self.book}{symbol}{signal}{bar.date}"
        if b["samples"] >= pm.MIN_SAMPLES:
            if b["expectancy"] > 0:
                risk, explore = RISK_PER_TRADE, False
            elif _coin_flip(seed, RETEST_PROBABILITY):
                risk, explore = EXPLORE_RISK, True
            else:
                return {"symbol": symbol, "signal": signal, "action": "skip", "why": f"belief says {b['expectancy']:+.2%} expectancy in {ctx['regime']}/{ctx['vol_bucket']} after {b['samples']} trades"}
        elif _coin_flip(seed, EXPLORE_PROBABILITY):
            risk, explore = EXPLORE_RISK, True
        else:
            return None
        play = PLAYBOOK[signal]
        stop_pct = play["stop_atr"] * ctx["atr_pct"]
        if stop_pct <= 0:
            return None
        equity = self.wallet.equity()
        notional = min(equity * risk / stop_pct, equity * MAX_POSITION_FRACTION, self.wallet.cash)
        if notional < MIN_NOTIONAL:
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": "not enough cash for the minimum order"}
        slip = slippage_bps(notional, volume_24h, candle_qv) / 1e4
        fill = bar.close * (1 + slip)
        fee = notional * TAKER_FEE
        qty = (notional - fee) / fill
        pos = Position(
            symbol=symbol, signal=signal, entry_time=bar.date, entry_price=fill, qty=qty, notional=notional,
            stop=fill * (1 - stop_pct), max_bars=play["max_bars"], exit_rule=play["exit"],
            context={**ctx, "risk_pct": stop_pct, "p_win_believed": round(b["p_win"], 3), "samples": b["samples"]},
            explore=explore, mark=bar.close, entry_fee=fee, entry_slip=notional * slip,
        )
        self.wallet.cash -= notional
        self.wallet.positions[key] = asdict(pos)
        return {"symbol": symbol, "signal": signal, "action": "buy", "price": fill, "notional": notional, "explore": explore, "stop": pos.stop, "p_win": round(b["p_win"], 2)}

    def _manage_exits(self, symbol: str, bar: Bar, cur: Reading, fired: list[str], volume_24h: float, candle_qv: float) -> list[dict]:
        events = []
        for key, raw in list(self.wallet.positions.items()):
            if raw["symbol"] != symbol:
                continue
            pos = Position(**raw)
            pos.bars_held += 1
            pos.mfe = max(pos.mfe, bar.high / pos.entry_price - 1)
            pos.mae = min(pos.mae, bar.low / pos.entry_price - 1)
            pos.mark = bar.close
            reason, price = None, bar.close
            if bar.low <= pos.stop:
                reason, price = "stop", min(bar.open, pos.stop)
            elif self._exit_rule_hit(pos.exit_rule, cur, fired):
                reason = "signal"
            elif pos.bars_held >= pos.max_bars:
                reason = "time"
            if reason is None:
                self.wallet.positions[key] = asdict(pos)
                continue
            events.append(self._close(pos, bar, price, reason, volume_24h, candle_qv))
            del self.wallet.positions[key]
        return events

    @staticmethod
    def _exit_rule_hit(rule: str, cur: Reading, fired: list[str]) -> bool:
        if rule == "rsi_recovered":
            return cur.rsi is not None and cur.rsi >= 55
        if rule == "back_to_middle":
            return cur.sma20 is not None and cur.close >= cur.sma20
        if rule == "lost_middle":
            return cur.sma20 is not None and cur.close < cur.sma20
        if rule == "death_cross":
            return "death_cross" in fired
        if rule == "macd_bearish":
            return "macd_bearish" in fired
        return False

    def _close(self, pos: Position, bar: Bar, price: float, reason: str, volume_24h: float, candle_qv: float) -> dict:
        gross_notional = pos.qty * price
        slip = slippage_bps(gross_notional, volume_24h, candle_qv) / 1e4
        fill = price * (1 - slip)
        proceeds = pos.qty * fill
        fee = proceeds * TAKER_FEE
        proceeds -= fee
        self.wallet.cash += proceeds
        self.wallet.closed += 1
        gross_ret = price / (pos.entry_price / (1 + pos.entry_slip / pos.notional if pos.notional else 1)) - 1  # move from the pre-slippage entry
        net_ret = proceeds / pos.notional - 1
        t = pm.Trade(
            book=self.book, symbol=pos.symbol, signal=pos.signal, entry_time=pos.entry_time, entry_price=pos.entry_price,
            exit_time=bar.date, exit_price=fill, exit_reason=reason, qty=pos.qty, notional=pos.notional,
            gross_ret=gross_ret, net_ret=net_ret, pnl=proceeds - pos.notional, fees=pos.entry_fee + fee,
            slippage=pos.entry_slip + gross_notional * slip, bars_held=pos.bars_held, mfe=pos.mfe, mae=pos.mae, context=pos.context, explore=pos.explore,
        )
        findings = pm.lenses(t)
        pm.update_belief(self.brain, t)
        pm.record(self.brain, t, findings)
        cell = pm.learn_postmortem(self.brain, t, findings)
        n_signal = self.brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = ? AND signal = ?", (self.book, pos.signal)).fetchone()[0]
        if n_signal % 10 == 0:
            pm.learn_summary(self.brain, self.book, pos.signal)
        self.brain.db.commit()
        return {"symbol": pos.symbol, "signal": pos.signal, "action": "sell", "reason": reason, "price": fill, "net_ret": net_ret, "pnl": t.pnl,
                "findings": [tag for tag, _ in findings], "postmortem": cell}

    def _mark_equity(self, market: dict[str, list[Bar]]) -> None:
        for raw in self.wallet.positions.values():
            bars = market.get(raw["symbol"])
            if bars and len(bars) >= 2:
                raw["mark"] = bars[-2].close
        eq = self.wallet.equity()
        self.wallet.peak = max(self.wallet.peak, eq)
        self.wallet.max_drawdown = min(self.wallet.max_drawdown, eq / self.wallet.peak - 1)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self.wallet.curve.append((stamp, round(eq, 2)))
        if len(self.wallet.curve) > 20000:
            self.wallet.curve = self.wallet.curve[-20000:]

    # --------------------------------------------------------------- report
    def report(self) -> dict:
        w = self.wallet
        rows = self.brain.db.execute(
            "SELECT signal, COUNT(*) AS n, SUM(CASE WHEN net_ret > 0 THEN 1 ELSE 0 END) AS wins, AVG(net_ret) AS avg_ret, SUM(pnl) AS pnl,"
            " SUM(fees) AS fees, SUM(slippage) AS slip FROM trades WHERE book = ? GROUP BY signal ORDER BY n DESC", (self.book,)
        ).fetchall()
        return {
            "equity": w.equity(), "cash": w.cash, "start": w.start, "return": w.equity() / w.start - 1, "max_drawdown": w.max_drawdown,
            "open": [{"symbol": p.symbol, "signal": p.signal, "entry": p.entry_price, "mark": p.mark, "unrealized": (p.mark / p.entry_price - 1) if p.mark else 0.0,
                      "bars": p.bars_held, "stop": p.stop, "explore": p.explore} for p in w.open_positions()],
            "closed": w.closed,
            "by_signal": [{"signal": r["signal"], "trades": r["n"], "win_rate": r["wins"] / r["n"], "avg_ret": r["avg_ret"], "pnl": r["pnl"], "fees": r["fees"], "slippage": r["slip"]} for r in rows],
        }

    def beliefs(self) -> list[dict]:
        rows = self.brain.db.execute(
            "SELECT signal, regime, vol_bucket, wins, losses, sum_ret FROM beliefs WHERE book = ? ORDER BY signal, regime, vol_bucket", (self.book,)
        ).fetchall()
        return [{"signal": r["signal"], "regime": r["regime"], "vol": r["vol_bucket"], "n": r["wins"] + r["losses"], "win_rate": r["wins"] / (r["wins"] + r["losses"]),
                 "avg_ret": r["sum_ret"] / (r["wins"] + r["losses"]), "verdict": "trade" if r["sum_ret"] > 0 and r["wins"] + r["losses"] >= pm.MIN_SAMPLES else "avoid" if r["wins"] + r["losses"] >= pm.MIN_SAMPLES else "exploring"} for r in rows]

    # ----------------------------------------------------------------- loop
    def seconds_until_next_close(self, now: float | None = None) -> float:
        period = INTERVAL_SECONDS[self.interval]
        now = time.time() if now is None else now
        return period - (now % period) + 8

    def run(self, log=print, once: bool = False, sleep=time.sleep) -> None:
        while True:
            try:
                started = time.time()
                r = self.tick()
                stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                log(f"[{stamp}] {r['symbols']} pairs in {time.time() - started:.0f}s | equity {r['equity']:.2f} USDT ({r['equity'] / self.wallet.start - 1:+.2%}) cash {r['cash']:.2f} open {r['open']}")
                for e in r["events"]:
                    if e["action"] == "buy":
                        log(f"    BUY  {e['symbol']:12} {e['signal']:18} @ {e['price']:.6g}  {e['notional']:.0f} USDT  stop {e['stop']:.6g}  p(win) {e['p_win']}{'  (exploring)' if e['explore'] else ''}")
                    elif e["action"] == "sell":
                        log(f"    SELL {e['symbol']:12} {e['signal']:18} @ {e['price']:.6g}  {e['net_ret']:+.2%} ({e['pnl']:+.2f} USDT) by {e['reason']}  findings: {', '.join(e['findings'])}")
                    elif e["action"] == "skip" and "belief" in e["why"]:
                        log(f"    SKIP {e['symbol']:12} {e['signal']:18} {e['why']}")
            except Exception as exc:
                log(f"trade error: {exc}")
            if once:
                return
            sleep(self.seconds_until_next_close())
