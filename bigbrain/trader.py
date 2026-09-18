"""The brain trades: many pairs, one wallet, decisions from its own evidence.

Every candle, for every pair in the universe, the trader sees which signals
fired and in what context (trend regime, volatility bucket, RSI, band
position). It asks the belief table what that signal has done in that
context before. Positive expectancy after costs: trade at full size.
Not enough evidence yet: explore at half size, because a brain that never
tries anything never learns. Evidence says it loses: stand aside, with an
occasional small re-test so a changed market can change the belief.

Costs are modelled honestly. Spot: Binance VIP 0 fees (0.1% taker each
way, market orders). Perps (the default): USDT-margined perpetual fees
(0.02% maker / 0.05% taker) plus funding, charged at the exchange's real
rate at every funding time a position is held through. Both add slippage
from a liquidity model (half the estimated spread, wider for thin pairs,
plus impact from the order's share of the candle's volume). Stops fill at
the stop or at the open if the candle gaps through it. Spot is long only;
perps trade both sides, at one-times effective leverage so the wallet can
never be liquidated.

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
from bigbrain.ingest.market import Bar, fetch_binance, fetch_binance_futures, funding_rates, top_usdt_pairs, top_usdt_perps
from bigbrain.watch import INTERVAL_SECONDS, Reading, detect, readings

# ---- costs (Binance VIP 0) --------------------------------------------------
FEES = {
    "spot": {"maker": 0.001, "taker": 0.001},
    "perps": {"maker": 0.0002, "taker": 0.0005},
}
MAKER_FEE = FEES["spot"]["maker"]
TAKER_FEE = FEES["spot"]["taker"]
MIN_NOTIONAL = 10.0  # Binance minimum order in USDT
FUNDING_HOURS = (0, 8, 16)  # UTC funding times on Binance perps

# ---- risk ---------------------------------------------------------------------
RISK_PER_TRADE = 0.01  # fraction of equity lost if the stop is hit
EXPLORE_RISK = 0.005
MAX_POSITION_FRACTION = 0.25
MAX_OPEN_RISK = 0.10  # sum over open positions of (notional x stop distance) / equity
LEVERAGE = {"spot": 1.0, "perps": 3.0}  # margin per position = notional / leverage; gross exposure <= equity x leverage
MAINTENANCE_MARGIN = 0.005  # Binance tier-1 maintenance rate; equity below this x gross notional is a liquidation
LOG_LINES = 80  # trade feed kept in state for the dashboard
EXPLORE_PROBABILITY = 0.7  # while a (signal, context) has too little evidence
RETEST_PROBABILITY = 0.1  # occasionally re-test a context the brain believes loses

# ---- long-only playbook: how each signal is managed -----------------------------
# stop in ATR multiples, maximum bars held, and the rule that closes it early
PLAYBOOK = {
    "rsi_oversold": {"side": 1, "stop_atr": 2.0, "max_bars": 16, "exit": "rsi_recovered"},
    "below_lower_band": {"side": 1, "stop_atr": 2.0, "max_bars": 16, "exit": "back_to_middle"},
    "above_upper_band": {"side": 1, "stop_atr": 2.0, "max_bars": 48, "exit": "lost_middle"},
    "golden_cross": {"side": 1, "stop_atr": 2.5, "max_bars": 96, "exit": "death_cross"},
    "macd_bullish": {"side": 1, "stop_atr": 2.0, "max_bars": 48, "exit": "macd_bearish"},
    # shorts: only on perps
    "rsi_overbought": {"side": -1, "stop_atr": 2.0, "max_bars": 16, "exit": "rsi_cooled"},
    "death_cross": {"side": -1, "stop_atr": 2.5, "max_bars": 96, "exit": "golden_cross"},
    "macd_bearish": {"side": -1, "stop_atr": 2.0, "max_bars": 48, "exit": "macd_bullish"},
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
    side: int = 1
    funding: float = 0.0  # USDT paid (positive) or received (negative) so far
    last_funding_hour: str = ""  # "YYYY-MM-DD HH" of the last funding time applied
    margin: float = 0.0  # cash posted for this position (notional / leverage)

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

    log: list = field(default_factory=list)

    def open_positions(self) -> list[Position]:
        return [Position(**p) for p in self.positions.values()]

    @staticmethod
    def _unrealized(p: dict) -> float:
        mark = p["mark"] or p["entry_price"]
        return p.get("side", 1) * p["qty"] * (mark - p["entry_price"])

    def equity(self) -> float:
        return self.cash + sum(p.get("margin") or p["notional"] for p in self.positions.values()) + sum(self._unrealized(p) for p in self.positions.values())

    def gross_notional(self) -> float:
        return sum(p["qty"] * (p["mark"] or p["entry_price"]) for p in self.positions.values())

    def open_risk(self) -> float:
        """USDT lost if every open stop is hit."""
        return sum(p["notional"] * (p["context"].get("risk_pct") or 0.0) for p in self.positions.values())


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
                 market: str = "perps", fetch_bars=None, fetch_universe=None, fetch_funding=None, workers: int = 10) -> None:
        if interval not in INTERVAL_SECONDS:
            raise ValueError(f"interval must be one of {list(INTERVAL_SECONDS)}")
        if market not in FEES:
            raise ValueError("market must be 'spot' or 'perps'")
        self.brain, self.book, self.interval, self.top, self.lookback, self.workers = brain, book, interval, top, lookback, workers
        self.key = f"trader:{book}"
        stored = brain.get_state(self.key)
        self.market = stored.get("market", market) if stored else market  # a book keeps the market it was opened on
        self.fees = FEES[self.market]
        self.leverage = LEVERAGE[self.market]
        self.fetch_bars = fetch_bars or (fetch_binance_futures if self.market == "perps" else fetch_binance)
        self.fetch_universe = fetch_universe or (top_usdt_perps if self.market == "perps" else top_usdt_pairs)
        self.fetch_funding = fetch_funding or (funding_rates if self.market == "perps" else (lambda: {}))
        self.funding: dict[str, dict] = {}
        if stored:
            stored.pop("market", None)
            self.wallet = Wallet(**stored)
        else:
            self.wallet = Wallet(cash=wallet, start=wallet, peak=wallet)

    # ------------------------------------------------------------------ tick
    def tick(self, market: dict[str, list[Bar]] | None = None, universe: list[dict] | None = None) -> dict:
        """One candle for every pair. ``market`` and ``universe`` can be injected (tests, replays)."""
        if universe is None:
            universe = self.fetch_universe(self.top)
        volumes = {u["symbol"]: u["quote_volume"] for u in universe}
        symbols = list(volumes) + [p.symbol for p in self.wallet.open_positions() if p.symbol not in volumes]
        if market is None:
            market = self._fetch_all(symbols)
        if self.market == "perps" and self.wallet.positions:
            try:
                self.funding = self.fetch_funding()
            except Exception:
                self.funding = {}
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
        self.brain.set_state(self.key, {**asdict(self.wallet), "market": self.market})
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
            play = PLAYBOOK.get(signal)
            if play is None or (play["side"] == -1 and self.market != "perps"):
                continue
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
        risk_room = equity * MAX_OPEN_RISK - self.wallet.open_risk()
        if risk_room <= 0:
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": f"open risk already at {MAX_OPEN_RISK:.0%} of equity"}
        exposure_room = equity * self.leverage - self.wallet.gross_notional()
        notional = min(equity * risk / stop_pct, equity * MAX_POSITION_FRACTION, risk_room / stop_pct, exposure_room, self.wallet.cash * self.leverage * 0.98)
        if notional < MIN_NOTIONAL:
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": "no room: cash, exposure or risk budget"}
        side = play["side"]
        slip = slippage_bps(notional, volume_24h, candle_qv) / 1e4
        fill = bar.close * (1 + side * slip)  # buying lifts the offer, selling hits the bid
        fee = notional * self.fees["taker"]
        qty = notional / fill
        margin = notional / self.leverage
        pos = Position(
            symbol=symbol, signal=signal, entry_time=bar.date, entry_price=fill, qty=qty, notional=notional,
            stop=fill * (1 - side * stop_pct), max_bars=play["max_bars"], exit_rule=play["exit"],
            context={**ctx, "risk_pct": stop_pct, "p_win_believed": round(b["p_win"], 3), "samples": b["samples"], "market": self.market},
            explore=explore, mark=bar.close, entry_fee=fee, entry_slip=notional * slip, side=side, last_funding_hour=bar.date[:13], margin=margin,
        )
        self.wallet.cash -= margin + fee
        self.wallet.positions[key] = asdict(pos)
        return {"symbol": symbol, "signal": signal, "action": "buy" if side == 1 else "short", "price": fill, "notional": notional, "explore": explore, "stop": pos.stop, "p_win": round(b["p_win"], 2)}

    def _manage_exits(self, symbol: str, bar: Bar, cur: Reading, fired: list[str], volume_24h: float, candle_qv: float) -> list[dict]:
        events = []
        for key, raw in list(self.wallet.positions.items()):
            if raw["symbol"] != symbol:
                continue
            pos = Position(**raw)
            pos.bars_held += 1
            if pos.side == 1:
                pos.mfe = max(pos.mfe, bar.high / pos.entry_price - 1)
                pos.mae = min(pos.mae, bar.low / pos.entry_price - 1)
            else:
                pos.mfe = max(pos.mfe, 1 - bar.low / pos.entry_price)
                pos.mae = min(pos.mae, 1 - bar.high / pos.entry_price)
            pos.mark = bar.close
            self._apply_funding(pos, bar)
            reason, price = None, bar.close
            if pos.side == 1 and bar.low <= pos.stop:
                reason, price = "stop", min(bar.open, pos.stop)
            elif pos.side == -1 and bar.high >= pos.stop:
                reason, price = "stop", max(bar.open, pos.stop)
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

    def _apply_funding(self, pos: Position, bar: Bar) -> None:
        """Charge or credit funding for every funding time between the last processed candle and this one."""
        if self.market != "perps":
            return
        hour_key = bar.date[:13]  # "YYYY-MM-DD HH"
        if hour_key == pos.last_funding_hour:
            return
        try:
            hour = int(bar.date[11:13])
            minute = int(bar.date[14:16])
        except (ValueError, IndexError):
            return
        pos.last_funding_hour = hour_key
        if hour in FUNDING_HOURS and minute == 0:
            rate = self.funding.get(pos.symbol, {}).get("rate", 0.0)
            charge = pos.side * rate * pos.qty * bar.close  # longs pay when the rate is positive
            pos.funding += charge
            self.wallet.cash -= charge

    @staticmethod
    def _exit_rule_hit(rule: str, cur: Reading, fired: list[str]) -> bool:
        if rule == "rsi_recovered":
            return cur.rsi is not None and cur.rsi >= 55
        if rule == "rsi_cooled":
            return cur.rsi is not None and cur.rsi <= 45
        if rule == "golden_cross":
            return "golden_cross" in fired
        if rule == "macd_bullish":
            return "macd_bullish" in fired
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
        fill = price * (1 - pos.side * slip)  # a long sells into the bid, a short buys back at the offer
        fee = pos.qty * fill * self.fees["taker"]
        margin = pos.margin or pos.notional
        trade_pnl = pos.side * pos.qty * (fill - pos.entry_price)
        self.wallet.cash += margin + trade_pnl - fee
        self.wallet.closed += 1
        raw_entry = pos.entry_price / (1 + pos.side * pos.entry_slip / pos.notional) if pos.notional else pos.entry_price
        gross_ret = pos.side * (price / raw_entry - 1)  # the move from the pre-slippage entry, signed by side
        pnl = trade_pnl - fee - pos.entry_fee - pos.funding  # everything the trade cost or made, funding included
        net_ret = pnl / pos.notional
        t = pm.Trade(
            book=self.book, symbol=pos.symbol, signal=pos.signal, entry_time=pos.entry_time, entry_price=pos.entry_price,
            exit_time=bar.date, exit_price=fill, exit_reason=reason, qty=pos.qty, notional=pos.notional,
            gross_ret=gross_ret, net_ret=net_ret, pnl=pnl, fees=pos.entry_fee + fee,
            slippage=pos.entry_slip + gross_notional * slip, bars_held=pos.bars_held, mfe=pos.mfe, mae=pos.mae, context=pos.context, explore=pos.explore,
            side=pos.side, funding=pos.funding,
        )
        findings = pm.lenses(t)
        pm.update_belief(self.brain, t)
        pm.record(self.brain, t, findings)
        cell = pm.learn_postmortem(self.brain, t, findings)
        n_signal = self.brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = ? AND signal = ?", (self.book, pos.signal)).fetchone()[0]
        if n_signal % 10 == 0:
            pm.learn_summary(self.brain, self.book, pos.signal)
        self.brain.db.commit()
        return {"symbol": pos.symbol, "signal": pos.signal, "action": "sell" if pos.side == 1 else "cover", "reason": reason, "price": fill, "net_ret": net_ret,
                "pnl": t.pnl, "funding": pos.funding, "findings": [tag for tag, _ in findings], "postmortem": cell}

    def _mark_equity(self, market: dict[str, list[Bar]]) -> None:
        for raw in self.wallet.positions.values():
            bars = market.get(raw["symbol"])
            if bars and len(bars) >= 2:
                raw["mark"] = bars[-2].close
        eq = self.wallet.equity()
        if self.wallet.positions and eq < MAINTENANCE_MARGIN * self.wallet.gross_notional():
            self._liquidate(market)
            eq = self.wallet.equity()
        self.wallet.peak = max(self.wallet.peak, eq)
        self.wallet.max_drawdown = min(self.wallet.max_drawdown, eq / self.wallet.peak - 1)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self.wallet.curve.append((stamp, round(eq, 2)))
        if len(self.wallet.curve) > 20000:
            self.wallet.curve = self.wallet.curve[-20000:]

    def _liquidate(self, market: dict[str, list[Bar]]) -> None:
        """Equity fell below maintenance margin: the exchange closes everything at market."""
        for key, raw in list(self.wallet.positions.items()):
            bars = market.get(raw["symbol"])
            if not bars:
                continue
            bar = bars[-2] if len(bars) >= 2 else bars[-1]
            self._close(Position(**raw), bar, bar.close, "liquidation", 0.0, max(bar.quote_volume, 1.0))
            del self.wallet.positions[key]
        self.brain._journal("liquidation", f"book {self.book}")

    def _log(self, line: str) -> None:
        self.wallet.log.append(line)
        if len(self.wallet.log) > LOG_LINES:
            self.wallet.log = self.wallet.log[-LOG_LINES:]

    # --------------------------------------------------------------- report
    def report(self) -> dict:
        w = self.wallet
        rows = self.brain.db.execute(
            "SELECT signal, COUNT(*) AS n, SUM(CASE WHEN net_ret > 0 THEN 1 ELSE 0 END) AS wins, AVG(net_ret) AS avg_ret, SUM(pnl) AS pnl,"
            " SUM(fees) AS fees, SUM(slippage) AS slip, SUM(funding) AS funding FROM trades WHERE book = ? GROUP BY signal ORDER BY n DESC", (self.book,)
        ).fetchall()
        return {
            "market": self.market, "equity": w.equity(), "cash": w.cash, "start": w.start, "return": w.equity() / w.start - 1, "max_drawdown": w.max_drawdown,
            "gross_notional": w.gross_notional(), "open_risk": w.open_risk(), "leverage": self.leverage,
            "open": [{"symbol": p.symbol, "signal": p.signal, "side": "long" if p.side == 1 else "short", "entry": p.entry_price, "mark": p.mark,
                      "unrealized": (p.side * (p.mark / p.entry_price - 1)) if p.mark else 0.0, "bars": p.bars_held, "stop": p.stop, "explore": p.explore, "funding": p.funding}
                     for p in w.open_positions()],
            "closed": w.closed,
            "by_signal": [{"signal": r["signal"], "trades": r["n"], "win_rate": r["wins"] / r["n"], "avg_ret": r["avg_ret"], "pnl": r["pnl"], "fees": r["fees"], "slippage": r["slip"], "funding": r["funding"] or 0.0} for r in rows],
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
        def out(line: str) -> None:
            log(line)
            self._log(line)

        while True:
            try:
                started = time.time()
                r = self.tick()
                stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                out(f"[{stamp}] {r['symbols']} pairs in {time.time() - started:.0f}s | equity {r['equity']:.2f} USDT ({r['equity'] / self.wallet.start - 1:+.2%}) cash {r['cash']:.2f} open {r['open']}")
                for e in r["events"]:
                    if e["action"] in ("buy", "short"):
                        out(f"    {e['action'].upper():5} {e['symbol']:12} {e['signal']:18} @ {e['price']:.6g}  {e['notional']:.0f} USDT  stop {e['stop']:.6g}  p(win) {e['p_win']}{'  (exploring)' if e['explore'] else ''}")
                    elif e["action"] in ("sell", "cover"):
                        fund = f" funding {e['funding']:+.2f}" if e.get("funding") else ""
                        out(f"    {e['action'].upper():5} {e['symbol']:12} {e['signal']:18} @ {e['price']:.6g}  {e['net_ret']:+.2%} ({e['pnl']:+.2f} USDT{fund}) by {e['reason']}  findings: {', '.join(e['findings'])}")
                    elif e["action"] == "skip" and "belief" in e["why"]:
                        out(f"    SKIP {e['symbol']:12} {e['signal']:18} {e['why']}")
                self.brain.set_state(self.key, {**asdict(self.wallet), "market": self.market})
            except Exception as exc:
                out(f"trade error: {exc}")
            if once:
                return
            sleep(self.seconds_until_next_close())
