"""The brain trades: many pairs, one wallet, decisions from its own evidence.

Every closed candle, for every pair in the universe, the trader sees which
signals fired and in what context (trend regime, volatility bucket, RSI,
band position). It asks the belief table what that signal has done in that
context before, and the verdict sets the size: enough evidence over enough
separate days that the mean minus its standard error is positive, full
size; positive but not proven, half; too little evidence, explore at half
size some of the time; evidence says it loses, stand aside with an
occasional small re-test.

Execution is strictly chronological. A signal is known only after its
candle closes, so nothing ever fills at a price that has already passed:

* live: the order fills at the current bid/ask (a timestamped book ticker
  fetched at decision time; buys lift the ask, sells hit the bid) plus
  market impact. If the quote is missing or stale the entry is skipped and
  an exit is queued.
* replay (no quotes): orders queue and fill at the next candle's open.
* stops, trailing stops and targets are resting orders: they fill at their
  level, or at the open when a candle gaps through it.

Costs are modelled the same way for every fill and every exit variant:
Binance VIP 0 fees (spot 0.1% taker; perps 0.02% maker / 0.05% taker),
funding on perps at the exchange's real rate for every funding time a
position is held through, and slippage from a liquidity model (half the
estimated spread for fills without a live quote, plus impact from the
order's share of the candle's volume). Perps post margin of a third of the
notional, so gross exposure can reach three times equity; a liquidation
check runs every candle.

A ``frozen`` book runs the same signals, rules, sizing and costs with no
learning at all: fixed size, the signal's own exit, beliefs ignored. Run
beside the adaptive book on identical inputs (``bigbrain trade
--with-frozen``) it is the baseline the learning must beat.

Every closed trade goes through ``bigbrain.postmortem``; the symbol's whole
step (fills, exits, trade record, belief update, saved wallet) is one
database transaction, so a crash can never leave the money and the evidence
out of step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bigbrain import exits
from bigbrain import postmortem as pm
from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import BINANCE_HOSTS, FUTURES_HOST, Bar, fetch_binance, fetch_binance_futures, funding_rates, top_usdt_pairs, top_usdt_perps
from bigbrain.net import http_get
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
QUOTE_MAX_AGE = 60.0  # seconds; an older quote is not a price anyone can trade at now

# ---- risk ---------------------------------------------------------------------
RISK_PER_TRADE = 0.01  # fraction of equity lost if the stop is hit
EXPLORE_RISK = 0.005
MAX_POSITION_FRACTION = 0.25
MAX_OPEN_RISK = 0.10  # sum over open positions of (notional x stop distance) / equity
MIN_STOP_PCT = 0.005  # a stop closer than this sits inside the spread and noise on quiet pairs
MIN_STOP_COST_MULTIPLE = 6.0  # and never closer than this many round-trip costs
LEVERAGE = {"spot": 1.0, "perps": 3.0}  # margin per position = notional / leverage; gross exposure <= equity x leverage
MAINTENANCE_MARGIN = 0.005  # Binance tier-1 maintenance rate; equity below this x gross notional is a liquidation
LOG_LINES = 80  # trade feed kept in state for the dashboard
EXPLORE_PROBABILITY = 0.7  # while a (signal, context) has too little evidence
RETEST_PROBABILITY = 0.1  # occasionally re-test a context the brain believes loses
PATH_LIMIT = 200  # bars of path kept per position for exit learning

# ---- playbook: how each signal is managed ----------------------------------------
# side, stop in ATR multiples, maximum bars held, and the rule that closes it early
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

_TRADER_CMD = re.compile(r"(^|[\s/])bigbrain(\.cli)?\s+trade(\s|$)")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _shift(bar_date: str, seconds: int) -> str:
    """The candle date ``seconds`` after ``bar_date`` ("YYYY-MM-DD HH:MM"), or "" if the date is not a real time."""
    try:
        t = datetime.strptime(bar_date[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return ""
    return (t + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---- market data helpers ----------------------------------------------------------
def live_prices(market: str) -> dict[str, float]:
    """Last traded price for every symbol in one request (for marking, not for fills)."""
    url = f"{FUTURES_HOST}/fapi/v1/ticker/price" if market == "perps" else f"{BINANCE_HOSTS[0]}/api/v3/ticker/price"
    rows = json.loads(http_get(url, headers={"Accept": "application/json"}).decode("utf-8"))
    return {r["symbol"]: float(r["price"]) for r in rows}


def live_quotes(market: str) -> dict[str, dict]:
    """Timestamped best bid and ask for every symbol: ``{symbol: {"bid", "ask", "at"}}`` with ``at`` in epoch seconds.

    Perps carry the exchange's own update time; spot does not, so the fetch time is used."""
    url = f"{FUTURES_HOST}/fapi/v1/ticker/bookTicker" if market == "perps" else f"{BINANCE_HOSTS[0]}/api/v3/ticker/bookTicker"
    rows = json.loads(http_get(url, headers={"Accept": "application/json"}).decode("utf-8"))
    now = time.time()
    out: dict[str, dict] = {}
    for r in rows:
        try:
            bid, ask = float(r["bidPrice"]), float(r["askPrice"])
        except (KeyError, ValueError):
            continue
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        at = min(float(r["time"]) / 1000.0, now) if r.get("time") else now
        out[r["symbol"]] = {"bid": bid, "ask": ask, "at": at}
    return out


def spread_bps(quote_volume_24h: float) -> float:
    """Estimated full bid-ask spread in basis points from 24h volume: BTC about 1bp, a 20M pair about 4.5bp, a 1M pair 20bp."""
    musd = max(quote_volume_24h, 1.0) / 1e6
    return min(30.0, max(1.0, 20.0 / math.sqrt(musd)))


def impact_bps(notional: float, candle_quote_volume: float) -> float:
    """Market impact in basis points: 1% of a candle's volume costs about 1bp, capped at 50bp."""
    participation = notional / max(candle_quote_volume, 1.0)
    return min(100.0 * participation, 50.0)


def slippage_bps(notional: float, quote_volume_24h: float, candle_quote_volume: float) -> float:
    """Slippage for a fill without a live quote: half the estimated spread plus impact."""
    return spread_bps(quote_volume_24h) / 2 + impact_bps(notional, candle_quote_volume)


# ---- state ---------------------------------------------------------------------------
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
    path: list = field(default_factory=list)  # [(open, high, low, close)] per bar since entry, for exit learning
    fund_path: list = field(default_factory=list)  # cumulative funding as a fraction of notional, one value per path bar
    exit_variant: str = "rule"  # the exit policy this position runs under
    exit_armed: bool = False
    exit_best: float = 0.0
    exit_target: float | None = None
    entered_at_open: bool = False  # filled at its entry candle's open, so that candle is managed too
    fill_basis: str = ""  # quote | next_open
    pending_exit: str = ""  # an exit decided at a close, waiting for the next candle's open
    rule_track: dict | None = None  # under a learned exit: how the signal's own rule is doing on this same path

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
    shadows: dict = field(default_factory=dict)  # variant-closed positions still tracked until the rule would have exited
    pending: dict = field(default_factory=dict)  # key -> entry order queued for the next candle's open (replay)

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

    def symbols(self) -> set[str]:
        """Every symbol the wallet still has business in: positions, queued orders, shadows."""
        return ({p["symbol"] for p in self.positions.values()} | {o["symbol"] for o in self.pending.values()}
                | {s["symbol"] for s in self.shadows.values()})


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
                 market: str = "perps", fetch_bars=None, fetch_universe=None, fetch_funding=None, fetch_prices=None, fetch_quotes=None,
                 workers: int = 10, frozen: bool = False) -> None:
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
        self.fetch_prices = fetch_prices or (lambda: live_prices(self.market))
        self.fetch_quotes = fetch_quotes or (lambda: live_quotes(self.market))
        self.funding: dict[str, dict] = {}
        self.quotes: dict[str, dict] = {}  # timestamped bid/ask at decision time, refreshed every live tick
        self.live = False  # True while the current tick runs on freshly fetched data (fills at quotes, never at a passed price)
        self._catch_up = False  # True while replaying missed candles: no quote may be used for a decision that belongs to the past
        self.frozen = stored.get("frozen", frozen) if stored else frozen  # a frozen book never adapts: fixed size, rule exits, no beliefs
        self.wallet = self._load(stored) if stored else Wallet(cash=wallet, start=wallet, peak=wallet)

    @staticmethod
    def _load(stored: dict) -> Wallet:
        stored = dict(stored)
        for extra in ("market", "interval", "last_tick", "frozen"):
            stored.pop(extra, None)
        return Wallet(**stored)

    def _reload(self) -> None:
        """Drop the in-memory wallet for the one in the database (after a rolled-back transaction)."""
        stored = self.brain.get_state(self.key)
        if stored:
            self.wallet = self._load(stored)

    # ------------------------------------------------------------------ tick
    def tick(self, market: dict[str, list[Bar]] | None = None, universe: list[dict] | None = None, quotes: dict[str, dict] | None = None,
             funding: dict[str, dict] | None = None, live: bool | None = None) -> dict:
        """One candle for every pair.

        ``market``, ``universe``, ``quotes`` and ``funding`` can be injected (tests, replays, a companion book fed the
        same inputs). ``live`` says whether fills may use the quotes; by default a tick is live only when it fetched
        its own market data. Without live quotes every order queues and fills at the next candle's open."""
        live = (market is None) if live is None else live
        if universe is None:
            universe = self.fetch_universe(self.top)
        volumes = {u["symbol"]: u["quote_volume"] for u in universe}
        symbols = list(volumes) + sorted(self.wallet.symbols() - set(volumes))
        if market is None:
            market = self._fetch_all(symbols)
        if funding is None:
            funding = {}
            if self.market == "perps" and (self.wallet.positions or self.wallet.shadows):
                try:
                    funding = self.fetch_funding()
                except Exception:
                    funding = {}
        self.funding = funding or {}
        if live and quotes is None:
            try:
                quotes = self.fetch_quotes() or {}
            except Exception:
                quotes = {}
        self.quotes, self.live = (quotes or {}), live
        events: list[dict] = []
        for symbol in symbols:
            bars = market.get(symbol)
            if not bars or len(bars) < 60:
                continue
            bars = bars[:-1]  # the last candle is still forming
            if self.wallet.last_bar.get(symbol) == bars[-1].date:
                continue
            try:
                with self.brain.batch():  # fills, exits, trade records, beliefs and the saved wallet: all or nothing
                    events += self._step_symbol(symbol, bars, volumes.get(symbol, 0.0))
                    self.wallet.last_bar[symbol] = bars[-1].date
                    self._save()
            except Exception:
                self._reload()  # the database rolled back; the wallet in memory must not run ahead of it
                raise
        self._mark_equity(market)
        self._save(last_tick=_now())
        self._upkeep()
        return {"events": events, "equity": self.wallet.equity(), "cash": self.wallet.cash, "open": len(self.wallet.positions),
                "pending": len(self.wallet.pending), "symbols": len(symbols)}

    PRUNE_AFTER_DAYS = 60

    def _upkeep(self) -> None:
        """Once a day: drop post-mortem cells older than PRUNE_AFTER_DAYS and compact the file."""
        last = self.brain.get_state("upkeep:last", "")
        today = _now()[:10]
        if last == today:
            return
        pruned = self.brain.prune("postmortem", self.PRUNE_AFTER_DAYS)
        self.brain.checkpoint()
        self.brain.set_state("upkeep:last", today)
        if pruned:
            self._log(f"    upkeep: forgot {pruned} post-mortems older than {self.PRUNE_AFTER_DAYS} days (their lessons live on in beliefs)")

    def _save(self, last_tick: str | None = None) -> None:
        state = {**asdict(self.wallet), "market": self.market, "interval": self.interval, "frozen": self.frozen}
        prev = self.brain.get_state(self.key) or {}
        state["last_tick"] = prev.get("last_tick", "") if last_tick is None else last_tick  # "" clears it (reset)
        self.brain.set_state(self.key, state)

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

    def fetch_inputs(self, extra_symbols: set[str] | None = None) -> tuple[list[dict], dict[str, list[Bar]], dict[str, dict], dict[str, dict]]:
        """Everything one live tick needs, fetched once: (universe, market, quotes, funding). Shared with companion books."""
        universe = self.fetch_universe(self.top)
        symbols = [u["symbol"] for u in universe]
        held = self.wallet.symbols() | (extra_symbols or set())
        symbols += sorted(held - set(symbols))
        market = self._fetch_all(symbols)
        funding: dict[str, dict] = {}
        if self.market == "perps" and held:
            try:
                funding = self.fetch_funding()
            except Exception:
                funding = {}
        try:
            quotes = self.fetch_quotes() or {}
        except Exception:
            quotes = {}
        return universe, market, quotes, funding

    # ---------------------------------------------------------------- symbol
    def _has_work(self, symbol: str) -> bool:
        return symbol in self.wallet.symbols()

    def _step_symbol(self, symbol: str, bars: list[Bar], volume_24h: float) -> list[dict]:
        rs = readings(bars)
        candle_qv = sum(b.quote_volume for b in bars[-20:]) / 20 or sum(b.volume * b.close for b in bars[-20:]) / 20
        events: list[dict] = []
        # Catch up: if candles closed while the trader was not running (sleep, restart, outage), queued
        # orders fill and open positions are managed through every missed candle in order, so stops that
        # were touched in the gap fire at the right price and holding periods count real candles.
        # No new decisions are taken on missed candles: their signals were never seen at the time.
        last_done = self.wallet.last_bar.get(symbol)
        dates = [b.date for b in bars]
        start = dates.index(last_done) + 1 if last_done in dates else len(bars) - 1
        for i in range(start, len(bars) - 1):
            if not self._has_work(symbol):
                break
            fired_i = detect(rs[i - 1], rs[i]) if i >= 1 else []
            events += self._process_bar(symbol, bars[i], rs[i], fired_i, volume_24h, candle_qv, catch_up=True)
        cur, prev = rs[-1], rs[-2]
        fired = detect(prev, cur)
        events += self._process_bar(symbol, bars[-1], cur, fired, volume_24h, candle_qv)
        ctx = context_for(rs, bars)
        for signal in fired:
            play = PLAYBOOK.get(signal)
            if play is None or (play["side"] == -1 and self.market != "perps"):
                continue
            ev = self._consider_entry(symbol, signal, bars[-1], ctx, volume_24h, candle_qv)
            if ev:
                events.append(ev)
        return events

    def _process_bar(self, symbol: str, bar: Bar, cur: Reading, fired: list[str], volume_24h: float, candle_qv: float, catch_up: bool = False) -> list[dict]:
        """One closed candle for one symbol, in market order: queued orders fill at the open, then the candle plays out.

        While catching up on missed candles nothing may fill at today's quote: those decisions belong
        to the past, so every market order queues and fills at the following candle's open."""
        self._catch_up = catch_up
        try:
            events = self._fill_pending(symbol, bar, volume_24h, candle_qv, catch_up)
            events += self._manage_exits(symbol, bar, cur, fired, volume_24h, candle_qv, catch_up)
            self._advance_shadows(symbol, bar, cur, fired, volume_24h, candle_qv)
        finally:
            self._catch_up = False
        return events

    # ---------------------------------------------------------------- quotes
    def _fresh_quote(self, symbol: str) -> dict | None:
        """The live bid/ask if it exists and is recent enough to trade at; otherwise None."""
        if not self.live or self._catch_up:
            return None
        q = self.quotes.get(symbol)
        if not q or q.get("bid", 0) <= 0 or q.get("ask", 0) <= 0 or q["ask"] < q["bid"]:
            return None
        if time.time() - float(q.get("at", 0)) > QUOTE_MAX_AGE:
            return None
        return q

    # --------------------------------------------------------------- entries
    def _consider_entry(self, symbol: str, signal: str, bar: Bar, ctx: dict, volume_24h: float, candle_qv: float) -> dict | None:
        key = f"{symbol}:{signal}"
        if key in self.wallet.positions or key in self.wallet.pending:
            return None
        side = PLAYBOOK[signal]["side"]
        opposite = [p for p in list(self.wallet.positions.values()) + list(self.wallet.pending.values()) if p["symbol"] == symbol and p.get("side", 1) != side]
        if opposite:
            held = ", ".join(p["signal"] for p in opposite)
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": f"conflict: already {'short' if side == 1 else 'long'} {symbol} via {held}; hedging one pair only pays double costs"}
        seed = f"{self.book}{symbol}{signal}{bar.date}"
        if self.frozen:
            b = {"p_win": 0.5, "expectancy": 0.0, "stderr": float("inf"), "samples": 0, "blocks": 0}
            risk, explore = RISK_PER_TRADE, False  # fixed rules, fixed size: the baseline the adaptive book must beat
        else:
            b = pm.belief(self.brain, self.book, signal, ctx["regime"], ctx["vol_bucket"])
            call = pm.verdict(b)
            if call == "full":
                risk, explore = RISK_PER_TRADE, False
            elif call == "half":
                risk, explore = EXPLORE_RISK, False
            elif call == "avoid":
                if _coin_flip(seed, RETEST_PROBABILITY):
                    risk, explore = EXPLORE_RISK, True
                else:
                    return {"symbol": symbol, "signal": signal, "action": "skip",
                            "why": f"belief: expectancy {b['expectancy']:+.2%} (se {b['stderr']:.2%}) over {b['samples']} trades on {b['blocks']} days in {ctx['regime']}/{ctx['vol_bucket']}"}
            elif _coin_flip(seed, EXPLORE_PROBABILITY):
                risk, explore = EXPLORE_RISK, True
            else:
                return None
        play = PLAYBOOK[signal]
        equity = self.wallet.equity()
        round_trip = 2 * self.fees["taker"] + 2 * slippage_bps(equity * MAX_POSITION_FRACTION, volume_24h, candle_qv) / 1e4
        stop_pct = max(play["stop_atr"] * ctx["atr_pct"], MIN_STOP_PCT, MIN_STOP_COST_MULTIPLE * round_trip)
        if stop_pct <= 0:
            return None
        risk_room = equity * MAX_OPEN_RISK - self.wallet.open_risk() - sum(o["notional"] * o["stop_pct"] for o in self.wallet.pending.values())
        if risk_room <= 0:
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": f"open risk already at {MAX_OPEN_RISK:.0%} of equity"}
        exposure_room = equity * self.leverage - self.wallet.gross_notional() - sum(o["notional"] for o in self.wallet.pending.values())
        cash_room = (self.wallet.cash - sum(o["notional"] / self.leverage for o in self.wallet.pending.values())) * self.leverage * 0.98
        notional = min(equity * risk / stop_pct, equity * MAX_POSITION_FRACTION, risk_room / stop_pct, exposure_room, cash_room)
        if notional < MIN_NOTIONAL:
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": "no room: cash, exposure or risk budget"}
        variant = "rule" if self.frozen else exits.current_policy(self.brain, self.book).get(signal, "rule")
        order = {
            "symbol": symbol, "signal": signal, "side": side, "notional": notional, "stop_pct": stop_pct, "explore": explore, "variant": variant,
            "max_bars": play["max_bars"], "exit_rule": play["exit"], "placed": bar.date, "signal_close": bar.close, "p_win": round(b["p_win"], 3), "samples": b["samples"],
            "ctx": ctx,
        }
        q = self._fresh_quote(symbol)
        if q is not None:
            # live: a market order now, at the side of the book it takes, plus impact
            price = q["ask"] if side == 1 else q["bid"]
            return self._fill_entry(order, price, "quote", volume_24h, candle_qv, entry_time=bar.date, entered_at_open=False,
                                    last_funding_hour=_shift(bar.date, INTERVAL_SECONDS[self.interval])[:13] or bar.date[:13])
        if self.live:
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": "no fresh quote at decision time; an entry is never filled at a price that has already passed"}
        # replay: the order waits for the next candle and fills at its open
        self.wallet.pending[key] = order
        return {"symbol": symbol, "signal": signal, "action": "queued", "notional": notional, "explore": explore}

    def _fill_entry(self, order: dict, price: float, basis: str, volume_24h: float, candle_qv: float, entry_time: str, entered_at_open: bool,
                    last_funding_hour: str = "") -> dict:
        side, notional, signal, symbol = order["side"], order["notional"], order["signal"], order["symbol"]
        slip = (impact_bps(notional, candle_qv) if basis == "quote" else slippage_bps(notional, volume_24h, candle_qv)) / 1e4
        fill = price * (1 + side * slip)  # buying lifts the offer, selling hits the bid
        fee = notional * self.fees["taker"]
        margin = notional / self.leverage
        if margin + fee > self.wallet.cash + 1e-9:
            return {"symbol": symbol, "signal": signal, "action": "skip", "why": "order cancelled: no cash left when it reached the market"}
        qty = notional / fill
        stop_pct, variant = order["stop_pct"], order["variant"]
        est = exits.init_state(variant, fill, side, stop_pct)
        # A learned exit is an overlay on the signal's own rule. The rule itself is tracked from the first
        # candle, so the 'rule' baseline every variant is scored against is what the rule would really have done.
        track = None if variant == "rule" else self._new_track(fill * (1 - side * stop_pct), last_funding_hour)
        pos = Position(
            symbol=symbol, signal=signal, entry_time=entry_time, entry_price=fill, qty=qty, notional=notional,
            stop=fill * (1 - side * stop_pct), max_bars=order["max_bars"], exit_rule=order["exit_rule"],
            context={**order["ctx"], "risk_pct": stop_pct, "p_win_believed": order["p_win"], "samples": order["samples"], "market": self.market,
                     "exit_variant": variant, "signal_close": order["signal_close"], "decided": order["placed"], "fill_basis": basis},
            explore=order["explore"], mark=price, entry_fee=fee, entry_slip=notional * slip, side=side, last_funding_hour=last_funding_hour, margin=margin,
            exit_variant=variant, exit_best=est.best, exit_target=est.target, entered_at_open=entered_at_open, fill_basis=basis, rule_track=track,
        )
        self.wallet.cash -= margin + fee
        self.wallet.positions[pos.key] = asdict(pos)
        return {"symbol": symbol, "signal": signal, "action": "buy" if side == 1 else "short", "price": fill, "notional": notional, "explore": order["explore"],
                "stop": pos.stop, "p_win": round(order["p_win"], 2), "basis": basis}

    def _fill_pending(self, symbol: str, bar: Bar, volume_24h: float, candle_qv: float, catch_up: bool) -> list[dict]:
        """Orders queued at the previous close fill at this candle's open: exits first, then entries."""
        events: list[dict] = []
        for key, raw in list(self.wallet.positions.items()):
            if raw["symbol"] != symbol or not raw.get("pending_exit"):
                continue
            pos = Position(**raw)
            self._apply_funding(pos, bar)  # still held at this candle's open, so a funding time here is charged
            if pos.rule_track:
                self._advance_track(pos.rule_track, pos, bar, None, [], len(pos.path))  # the rule's own order fills at this open too
            ev = self._close(pos, bar, bar.open, pos.pending_exit, volume_24h, candle_qv, basis="next_open")
            if catch_up:
                ev["catch_up"] = True
            events.append(ev)
            del self.wallet.positions[key]
        for key, order in list(self.wallet.pending.items()):
            if order["symbol"] != symbol or order["placed"] >= bar.date:
                continue
            del self.wallet.pending[key]
            ev = self._fill_entry(order, bar.open, "next_open", volume_24h, candle_qv, entry_time=bar.date, entered_at_open=True)
            if catch_up:
                ev["catch_up"] = True
            events.append(ev)
        return events

    # ----------------------------------------------------------------- exits
    def _manage_exits(self, symbol: str, bar: Bar, cur: Reading, fired: list[str], volume_24h: float, candle_qv: float, catch_up: bool = False) -> list[dict]:
        events = []
        for key, raw in list(self.wallet.positions.items()):
            if raw["symbol"] != symbol or raw.get("pending_exit"):
                continue
            pos = Position(**raw)
            if pos.entry_time > bar.date or (pos.entry_time == bar.date and not pos.entered_at_open):
                continue  # not in the market during this candle
            pos.bars_held += 1
            if pos.side == 1:
                pos.mfe = max(pos.mfe, bar.high / pos.entry_price - 1)
                pos.mae = min(pos.mae, bar.low / pos.entry_price - 1)
            else:
                pos.mfe = max(pos.mfe, 1 - bar.low / pos.entry_price)
                pos.mae = min(pos.mae, 1 - bar.high / pos.entry_price)
            pos.mark = bar.close
            self._apply_funding(pos, bar)
            if pos.rule_track:
                self._advance_track(pos.rule_track, pos, bar, cur, fired, len(pos.path))
            pos.path.append((bar.open, bar.high, bar.low, bar.close))
            pos.fund_path.append(pos.funding / pos.notional if pos.notional else 0.0)
            if len(pos.path) > PATH_LIMIT:
                pos.path, pos.fund_path = pos.path[-PATH_LIMIT:], pos.fund_path[-PATH_LIMIT:]
            risk_pct = pos.context.get("risk_pct") or 0.0
            est = exits.ExitState(stop=pos.stop, best=pos.exit_best or pos.entry_price, armed=pos.exit_armed, target=pos.exit_target)
            hit, px = exits.step(pos.exit_variant, est, pos.entry_price, pos.side, risk_pct, bar.high, bar.low, bar.open)
            pos.stop, pos.exit_best, pos.exit_armed = est.stop, est.best, est.armed
            if hit in ("stop", "trail", "target"):
                ev = self._close(pos, bar, px, hit, volume_24h, candle_qv, basis="resting")  # a resting order fills at its level, or the open on a gap
                if catch_up:
                    ev["catch_up"] = True
                events.append(ev)
                del self.wallet.positions[key]
                continue
            reason = None
            if self._exit_rule_hit(pos.exit_rule, cur, fired):
                reason = "signal"
            elif pos.bars_held >= pos.max_bars:
                reason = "time"
            if reason is None:
                self.wallet.positions[key] = asdict(pos)
                continue
            q = self._fresh_quote(symbol)
            if q is not None:
                price = q["bid"] if pos.side == 1 else q["ask"]  # a long sells into the bid, a short buys back at the ask
                ev = self._close(pos, bar, price, reason, volume_24h, candle_qv, basis="quote")
                if catch_up:
                    ev["catch_up"] = True
                events.append(ev)
                del self.wallet.positions[key]
            else:
                pos.pending_exit = reason  # a market order that reaches the market at the next candle's open
                self.wallet.positions[key] = asdict(pos)
        return events

    def _apply_funding(self, pos: Position, bar: Bar) -> None:
        """Charge or credit funding if this candle opens at a funding time the position is held through."""
        rate, key = self._funding_at(pos.symbol, bar, pos.last_funding_hour)
        if key:
            pos.last_funding_hour = key
        if rate is not None:
            charge = pos.side * rate * pos.qty * bar.close  # longs pay when the rate is positive
            pos.funding += charge
            self.wallet.cash -= charge

    def _funding_at(self, symbol: str, bar: Bar, last_hour: str) -> tuple[float | None, str]:
        """(rate charged at this candle's open or None, the hour key to remember)."""
        if self.market != "perps":
            return None, ""
        hour_key = bar.date[:13]  # "YYYY-MM-DD HH"
        if hour_key == last_hour:
            return None, ""
        try:
            hour, minute = int(bar.date[11:13]), int(bar.date[14:16])
        except (ValueError, IndexError):
            return None, ""
        if hour in FUNDING_HOURS and minute == 0:
            return self.funding.get(symbol, {}).get("rate", 0.0), hour_key
        return None, hour_key

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

    def _close(self, pos: Position, bar: Bar, price: float, reason: str, volume_24h: float, candle_qv: float, basis: str = "resting") -> dict:
        """Close a position at ``price`` (a quote, a resting order's level, or a candle open) and learn from it.

        Returns the event. Action ``duplicate`` means an identical trade was already on record: the wallet
        still closes the position, but no belief, exit statistic or shadow is touched a second time."""
        gross_notional = pos.qty * price
        slip = (impact_bps(gross_notional, candle_qv) if basis == "quote" else slippage_bps(gross_notional, volume_24h, candle_qv)) / 1e4
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
            slippage=pos.entry_slip + gross_notional * slip, bars_held=pos.bars_held, mfe=pos.mfe, mae=pos.mae,
            context={**pos.context, "exit_basis": basis}, explore=pos.explore, side=pos.side, funding=pos.funding,
        )
        findings = pm.lenses(t)
        risk_pct = pos.context.get("risk_pct") or 0.0
        variants: dict = {}
        if pos.exit_variant == "rule":
            # the path ran to the rule's own exit: every variant can be scored on it now, each at its own fill and fee
            costs = self._exit_costs(pos.entry_fee, pos.notional, pos.qty, basis, volume_24h, candle_qv, pos.funding / pos.notional if pos.notional else 0.0)
            variants = exits.simulate(pos.path, pos.entry_price, pos.side, risk_pct, price, costs, pos.fund_path)
            variants["rule"] = net_ret  # what actually happened (identical to the replay's own figure; see tests)
        elif pos.rule_track and pos.rule_track["done"]:
            # the rule would already have exited: score every variant on the rule's horizon now
            variants = self._score_track(pos.rule_track, pos.entry_price, pos.side, risk_pct, pos.path, pos.fund_path, pos.entry_fee / pos.notional if pos.notional else 0.0,
                                         pos.qty, volume_24h, candle_qv)
        event = {"symbol": pos.symbol, "signal": pos.signal, "action": "sell" if pos.side == 1 else "cover", "reason": reason, "price": fill,
                 "net_ret": net_ret, "pnl": pnl, "funding": pos.funding, "basis": basis, "findings": [tag for tag, _ in findings], "postmortem": None}
        if not pm.record(self.brain, t, findings, variants):
            return {**event, "action": "duplicate", "findings": []}
        pm.update_belief(self.brain, t)
        if variants:
            if not self.frozen:
                exits.record(self.brain, self.book, pos.signal, variants)
                self._maybe_change_policy(pos.signal)
        else:
            self._open_shadow(pos, bar.date)  # keep watching until the original rule would have exited
        event["postmortem"] = pm.learn_postmortem(self.brain, t, findings)
        n_signal = self.brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = ? AND signal = ?", (self.book, pos.signal)).fetchone()[0]
        if n_signal % 10 == 0:
            pm.learn_summary(self.brain, self.book, pos.signal)
        self.brain.commit()
        return event

    def _maybe_change_policy(self, signal: str) -> None:
        changed = exits.update_policy(self.brain, self.book, signal)
        if changed:
            self._log(f"    EXIT POLICY {signal}: {changed[0]} -> {changed[1]} (learned from {exits.stats(self.brain, self.book, signal)['rule'][0]} trades)")

    # ------------------------------------------------------------ rule track
    @staticmethod
    def _new_track(stop: float, last_funding_hour: str) -> dict:
        return {"stop": stop, "bars_held": 0, "done": False, "pending": False, "exit_price": None, "basis": None, "horizon": None,
                "funding": 0.0, "last_funding_hour": last_funding_hour}

    def _advance_track(self, track: dict, pos, bar: Bar, cur: Reading | None, fired: list[str], n_path: int) -> None:
        """Run the signal's own rule one candle further, exactly as a position under that rule would be run.

        ``pos`` carries side, entry price, risk, exit rule and max bars (a Position or a shadow dict);
        ``n_path`` is the number of path bars before this candle. A queued rule exit fills at this
        candle's open (horizon: the bars before it); a stop or a live-quote exit fills within this
        candle (horizon: the bars up to and including it)."""
        if track["done"]:
            return
        side, entry = (pos.side, pos.entry_price) if isinstance(pos, Position) else (pos["side"], pos["entry_price"])
        risk_pct = (pos.context.get("risk_pct") or 0.0) if isinstance(pos, Position) else pos["risk_pct"]
        exit_rule, max_bars, symbol = (pos.exit_rule, pos.max_bars, pos.symbol) if isinstance(pos, Position) else (pos["exit_rule"], pos["max_bars"], pos["symbol"])
        rate, hour_key = self._funding_at(symbol, bar, track["last_funding_hour"])
        if hour_key:
            track["last_funding_hour"] = hour_key
        if rate is not None:
            track["funding"] += side * rate * bar.close / entry  # the charge a real position of this size pays at this open
        if track["pending"]:
            track.update(done=True, pending=False, exit_price=bar.open, basis="next_open", horizon=n_path)
            return
        track["bars_held"] += 1
        st = exits.ExitState(stop=track["stop"], best=entry)
        hit, px = exits.step("rule", st, entry, side, risk_pct, bar.high, bar.low, bar.open)
        if hit == "stop":
            track.update(done=True, exit_price=px, basis="resting", horizon=n_path + 1)
        elif (cur is not None and self._exit_rule_hit(exit_rule, cur, fired)) or track["bars_held"] >= max_bars or n_path + 1 >= PATH_LIMIT:
            q = self._fresh_quote(symbol)
            if q is not None:
                track.update(done=True, exit_price=q["bid"] if side == 1 else q["ask"], basis="quote", horizon=n_path + 1)
            else:
                track["pending"] = True

    def _slip_fn(self, basis: str, qty: float, volume_24h: float, candle_qv: float):
        """Slippage as a function of the fill level, matching how the live trader fills on that basis."""
        if basis == "quote":
            return lambda price: impact_bps(qty * price, candle_qv) / 1e4  # the spread is already in the quote
        return lambda price: slippage_bps(qty * price, volume_24h, candle_qv) / 1e4

    def _exit_costs(self, entry_fee: float, notional: float, qty: float, rule_basis: str, volume_24h: float, candle_qv: float, rule_funding: float) -> dict:
        return {"entry": entry_fee / notional if notional else 0.0, "fee": self.fees["taker"], "slip": self._slip_fn("resting", qty, volume_24h, candle_qv),
                "rule_slip": self._slip_fn(rule_basis, qty, volume_24h, candle_qv), "rule_funding": rule_funding}

    def _score_track(self, track: dict, entry: float, side: int, risk_pct: float, path: list, fund_path: list, entry_fee_ratio: float, qty: float,
                     volume_24h: float, candle_qv: float) -> dict:
        h = track["horizon"]
        costs = self._exit_costs(entry_fee_ratio, 1.0, qty, track["basis"], volume_24h, candle_qv, track["funding"])
        return exits.simulate(path[:h], entry, side, risk_pct, track["exit_price"], costs, fund_path[:h])

    # --------------------------------------------------------------- shadows
    def _open_shadow(self, pos: Position, last_bar: str) -> None:
        """A position closed by a learned exit before its rule would have exited leaves a shadow that keeps
        running the rule, so the 'rule' baseline stays the rule and every variant is scored on one complete
        path with the funding that path would have paid."""
        track = pos.rule_track or self._new_track(pos.entry_price * (1 - pos.side * (pos.context.get("risk_pct") or 0.0)), pos.last_funding_hour)
        self.wallet.shadows[f"{pos.symbol}:{pos.signal}:{pos.entry_time}"] = {
            "symbol": pos.symbol, "signal": pos.signal, "side": pos.side, "entry_price": pos.entry_price, "entry_time": pos.entry_time,
            "risk_pct": pos.context.get("risk_pct") or 0.0, "max_bars": pos.max_bars, "exit_rule": pos.exit_rule, "qty": pos.qty,
            "entry_fee_ratio": pos.entry_fee / pos.notional if pos.notional else 0.0, "path": list(pos.path), "fund_path": list(pos.fund_path),
            "track": track, "last_bar": last_bar,
        }

    def _advance_shadows(self, symbol: str, bar: Bar, cur: Reading, fired: list[str], volume_24h: float, candle_qv: float) -> None:
        for key, sh in list(self.wallet.shadows.items()):
            if sh["symbol"] != symbol or bar.date <= sh.get("last_bar", sh["entry_time"]):
                continue  # the position itself already saw this candle
            sh["last_bar"] = bar.date
            track = sh["track"]
            was_pending = track["pending"]
            self._advance_track(track, sh, bar, cur, fired, len(sh["path"]))
            if not was_pending:  # a queued exit filled at the open, before this candle; otherwise the rule sat through it
                sh["path"].append((bar.open, bar.high, bar.low, bar.close))
                sh["fund_path"].append(track["funding"])
            if track["done"]:
                variants = self._score_track(track, sh["entry_price"], sh["side"], sh["risk_pct"], sh["path"], sh["fund_path"], sh["entry_fee_ratio"], sh["qty"], volume_24h, candle_qv)
                self._record_variants(sh["symbol"], sh["signal"], sh["entry_time"], variants)
                del self.wallet.shadows[key]

    def _record_variants(self, symbol: str, signal: str, entry_time: str, variants: dict) -> None:
        self.brain.db.execute(
            "UPDATE trades SET variants = ? WHERE book = ? AND symbol = ? AND signal = ? AND entry_time = ?",
            (json.dumps(variants), self.book, symbol, signal, entry_time),
        )
        if not self.frozen:
            exits.record(self.brain, self.book, signal, variants)
            self._maybe_change_policy(signal)
        self.brain.commit()

    # ---------------------------------------------------------------- marks
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
        """Equity fell below maintenance margin: the exchange closes everything at market and cancels the rest."""
        with self.brain.batch():
            for key, raw in list(self.wallet.positions.items()):
                bars = market.get(raw["symbol"])
                if not bars:
                    continue
                bar = bars[-2] if len(bars) >= 2 else bars[-1]
                self._close(Position(**raw), bar, bar.close, "liquidation", 0.0, max(bar.quote_volume, 1.0), basis="liquidation")
                del self.wallet.positions[key]
            self.wallet.pending.clear()
            self.brain._journal("liquidation", f"book {self.book}")
            self._save()

    def _log(self, line: str) -> None:
        self.wallet.log.append(line)
        if len(self.wallet.log) > LOG_LINES:
            self.wallet.log = self.wallet.log[-LOG_LINES:]

    # ----------------------------------------------------------------- lock
    def lock_path(self) -> Path:
        base = Path(self.brain.path).parent if self.brain.path != ":memory:" else Path(os.environ.get("TMPDIR", "/tmp"))
        return base / f"trade-{self.book}.lock"

    def other_traders(self) -> list[tuple[int, str]]:
        """Other live trader processes on the same book, whatever started them (terminal, launchd, old code).
        A trader on a different --book is allowed: that is how a frozen baseline runs beside the adaptive book."""
        import subprocess

        try:
            out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return []
        found = []
        me, parent = os.getpid(), os.getppid()
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            pid_str, _, cmd = line.partition(" ")
            if not pid_str.isdigit() or int(pid_str) in (me, parent):
                continue  # ourselves, or the wrapper that launched us (caffeinate, a shell)
            cmd = cmd.strip()
            if "caffeinate" in cmd or cmd.startswith("grep") or " grep " in cmd or "/bin/sh -c" in cmd or "/bin/bash -c" in cmd:
                continue  # wrappers and shells mention the command without being a trader
            if _TRADER_CMD.search(cmd):
                m = re.search(r"--book[= ]+(\S+)", cmd)
                other_book = m.group(1) if m else "main"
                if other_book == self.book:
                    found.append((int(pid_str), cmd))
        return found

    def acquire_lock(self) -> None:
        """Refuse to run two traders on one book: they would each close the same positions."""
        others = self.other_traders()
        if others:
            listing = "; ".join(f"pid {pid}: {cmd[:80]}" for pid, cmd in others)
            raise RuntimeError(
                f"another trader process is running ({listing}). Two traders on one brain corrupt the book. "
                "If it is the background service, run `scripts/macos-service.sh uninstall` (or reinstall it to update it); otherwise `kill <pid>`."
            )
        path = self.lock_path()
        if path.exists():
            try:
                pid = int(path.read_text().strip() or 0)
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise RuntimeError(f"another trader (pid {pid}) is already running book '{self.book}'. Stop it first, or use --book for a separate book.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()))

    def release_lock(self) -> None:
        try:
            if self.lock_path().read_text().strip() == str(os.getpid()):
                self.lock_path().unlink()
        except OSError:
            pass

    # -------------------------------------------------------------- rebuild
    def rebuild_stats(self) -> dict:
        """Recompute beliefs and exit statistics from the trade log, the single source of truth.

        Only trades recorded under the current results version count: older ones were measured under a
        different execution model and are evidence about that model, not this one."""
        self.brain.db.execute("DELETE FROM beliefs WHERE book = ?", (self.book,))
        self.brain.db.execute("DELETE FROM exit_stats WHERE book = ?", (self.book,))
        n = 0
        for r in self.brain.db.execute("SELECT * FROM trades WHERE book = ? AND model_version = ? ORDER BY id", (self.book, self.brain.RESULTS_VERSION)).fetchall():
            ctx = json.loads(r["context"])
            t = pm.Trade(book=self.book, symbol=r["symbol"], signal=r["signal"], entry_time=r["entry_time"], entry_price=r["entry_price"], exit_time=r["exit_time"],
                         exit_price=r["exit_price"], exit_reason=r["exit_reason"], qty=r["qty"], notional=r["notional"], gross_ret=r["gross_ret"], net_ret=r["net_ret"],
                         pnl=r["pnl"], fees=r["fees"], slippage=r["slippage"], bars_held=r["bars_held"], mfe=r["mfe"], mae=r["mae"], context=ctx, explore=bool(r["explore"]),
                         side=r["side"], funding=r["funding"])
            pm.update_belief(self.brain, t)
            variants = json.loads(r["variants"] or "{}")
            if variants and not self.frozen:
                exits.record(self.brain, self.book, r["signal"], variants)
            n += 1
        self.brain.commit()
        return {"trades": n}

    # ---------------------------------------------------------------- reset
    def reset(self, wallet: float | None = None, forget: bool = False) -> dict:
        """Close every position without a post-mortem and start the wallet again.

        With ``forget`` the book's trade history, beliefs, exit statistics, policies and post-mortem
        cells are erased too; otherwise what the brain learned is kept and only the money resets."""
        n_positions = len(self.wallet.positions)
        start = wallet if wallet is not None else self.wallet.start
        self.wallet = Wallet(cash=start, start=start, peak=start)
        removed = 0
        if forget:
            self.brain.db.execute("DELETE FROM trades WHERE book = ?", (self.book,))
            self.brain.db.execute("DELETE FROM beliefs WHERE book = ?", (self.book,))
            self.brain.db.execute("DELETE FROM exit_stats WHERE book = ?", (self.book,))
            self.brain.db.execute("DELETE FROM state WHERE key IN (?, ?)", (exits.policy_key(self.book), f"policy_log:{self.book}"))
            removed = self.brain.forget(source=f"trade:{self.book}")
        self._save(last_tick="")
        self.brain._journal("reset", f"book {self.book}: closed {n_positions} positions, wallet {start:.0f}" + (", history forgotten" if forget else ""))
        self.brain.commit()
        return {"positions_closed": n_positions, "wallet": start, "forgot_cells": removed}

    # --------------------------------------------------------------- report
    def report(self) -> dict:
        w = self.wallet
        rows = self.brain.db.execute(
            "SELECT signal, COUNT(*) AS n, SUM(CASE WHEN net_ret > 0 THEN 1 ELSE 0 END) AS wins, AVG(net_ret) AS avg_ret, SUM(pnl) AS pnl,"
            " SUM(fees) AS fees, SUM(slippage) AS slip, SUM(funding) AS funding FROM trades WHERE book = ? GROUP BY signal ORDER BY n DESC", (self.book,)
        ).fetchall()
        return {
            "market": self.market, "frozen": self.frozen, "equity": w.equity(), "cash": w.cash, "start": w.start, "return": w.equity() / w.start - 1,
            "max_drawdown": w.max_drawdown, "gross_notional": w.gross_notional(), "open_risk": w.open_risk(), "leverage": self.leverage,
            "open": [{"symbol": p.symbol, "signal": p.signal, "side": "long" if p.side == 1 else "short", "entry": p.entry_price, "mark": p.mark,
                      "unrealized": (p.side * (p.mark / p.entry_price - 1)) if p.mark else 0.0, "bars": p.bars_held, "stop": p.stop, "explore": p.explore,
                      "funding": p.funding, "pending_exit": p.pending_exit}
                     for p in w.open_positions()],
            "pending": len(w.pending), "shadows": len(w.shadows), "closed": w.closed,
            "by_signal": [{"signal": r["signal"], "trades": r["n"], "win_rate": r["wins"] / r["n"], "avg_ret": r["avg_ret"], "pnl": r["pnl"], "fees": r["fees"],
                           "slippage": r["slip"], "funding": r["funding"] or 0.0} for r in rows],
        }

    def beliefs(self) -> list[dict]:
        """The belief table with the verdict the trader actually acts on (evidence counted in time blocks)."""
        rows = self.brain.db.execute(
            "SELECT signal, regime, vol_bucket, wins, losses, sum_ret FROM beliefs WHERE book = ? ORDER BY signal, regime, vol_bucket", (self.book,)
        ).fetchall()
        words = {"full": "trade", "half": "half size", "avoid": "avoid", "explore": "exploring"}
        out = []
        for r in rows:
            n = r["wins"] + r["losses"]
            b = pm.belief(self.brain, self.book, r["signal"], r["regime"], r["vol_bucket"])
            out.append({"signal": r["signal"], "regime": r["regime"], "vol": r["vol_bucket"], "n": n, "win_rate": r["wins"] / n, "avg_ret": r["sum_ret"] / n,
                        "stderr": b["stderr"], "blocks": b["blocks"], "verdict": words[pm.verdict(b)]})
        return out

    # ----------------------------------------------------------------- loop
    def seconds_until_next_close(self, now: float | None = None) -> float:
        period = INTERVAL_SECONDS[self.interval]
        now = time.time() if now is None else now
        return period - (now % period) + 8

    def run(self, log=print, once: bool = False, sleep=time.sleep, companions: tuple["Trader", ...] = ()) -> None:
        """Trade live until stopped. ``companions`` (a frozen baseline, say) tick on exactly the same inputs each candle."""
        def out(line: str) -> None:
            log(line)
            self._log(line)

        books = (self, *companions)
        for t in books:
            t.acquire_lock()
        for t in books:
            try:
                rebuilt = t.rebuild_stats()
                if rebuilt["trades"]:
                    out(f"[{t.book}] beliefs and exit statistics rebuilt from {rebuilt['trades']} recorded trades")
            except Exception as exc:  # a locked database must not stop trading; stats are rebuilt on the next start
                out(f"[{t.book}] could not rebuild statistics now ({exc}); continuing with the stored beliefs")
        while True:
            try:
                started = time.time()
                universe, market, quotes, funding = self.fetch_inputs(set().union(*(c.wallet.symbols() for c in companions)) if companions else None)
                for t in books:
                    r = t.tick(market, universe, quotes, funding, live=True)
                    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    tag = "" if t is self else f" [{t.book}]"
                    out(f"[{stamp}]{tag} {r['symbols']} pairs in {time.time() - started:.0f}s | equity at close {r['equity']:.2f} USDT ({r['equity'] / t.wallet.start - 1:+.2%}) "
                        f"cash {r['cash']:.2f} open {r['open']}" + (f" (of which {sum(1 for p in t.wallet.positions.values() if p.get('pending_exit'))} exiting at next open)" if any(p.get("pending_exit") for p in t.wallet.positions.values()) else ""))
                    for e in r["events"]:
                        if e["action"] in ("buy", "short"):
                            out(f"   {tag} {e['action'].upper():5} {e['symbol']:12} {e['signal']:18} @ {e['price']:.6g}  {e['notional']:.0f} USDT  stop {e['stop']:.6g}  p(win) {e['p_win']}{'  (exploring)' if e['explore'] else ''}")
                        elif e["action"] == "duplicate":
                            out(f"   {tag} NOTE {e['symbol']:12} {e['signal']:18} closed again? record already existed; nothing learned twice")
                        elif e["action"] in ("sell", "cover"):
                            fund = f" funding {e['funding']:+.2f}" if e.get("funding") else ""
                            late = " [caught up from a missed candle]" if e.get("catch_up") else ""
                            out(f"   {tag} {e['action'].upper():5} {e['symbol']:12} {e['signal']:18} @ {e['price']:.6g}  {e['net_ret']:+.2%} ({e['pnl']:+.2f} USDT{fund}) by {e['reason']}{late}  findings: {', '.join(e['findings'])}")
                        elif e["action"] == "skip" and ("belief" in e["why"] or "conflict" in e["why"] or "quote" in e["why"]):
                            out(f"   {tag} SKIP {e['symbol']:12} {e['signal']:18} {e['why']}")
                    t._save()
            except Exception as exc:
                out(f"trade error: {exc}")
            if once:
                for t in books:
                    t.release_lock()
                return
            sleep(self.seconds_until_next_close())
