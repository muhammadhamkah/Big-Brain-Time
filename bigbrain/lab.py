"""The lab: test a trading hypothesis the hard way, so the brain never has to pay for it live.

A hypothesis is a *rule*: a function of past candles and a few parameters that says, after
each candle closes, what position to hold next and where its stop sits. The lab runs a rule
over months of candles, on one market or a whole universe, with the same honesty as the live
trader:

* decisions after the close, market orders filled at the next candle's open;
* stops and stop-entries are resting orders that fill at their level, or at the open when
  a candle gaps through it;
* every fill pays the fee, half the spread and slippage, both ways; on perps every funding
  time a position is held through is paid or received at the real historical rate.

Then it does the three things a backtest screenshot never does:

* **grid**: every parameter combination, with a *plateau* score. A real edge is a region
  where neighbouring settings all work; a lone spike is one lucky fit to one stretch of noise.
* **walk-forward**: parameters are chosen on the past only and judged on the window that
  follows, window after window, cut by time so a universe is judged on the same dates. The
  stitched out-of-sample record is the only number that says anything about the future.
* **memory**: the verdict becomes a lesson cell, so the brain can answer "does parabolic SAR
  work on one-minute EURUSD" or "which of my signals has an edge" from evidence.

Rules come in two shapes. A *market rule* sees one symbol's candles. A *portfolio rule* sees
the whole universe at once (cross-sectional momentum ranks every symbol against the others).
The playbook rule runs the live trader's own eight signals through the same test, so the
brain's beliefs can be checked against months of history in minutes instead of paid for
one candle at a time.

Nothing here can tune a rule to catch tops and bottoms. Anything that appears to has been
fitted to the past, and the walk-forward exists to show that.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import Bar
from bigbrain.watch import INTERVAL_SECONDS, detect, readings

Markets = dict[str, list]  # symbol -> candles


# ------------------------------------------------------------------ costs
@dataclass
class Costs:
    fee: float = 0.0005  # per side, fraction of notional (Binance perps taker 0.05%; spot 0.1%; an FX broker 0)
    spread_bps: float = 1.0  # full bid-ask spread in basis points; half is paid on every fill
    slippage_bps: float = 0.0  # extra slippage per fill in basis points (stops in a fast market fill worse)

    def fill(self, price: float, side: int) -> float:
        """The price actually paid: buying lifts the ask, selling hits the bid, plus slippage."""
        return price * (1 + side * (self.spread_bps / 2 + self.slippage_bps) / 1e4)


# ------------------------------------------------------------------ rules
# A market rule maps (bars, params, extras) to one decision per candle, made after that candle
# closed and acting in the next one: {"target": +1|-1|0, "stop": level or None, "reverse": bool,
# "weight": fraction of the account (default 1)}. With reverse=True a stop that fills also opens the
# opposite side at the same level (a stop-and-reverse system such as parabolic SAR).
# A portfolio rule maps (markets, params, extras) to {symbol: {date: decision}}.

RULES: dict[str, dict] = {}


def rule(name: str, defaults: dict, doc: str, portfolio: bool = False, needs: tuple = (), hint: str = ""):
    def wrap(fn):
        RULES[name] = {"fn": fn, "defaults": defaults, "doc": doc, "portfolio": portfolio, "needs": needs, "hint": hint}
        return fn

    return wrap


def _flat() -> dict:
    return {"target": 0, "stop": None, "reverse": False}


def psar_series(bars: list[Bar], start: float = 0.02, increment: float = 0.02, maximum: float = 0.2) -> list[tuple[float, bool, float]]:
    """Wilder's parabolic SAR exactly as the common TradingView script computes it.

    Returns (sar for this candle, uptrend, sar for the next candle) per candle; the first is empty."""
    out: list[tuple[float, bool, float]] = [(math.nan, True, math.nan)]
    if len(bars) < 2:
        return out
    uptrend, ep, sar, af, next_sar = True, 0.0, 0.0, start, math.nan
    for i in range(1, len(bars)):
        b, p = bars[i], bars[i - 1]
        first = False
        sar = next_sar
        if i == 1:
            if b.close > p.close:
                uptrend, ep, prev_sar, prev_ep = True, b.high, p.low, b.high
            else:
                uptrend, ep, prev_sar, prev_ep = False, b.low, p.high, b.low
            first = True
            sar = prev_sar + start * (prev_ep - prev_sar)
        if uptrend:
            if sar > b.low:
                first, uptrend, sar, ep, af = True, False, max(ep, b.high), b.low, start
        else:
            if sar < b.high:
                first, uptrend, sar, ep, af = True, True, min(ep, b.low), b.high, start
        if not first:
            if uptrend and b.high > ep:
                ep, af = b.high, min(af + increment, maximum)
            elif not uptrend and b.low < ep:
                ep, af = b.low, min(af + increment, maximum)
        if uptrend:
            sar = min(sar, p.low, bars[i - 2].low if i > 1 else p.low)
        else:
            sar = max(sar, p.high, bars[i - 2].high if i > 1 else p.high)
        next_sar = sar + af * (ep - sar)
        out.append((sar, uptrend, next_sar))
    return out


@rule("psar", {"start": 0.02, "increment": 0.02, "maximum": 0.2}, "parabolic SAR stop-and-reverse: always in the market, flips at the SAR")
def psar_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    series = psar_series(bars, params["start"], params["increment"], params["maximum"])
    out = []
    for i, (sar, up, nxt) in enumerate(series):
        if i == 0 or math.isnan(nxt):
            out.append(_flat())
        else:
            out.append({"target": 1 if up else -1, "stop": nxt, "reverse": True})
    return out


@rule("sma_cross", {"fast": 20, "slow": 50}, "moving-average crossover: long above, short below, market orders at the next open")
def sma_cross_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    closes = [b.close for b in bars]
    fast, slow = ind.sma(closes, int(params["fast"])), ind.sma(closes, int(params["slow"]))
    out = []
    for f, s in zip(fast, slow):
        target = 0 if f is None or s is None else (1 if f > s else -1)
        out.append({"target": target, "stop": None, "reverse": False})
    return out


@rule("rsi_reversion", {"period": 14, "entry": 30.0, "exit": 55.0, "stop_pct": 0.02},
      "long when RSI falls below entry, out when it recovers past exit, with a fixed stop")
def rsi_reversion_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    closes = [b.close for b in bars]
    r = ind.rsi(closes, int(params["period"]))
    out, holding, entry_px = [], False, 0.0
    for i, v in enumerate(r):
        if v is None:
            out.append(_flat())
            continue
        if holding and v >= params["exit"]:
            holding = False
        elif not holding and v <= params["entry"]:
            holding, entry_px = True, closes[i]
        out.append({"target": 1 if holding else 0, "stop": entry_px * (1 - params["stop_pct"]) if holding else None, "reverse": False})
    return out


PLAYBOOK_SIGNALS = ("rsi_oversold", "below_lower_band", "above_upper_band", "golden_cross", "macd_bullish", "rsi_overbought", "death_cross", "macd_bearish")


_INDICATOR_CACHE: dict[tuple, tuple] = {}


def _indicators(bars: list[Bar]) -> tuple[list, list]:
    """Readings and ATR for a candle list, cached: a grid runs the same candles through many settings."""
    key = (bars[0].date, bars[-1].date, len(bars), bars[-1].close)
    hit = _INDICATOR_CACHE.get(key)
    if hit is None:
        if len(_INDICATOR_CACHE) > 64:
            _INDICATOR_CACHE.clear()
        hit = (readings(bars), ind.atr([b.high for b in bars], [b.low for b in bars], [b.close for b in bars], 14))
        _INDICATOR_CACHE[key] = hit
    return hit


@rule("playbook", {"signal": "rsi_oversold", "stop_atr": 2.0, "max_bars": 16},
      "one of the live trader's own signals, managed exactly as the trader manages it: ATR stop, the signal's exit rule, a time limit",
      hint="signal=* runs all eight")
def playbook_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    from bigbrain.trader import PLAYBOOK, Trader

    signal = params["signal"]
    play = PLAYBOOK[signal]
    side, exit_rule, max_bars = play["side"], play["exit"], int(params["max_bars"])
    rs, atr = _indicators(bars)
    out, holding, stop, held = [], False, None, 0
    for i in range(len(bars)):
        if i == 0:
            out.append(_flat())
            continue
        fired = detect(rs[i - 1], rs[i])
        if holding:
            held += 1
            if Trader._exit_rule_hit(exit_rule, rs[i], fired) or held >= max_bars:
                holding = False
        if not holding and signal in fired and atr[i]:
            holding, held = True, 0
            stop = bars[i].close * (1 - side * max(params["stop_atr"] * atr[i] / bars[i].close, 0.005))
        out.append({"target": side if holding else 0, "stop": stop if holding else None, "reverse": False})
    return out


@rule("trend_vt", {"sma": 100, "vol_window": 20, "target_vol": 0.4, "short": 0},
      "daily trend with volatility targeting: long above the moving average (short below if short=1), sized so realized volatility meets the target",
      hint="daily candles; the boring edge")
def trend_vt_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    closes = [b.close for b in bars]
    per_year = int(365 * 86400 / INTERVAL_SECONDS[(extras or {}).get("interval", "1d")])
    sma = ind.sma(closes, int(params["sma"]))
    vol = ind.realized_volatility(closes, int(params["vol_window"]), per_year)
    out = []
    for c, m, v in zip(closes, sma, vol):
        if m is None or v is None or v <= 0:
            out.append(_flat())
            continue
        target = 1 if c > m else (-1 if int(params["short"]) else 0)
        out.append({"target": target, "stop": None, "reverse": False, "weight": min(1.0, params["target_vol"] / v)})
    return out


@rule("funding_carry", {"threshold": 0.30, "lookback": 3, "exit_ratio": 0.5, "stop_pct": 0.08},
      "perps funding carry: short when the crowd pays to be long (annualized funding above the threshold), long when it pays to be short; out when it normalizes",
      needs=("funding",), hint="perps only; funding is paid or received at the real historical rate")
def funding_carry_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    funding = (extras or {}).get("funding") or {}
    per_year = 3 * 365  # funding is paid every eight hours
    out, holding, entry_px, recent = [], 0, 0.0, []
    for b in bars:
        if b.date in funding:
            recent = (recent + [funding[b.date]])[-int(params["lookback"]):]
        ann = (sum(recent) / len(recent)) * per_year if len(recent) >= int(params["lookback"]) else 0.0
        if holding and abs(ann) < params["threshold"] * params["exit_ratio"]:
            holding = 0
        if not holding:
            if ann > params["threshold"]:
                holding, entry_px = -1, b.close
            elif ann < -params["threshold"]:
                holding, entry_px = 1, b.close
        out.append({"target": holding, "stop": entry_px * (1 - holding * params["stop_pct"]) if holding else None, "reverse": False})
    return out


@rule("xs_momentum", {"lookback": 30, "skip": 1, "hold": 7, "top": 0.2, "stop_pct": 0.25},
      "cross-sectional momentum: every `hold` candles rank the universe by trailing return, long the top fraction, short the bottom, market neutral",
      portfolio=True, hint="daily candles over a universe, e.g. --symbols top:30")
def xs_momentum_rule(markets: Markets, params: dict, extras: dict | None = None) -> dict[str, dict[str, dict]]:
    dates = sorted({b.date for bars in markets.values() for b in bars})
    closes = {s: {b.date: b.close for b in bars} for s, bars in markets.items()}
    lookback, skip, hold, top = int(params["lookback"]), int(params["skip"]), int(params["hold"]), params["top"]
    out: dict[str, dict[str, dict]] = {s: {} for s in markets}
    current: dict[str, dict] = {s: _flat() for s in markets}
    for i, d in enumerate(dates):
        if i % hold == 0 and i >= lookback:
            scores = {}
            for s in markets:
                then, now = dates[i - lookback], dates[i - skip]
                if then in closes[s] and now in closes[s] and d in closes[s]:
                    scores[s] = closes[s][now] / closes[s][then] - 1
            k = max(1, int(len(scores) * top)) if scores else 0
            ranked = sorted(scores, key=scores.get)
            longs, shorts = set(ranked[-k:]) if k else set(), set(ranked[:k]) if k else set()
            w = 1.0 / (2 * k) if k else 0.0
            for s in markets:
                side = 1 if s in longs else -1 if s in shorts else 0
                stop = closes[s][d] * (1 - side * params.get("stop_pct", 0.25)) if side else None
                current[s] = {"target": side, "stop": stop, "reverse": False, "weight": w} if side else _flat()
        for s in markets:
            if d in closes[s]:
                out[s][d] = dict(current[s])
    return out


# -------------------------------------------------------------- simulator
@dataclass
class Trade:
    side: int
    entry_i: int
    entry: float  # fill, costs included
    exit_i: int
    exit: float
    ret: float  # net return on notional: fees, slippage and funding included
    reason: str
    weight: float = 1.0  # fraction of the account the position used
    date: str = ""  # exit candle
    symbol: str = ""
    funding: float = 0.0  # funding paid (positive) as a fraction of notional


@dataclass
class Result:
    trades: int = 0
    wins: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # mean net return per trade, on the trade's notional
    net_return: float = 0.0  # compounded on the account
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    tstat: float = 0.0  # expectancy / standard error: how many standard errors the mean is from zero
    symbols: int = 1
    log: list = field(default_factory=list)  # Trade records

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "log"}


def simulate(bars: list[Bar], decisions: list[dict], costs: Costs, interval: str = "15m", size: float = 1.0, start: int = 0,
             funding: dict[str, float] | None = None, symbol: str = "") -> Result:
    """Run one rule's decisions through one market, strictly in order.

    ``start``: trades entered before this index are warm-up and not counted. ``funding``: {candle date: rate}
    for perps; a position held through a funding candle pays side x rate (longs pay when it is positive)."""
    pos: dict | None = None
    equity, curve, trades = 1.0, [], []
    funding = funding or {}

    def close_at(price: float, i: int, reason: str) -> None:
        nonlocal pos, equity
        fill = costs.fill(price, -pos["side"])
        gross = pos["side"] * (fill / pos["entry"] - 1)
        ret = gross - costs.fee - costs.fee * fill / pos["entry"] - pos["funding"]
        if pos["entry_i"] >= start:
            trades.append(Trade(pos["side"], pos["entry_i"], pos["entry"], i, fill, ret, reason, pos["weight"], bars[i].date, symbol, pos["funding"]))
            equity = max(0.0, equity * (1 + size * pos["weight"] * ret))  # a loss beyond the account is a liquidation, not a debt
        pos = None

    def open_at(price: float, i: int, side: int, weight: float) -> None:
        nonlocal pos
        pos = {"side": side, "entry": costs.fill(price, side), "entry_i": i, "weight": weight, "funding": 0.0}

    for i in range(1, len(bars)):
        d, bar = decisions[i - 1], bars[i]
        target, weight = d.get("target", 0), float(d.get("weight", 1.0))
        if pos and target != pos["side"]:
            close_at(bar.open, i, "signal")
        if pos is None and target and weight > 0:
            open_at(bar.open, i, target, weight)
        if pos and bar.date in funding:
            pos["funding"] += pos["side"] * funding[bar.date] * bar.open / pos["entry"]
        stop = d.get("stop")
        if pos and stop is not None:
            hit = bar.low <= stop if pos["side"] == 1 else bar.high >= stop
            if hit:
                gapped = bar.open < stop if pos["side"] == 1 else bar.open > stop
                level = bar.open if gapped else stop
                side = pos["side"]
                close_at(level, i, "stop")
                if d.get("reverse"):
                    open_at(level, i, -side, weight)
        mark = equity
        if pos and pos["entry_i"] >= start:
            mark = max(0.0, equity * (1 + size * pos["weight"] * pos["side"] * (bar.close / pos["entry"] - 1)))
        curve.append(mark)
    if pos and pos["entry_i"] >= start:
        close_at(bars[-1].close, len(bars) - 1, "end")
    return summarize(trades, curve, interval)


def summarize(trades: list[Trade], curve: list[float], interval: str) -> Result:
    """Statistics of one market's trade list and its marked-to-market equity curve."""
    r = Result(log=trades, trades=len(trades))
    _trade_stats(r, trades)
    if not trades:
        return r
    r.net_return = curve[-1] - 1 if curve else 0.0
    r.max_drawdown = ind.max_drawdown(curve) if curve else 0.0
    per_bar = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve)) if curve[i - 1] > 0]
    r.sharpe = ind.sharpe(per_bar, periods_per_year=int(365 * 86400 / INTERVAL_SECONDS[interval]))
    return r


def summarize_pool(trades: list[Trade], symbols: int) -> Result:
    """Statistics of trades pooled across a universe: equity compounds each trade by its weight, in exit order,
    and the Sharpe ratio comes from daily account returns."""
    r = Result(log=trades, trades=len(trades), symbols=symbols)
    _trade_stats(r, trades)
    if not trades:
        return r
    equity, curve, daily = 1.0, [1.0], {}
    for t in sorted(trades, key=lambda t: (t.date, t.entry_i)):
        equity = max(0.0, equity * (1 + t.weight * t.ret))  # liquidation floors the account at zero
        curve.append(equity)
        daily[t.date[:10]] = daily.get(t.date[:10], 0.0) + t.weight * t.ret
    r.net_return = equity - 1
    r.max_drawdown = ind.max_drawdown(curve)
    r.sharpe = ind.sharpe(list(daily.values()), periods_per_year=365) if len(daily) > 2 else 0.0
    return r


def _trade_stats(r: Result, trades: list[Trade]) -> None:
    if not trades:
        return
    rets = [t.ret for t in trades]
    wins = [x for x in rets if x > 0]
    losses = [x for x in rets if x <= 0]
    r.wins = len(wins)
    r.profit_factor = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    r.expectancy = sum(rets) / len(rets)
    if len(rets) > 1:
        var = sum((x - r.expectancy) ** 2 for x in rets) / (len(rets) - 1)
        se = math.sqrt(var / len(rets))
        r.tstat = r.expectancy / se if se > 0 else 0.0


# ------------------------------------------------------------- evaluation
def _as_markets(data) -> Markets:
    return data if isinstance(data, dict) else {"_": data}


def decisions_for(rule_name: str, markets: Markets, params: dict, extras: dict | None = None) -> dict[str, list[dict]]:
    """One decision list per symbol, whatever the rule's shape."""
    spec = RULES[rule_name]
    full = {**spec["defaults"], **params}
    extras = extras or {}
    if spec.get("portfolio"):
        by_date = spec["fn"](markets, full, extras)
        return {s: [by_date.get(s, {}).get(b.date, _flat()) for b in bars] for s, bars in markets.items()}
    return {s: spec["fn"](bars, full, {**extras, "funding": (extras.get("funding") or {}).get(s, {})}) for s, bars in markets.items()}


def evaluate(rule_name: str, data, params: dict, costs: Costs, interval: str = "15m", start: int = 0, extras: dict | None = None,
             start_date: str | None = None, end_date: str | None = None) -> Result:
    """Run a rule over one market (a candle list) or a universe (a dict). ``start`` (an index) or ``start_date`` marks
    where counted trades begin; ``end_date`` truncates every market's candles before that date."""
    markets = _as_markets(data)
    if end_date is not None:
        markets = {s: [b for b in bars if b.date < end_date] for s, bars in markets.items()}
    markets = {s: bars for s, bars in markets.items() if len(bars) > 2}
    if not markets:
        return Result()
    extras = {**(extras or {}), "interval": interval}
    decs = decisions_for(rule_name, markets, params, extras)
    funding = extras.get("funding") or {}
    single = len(markets) == 1 and "_" in markets
    if single:
        bars = markets["_"]
        first = start if start_date is None else next((i for i, b in enumerate(bars) if b.date >= start_date), len(bars))
        return simulate(bars, decs["_"], costs, interval, start=first, funding=funding.get("_"))
    pooled: list[Trade] = []
    scale = 1.0 if RULES[rule_name]["portfolio"] else 1.0 / len(markets)
    for s, bars in markets.items():
        first = start if start_date is None else next((i for i, b in enumerate(bars) if b.date >= start_date), len(bars))
        r = simulate(bars, decs[s], costs, interval, start=first, funding=funding.get(s), symbol=s)
        for t in r.log:
            t.weight *= scale
        pooled += r.log
    return summarize_pool(pooled, len(markets))


# ------------------------------------------------------------------- grid
def parse_grid(text: str, defaults: dict) -> dict[str, list]:
    """'start=0.01,0.02;maximum=0.1,0.2' -> {'start': [0.01, 0.02], 'maximum': [0.1, 0.2]}; missing parameters keep their default.
    A string parameter accepts '*' for every choice the rule knows (the playbook's eight signals)."""
    grid: dict[str, list] = {k: [v] for k, v in defaults.items()}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        key, _, values = part.partition("=")
        key = key.strip()
        if key not in defaults:
            raise ValueError(f"unknown parameter '{key}'; this rule has {sorted(defaults)}")
        kind = type(defaults[key])
        if kind is str and values.strip() == "*":
            grid[key] = list(PLAYBOOK_SIGNALS)
        else:
            grid[key] = sorted({kind(v.strip()) for v in values.split(",") if v.strip()})
    return grid


def grid_points(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def grid_search(rule_name: str, data, grid: dict[str, list], costs: Costs, interval: str = "15m", workers: int = 4, start: int = 0,
                extras: dict | None = None, start_date: str | None = None, end_date: str | None = None) -> list[tuple[dict, Result]]:
    points = grid_points(grid)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda p: evaluate(rule_name, data, p, costs, interval, start, extras, start_date, end_date), points))
    return list(zip(points, results))


def _score(r: Result) -> float:
    """What a fit is judged by: how many standard errors the mean trade is above zero (few lucky trades score low)."""
    return r.tstat if r.trades >= 20 else -abs(r.tstat) - 1.0


def plateau(rows: list[tuple[dict, Result]], grid: dict[str, list]) -> list[dict]:
    """For every grid point: its own profit factor, its neighbours' mean, and the plateau score (the lower of the two).

    Neighbours differ in exactly one numeric parameter by one step. A setting whose neighbours fail is a spike, not an edge."""
    index = {tuple(p[k] for k in grid): r for p, r in rows}
    keys = list(grid)
    out = []
    for p, r in rows:
        neighbours = []
        for k in keys:
            values = grid[k]
            if isinstance(p[k], str):
                continue
            j = values.index(p[k])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(values):
                    q = dict(p)
                    q[k] = values[jj]
                    n = index.get(tuple(q[kk] for kk in keys))
                    if n is not None:
                        neighbours.append(min(n.profit_factor, 10.0))
        own = min(r.profit_factor, 10.0)
        nmean = sum(neighbours) / len(neighbours) if neighbours else own
        out.append({"params": p, "result": r, "neighbours": nmean, "plateau": min(own, nmean), "n_neighbours": len(neighbours)})
    return out


# ------------------------------------------------------------ walk-forward
def walk_forward(rule_name: str, data, grid: dict[str, list], costs: Costs, interval: str = "15m", folds: int = 6, workers: int = 4,
                 extras: dict | None = None) -> dict:
    """Anchored walk-forward, cut by time: for each window after the first, choose parameters on everything before it,
    then trade that window with them. A universe is judged on the same dates for every symbol. The stitched
    out-of-sample trades are the honest record."""
    markets = _as_markets(data)
    dates = sorted({b.date for bars in markets.values() for b in bars})
    if folds < 2 or len(dates) < folds * 50:
        raise ValueError("walk-forward needs at least two windows of fifty candles each")
    window = len(dates) // folds
    steps, oos_trades = [], []
    for k in range(1, folds):
        train_end = dates[k * window]
        test_end = dates[(k + 1) * window] if k < folds - 1 else None
        fitted = grid_search(rule_name, data, grid, costs, interval, workers, extras=extras, end_date=train_end)
        best_params, best = max(fitted, key=lambda pr: _score(pr[1]))
        test = evaluate(rule_name, data, best_params, costs, interval, extras=extras, start_date=train_end, end_date=test_end)  # history for warm-up, trades only in the window
        steps.append({"window": k, "train_from": dates[0], "train_to": train_end, "test_to": test_end or dates[-1], "params": best_params,
                      "in_sample": best.as_dict(), "out_of_sample": test.as_dict()})
        oos_trades += test.log
    stitched = summarize_pool(oos_trades, len(markets))
    return {"folds": folds, "steps": steps, "out_of_sample": stitched.as_dict(), "oos_trades": oos_trades,
            "benchmark": buy_and_hold(markets, dates[window])}  # holding the universe over the same out-of-sample span


def buy_and_hold(markets: Markets, start_date: str, end_date: str | None = None) -> dict:
    """Equal-weight holding of the universe from ``start_date``: what doing nothing clever returned over the same span,
    so a long-only signal in a rising market is not mistaken for an edge."""
    curves: dict[str, list[float]] = {}
    for s, bars in markets.items():
        window = [b for b in bars if b.date >= start_date and (end_date is None or b.date < end_date)]
        if len(window) > 1:
            curves[s] = [b.close / window[0].close for b in window]
    if not curves:
        return {"symbols": 0, "net_return": 0.0, "max_drawdown": 0.0}
    n = max(len(c) for c in curves.values())
    equity = [sum(c[min(i, len(c) - 1)] for c in curves.values()) / len(curves) for i in range(n)]
    return {"symbols": len(curves), "net_return": equity[-1] - 1, "max_drawdown": ind.max_drawdown(equity)}


# ---------------------------------------------------------------- verdict
def verdict(oos: dict, best_is: dict | None = None) -> tuple[str, str]:
    """(label, sentence) for a stitched out-of-sample record."""
    n, pf, t = oos.get("trades", 0), oos.get("profit_factor", 0.0), oos.get("tstat", 0.0)
    if n < 30:
        return "insufficient", f"only {n} out-of-sample trades: not enough to say anything."
    if pf < 1.0:
        return "no edge", f"out-of-sample profit factor {pf:.2f} over {n} trades: the rule loses money after costs."
    if pf < 1.3 or t < 2.0:
        return "noise", f"out-of-sample profit factor {pf:.2f} over {n} trades ({t:.1f} standard errors from zero): indistinguishable from luck."
    shrink = ""
    if best_is and best_is.get("profit_factor"):
        ratio = min(pf, 10.0) / min(best_is["profit_factor"], 10.0)
        shrink = f" It kept {ratio:.0%} of its in-sample profit factor out of sample."
    return "worth a look", f"out-of-sample profit factor {pf:.2f} over {n} trades ({t:.1f} standard errors from zero).{shrink} Worth a paper-trade with a small size."


# ------------------------------------------------------------------- data
def _ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get_json(url: str, attempts: int = 6):
    """Binance returns 429 (and 418 after abuse) when a burst of requests exceeds its weight limit: wait and try again."""
    from bigbrain.net import HTTPStatusError, http_get

    delay = 2.0
    for attempt in range(attempts):
        try:
            return json.loads(http_get(url, headers={"Accept": "application/json"}).decode("utf-8"))
        except HTTPStatusError as exc:
            if exc.status not in (429, 418) or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("retry-after") or exc.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
            delay = min(delay * 2, 60.0)


def fetch_history(symbol: str, interval: str, days: int, market: str = "spot", cache_dir: str | Path | None = None, sleep: float = 0.25) -> list[Bar]:
    """``days`` of Binance candles, paginated (1,000 per request on spot, 1,500 on perps), cached on disk per day."""
    from bigbrain.ingest.market import BINANCE_HOSTS, FUTURES_HOST, parse_binance_klines

    cache = None
    if cache_dir:
        cache = Path(cache_dir) / f"{symbol.upper()}-{market}-{interval}-{days}d-{datetime.now(timezone.utc):%Y%m%d}.json"
        if cache.exists():
            return [Bar(**row) for row in json.loads(cache.read_text())]
    step = INTERVAL_SECONDS[interval] * 1000
    now_ms = int(time.time() * 1000)
    cursor = now_ms - days * 86400 * 1000
    limit = 1500 if market == "perps" else 1000
    base = f"{FUTURES_HOST}/fapi/v1/klines" if market == "perps" else f"{BINANCE_HOSTS[0]}/api/v3/klines"
    bars: list[Bar] = []
    while cursor < now_ms:
        url = f"{base}?symbol={quote(symbol.upper())}&interval={interval}&startTime={cursor}&limit={limit}"
        chunk = parse_binance_klines(_get_json(url))
        if not chunk:
            break
        bars += chunk
        cursor = _ms(chunk[-1].date) + step
        if len(chunk) < limit:
            break
        time.sleep(sleep)
    seen, unique = set(), []
    for b in bars:
        if b.date not in seen:
            seen.add(b.date)
            unique.append(b)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([b.__dict__ for b in unique]))
    return unique


def fetch_funding_history(symbol: str, interval: str, days: int, cache_dir: str | Path | None = None, sleep: float = 0.25) -> dict[str, float]:
    """Historical funding rates on a perp, keyed by the candle (of ``interval``) that opens at or contains the funding time.
    Daily candles collect all three of the day's payments."""
    from bigbrain.ingest.market import FUTURES_HOST

    cache = None
    if cache_dir:
        cache = Path(cache_dir) / f"{symbol.upper()}-funding-{interval}-{days}d-{datetime.now(timezone.utc):%Y%m%d}.json"
        if cache.exists():
            return json.loads(cache.read_text())
    step = INTERVAL_SECONDS[interval]
    now_ms = int(time.time() * 1000)
    cursor = now_ms - days * 86400 * 1000
    out: dict[str, float] = {}
    while cursor < now_ms:
        url = f"{FUTURES_HOST}/fapi/v1/fundingRate?symbol={quote(symbol.upper())}&startTime={cursor}&limit=1000"
        rows = _get_json(url)
        if not rows:
            break
        for r in rows:
            ts = int(r["fundingTime"]) // 1000
            key = datetime.fromtimestamp(ts - ts % step, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            out[key] = out.get(key, 0.0) + float(r["fundingRate"])
        cursor = int(rows[-1]["fundingTime"]) + 1
        if len(rows) < 1000:
            break
        time.sleep(sleep)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


def resolve_symbols(spec: str, market: str) -> list[str]:
    """'BTCUSDT,ETHUSDT', 'top:30' (by 24h volume on the chosen market), or 'top:31-80' (a slice: pairs a hypothesis was not found on)."""
    if spec.startswith("top:"):
        from bigbrain.ingest.market import top_usdt_pairs, top_usdt_perps

        lo, _, hi = spec[4:].partition("-")
        first, last = (int(lo), int(hi)) if hi else (1, int(lo))
        ranked = top_usdt_perps(last) if market == "perps" else top_usdt_pairs(last)
        return [u["symbol"] for u in ranked[first - 1:last]]
    return [s.strip().upper() for s in spec.split(",") if s.strip()]


def rank_at_start(markets: Markets, n: int, days: int = 30, min_history: float = 0.8) -> Markets:
    """Keep the ``n`` symbols with the most quote volume over the first ``days`` of the history, among those that
    existed for at least ``min_history`` of it.

    A universe chosen by today's volume is a list of coins that already went up: a long-only rule tested on it
    looks better than it is, and so does the buy-and-hold benchmark. Ranking at the start uses only what was
    known then. Coins that were delisted since are still missing, so the bias is reduced, not gone."""
    if not markets:
        return markets
    first = min(bars[0].date for bars in markets.values())
    last = max(bars[-1].date for bars in markets.values())
    span = max(1, len({b.date for bars in markets.values() for b in bars}))
    per_day = max(1, int(86400 / INTERVAL_SECONDS.get(_interval_of(markets), 86400)))
    early_cut = days * per_day
    scored = []
    for s, bars in markets.items():
        if bars[0].date > first or len(bars) < min_history * span:
            continue  # listed later, or not around for enough of the period
        early = bars[:early_cut]
        scored.append((sum(b.quote_volume or b.volume * b.close for b in early), s))
    keep = {s for _, s in sorted(scored, reverse=True)[:n]}
    return {s: bars for s, bars in markets.items() if s in keep}


def _interval_of(markets: Markets) -> str:
    for bars in markets.values():
        if len(bars) > 1:
            gap = (_ms(bars[1].date) - _ms(bars[0].date)) // 1000
            for name, seconds in INTERVAL_SECONDS.items():
                if seconds == gap:
                    return name
    return "1d"


def fetch_universe(symbols: list[str], interval: str, days: int, market: str, cache_dir=None, workers: int = 4, funding: bool = False, log=None) -> tuple[Markets, dict]:
    """Candles for every symbol (and funding history when asked), fetched in parallel. Symbols that fail are dropped with a note."""
    markets: Markets = {}
    fund: dict[str, dict[str, float]] = {}

    def one(symbol: str):
        try:
            bars = fetch_history(symbol, interval, days, market, cache_dir)
            f = fetch_funding_history(symbol, interval, days, cache_dir) if funding and market == "perps" else None
            return symbol, bars, f, None
        except Exception as exc:  # noqa: BLE001
            return symbol, None, None, exc

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for symbol, bars, f, err in pool.map(one, symbols):
            if err is not None or not bars:
                if log:
                    log(f"  skipping {symbol}: {err or 'no candles'}")
                continue
            markets[symbol] = bars
            if f is not None:
                fund[symbol] = f
    return markets, {"funding": fund} if funding else {}


# ----------------------------------------------------------------- report
def run(rule_name: str, data, grid: dict[str, list], costs: Costs, interval: str, folds: int = 6, workers: int = 4, extras: dict | None = None) -> dict:
    """Grid, plateau and walk-forward in one go, on one market or a universe."""
    markets = _as_markets(data)
    rows = grid_search(rule_name, data, grid, costs, interval, workers, extras=extras)
    table = plateau(rows, grid)
    table.sort(key=lambda e: (-e["plateau"], -e["result"].tstat))
    best_fit = max(rows, key=lambda pr: _score(pr[1]))
    n_dates = len({b.date for bars in markets.values() for b in bars})
    wf = walk_forward(rule_name, data, grid, costs, interval, folds, workers, extras) if n_dates >= folds * 50 else None
    label, sentence = verdict(wf["out_of_sample"], best_fit[1].as_dict()) if wf else ("insufficient", "not enough candles for a walk-forward.")
    return {"rule": rule_name, "grid": grid, "candles": n_dates, "symbols": sorted(markets), "costs": costs.__dict__, "table": table,
            "best_fit": {"params": best_fit[0], "result": best_fit[1].as_dict()}, "walk_forward": wf, "verdict": label, "sentence": sentence}


def describe_params(p: dict) -> str:
    return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def _universe_name(report: dict, symbol: str) -> str:
    syms = report.get("symbols") or [symbol]
    if len(syms) == 1:
        return symbol if syms == ["_"] else syms[0]
    return f"{len(syms)} pairs ({', '.join(syms[:4])}{', ...' if len(syms) > 4 else ''})"


def learn_result(brain: Brain, report: dict, symbol: str, interval: str, days: int) -> str:
    """The brain remembers what the lab found, as evidence it can recall."""
    best, wf = report["best_fit"], report["walk_forward"]
    top = report["table"][0]
    where = _universe_name(report, symbol)
    text = (
        f"Lab test of the '{report['rule']}' rule ({RULES[report['rule']]['doc']}) on {where}, {interval} candles over {days} days ({report['candles']} candles), "
        f"with costs of {report['costs']['fee']:.2%} fee per side, {report['costs']['spread_bps']:g} bp spread and {report['costs']['slippage_bps']:g} bp slippage. "
        f"Across {len(report['table'])} parameter settings the best in-sample fit was {describe_params(best['params'])} with profit factor {min(best['result']['profit_factor'], 99):.2f} "
        f"over {best['result']['trades']} trades; the most robust setting (its neighbours work too) was {describe_params(top['params'])} with plateau score {top['plateau']:.2f}. "
    )
    if wf:
        oos = wf["out_of_sample"]
        text += (f"Walk-forward over {wf['folds']} windows, choosing parameters on the past only: out-of-sample profit factor {min(oos['profit_factor'], 99):.2f}, "
                 f"expectancy {oos['expectancy']:+.3%} per trade over {oos['trades']} trades, {oos['tstat']:.1f} standard errors from zero, maximum drawdown {oos['max_drawdown']:.1%}, "
                 f"net {oos['net_return']:+.1%} against {wf['benchmark']['net_return']:+.1%} for simply holding the same universe over the same span. ")
    text += f"Verdict: {report['verdict']}. {report['sentence']} A profit factor near one after costs means the rule has no edge on this market and timeframe, whatever a single week's chart suggests."
    title = f"Lab: {report['rule']} on {where} {interval} ({report['verdict']})"
    brain.forget(title=title)
    concepts = ["backtesting", "overfitting", "walk-forward", "profit factor", report["rule"]] + [s.lower() for s in (report.get("symbols") or [symbol])[:10] if s != "_"]
    cell, _ = brain.learn("lesson", title, text, source="lab", extra_concepts=concepts)
    brain.commit()
    return cell.title


def format_report(report: dict, symbol: str, interval: str, top: int = 10) -> str:
    where = _universe_name(report, symbol)
    lines = [f"lab: {report['rule']} on {where}, {interval} candles ({report['candles']} dates), costs fee {report['costs']['fee']:.2%}/side, spread {report['costs']['spread_bps']:g} bp, slippage {report['costs']['slippage_bps']:g} bp",
             "", f"grid: {len(report['table'])} settings, ranked by plateau score (own profit factor or its neighbours' mean, whichever is lower)",
             f"  {'params':44} {'trades':>6} {'win':>5} {'PF':>6} {'expect':>8} {'net':>8} {'maxDD':>7} {'t':>5} {'nbrs':>6} {'plateau':>8}"]
    for e in report["table"][:top]:
        r = e["result"]
        pf = f"{min(r.profit_factor, 99):6.2f}"
        lines.append(f"  {describe_params(e['params']):44} {r.trades:6} {r.wins / r.trades if r.trades else 0:5.0%} {pf} {r.expectancy:+8.3%} {r.net_return:+8.1%} {r.max_drawdown:7.1%} {r.tstat:5.1f} {e['neighbours']:6.2f} {e['plateau']:8.2f}")
    b = report["best_fit"]
    lines.append(f"  best in-sample fit: {describe_params(b['params'])}  PF {min(b['result']['profit_factor'], 99):.2f} over {b['result']['trades']} trades (this is the number a screenshot shows; it is not evidence)")
    wf = report["walk_forward"]
    if wf:
        lines += ["", f"walk-forward: {wf['folds']} windows cut by time, parameters chosen on the past only, judged on the window that follows",
                  f"  {'window':>6} {'chosen':44} {'IS PF':>6} {'IS n':>5} {'OOS PF':>7} {'OOS n':>5} {'OOS expect':>10}  test until"]
        for s in wf["steps"]:
            i, o = s["in_sample"], s["out_of_sample"]
            lines.append(f"  {s['window']:6} {describe_params(s['params']):44} {min(i['profit_factor'], 99):6.2f} {i['trades']:5} {min(o['profit_factor'], 99):7.2f} {o['trades']:5} {o['expectancy']:+10.3%}  {s['test_to'][:10]}")
        o = wf["out_of_sample"]
        lines.append(f"  stitched out-of-sample: PF {min(o['profit_factor'], 99):.2f}, expectancy {o['expectancy']:+.3%}/trade, {o['trades']} trades, net {o['net_return']:+.1%}, maxDD {o['max_drawdown']:.1%}, Sharpe {o['sharpe']:.2f}, {o['tstat']:.1f} standard errors from zero")
        bh = wf.get("benchmark")
        if bh and bh["symbols"]:
            lines.append(f"  benchmark, just holding the universe over the same span: net {bh['net_return']:+.1%}, maxDD {bh['max_drawdown']:.1%}  (a long-only rule must beat this to be worth anything)")
    lines += ["", f"verdict: {report['verdict']}. {report['sentence']}"]
    return "\n".join(lines)
