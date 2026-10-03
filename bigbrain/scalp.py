"""Scalping: short-horizon setups with a stop, a target and a time limit, tested before anyone trusts them.

A scalper lives on a few basis points per trade, so costs decide everything. A setup that moves
the price 6 bp in its favour on average is a winner at a 4 bp round trip and a slow bleed at
12 bp. Every scan therefore reports two numbers next to the verdict: the *gross edge* (what the
setup captured before fees, spread, slippage and funding) and what the trading cost, so it is
plain whether a setup failed because it has no edge or because the edge is smaller than the toll.

The setups are the three families that short-horizon traders actually argue about:

* ``vwap_fade``: price stretched far from its rolling volume-weighted average tends to snap back.
  Enter against the stretch once it starts to turn, target the VWAP itself.
* ``sweep``: a candle runs the stops beyond the recent low (or high) and closes back inside the
  range. The breakout failed, the trapped side has to cover. Stop beyond the wick, target a
  multiple of the risk.
* ``squeeze``: a volatility squeeze (Bollinger bandwidth near its lowest in a while) resolves with
  a close outside the band on heavy volume. Go with the break.

Each is a lab rule, so ``bigbrain lab --rule sweep ...`` works too. Every setup can be filtered
by a trend average (``trend``: EMA period, 0 for off) and restricted to one side (``side``: 1 long
only, -1 short only, 0 both; spot is long only).

The bracket (stop, target, time limit) is tracked inside the rule with the simulator's own fill
order (the stop is checked before the target when one candle touches both), so the rule's idea
of whether it is in a position never drifts from the simulator's.

``bigbrain scalp`` runs every setup across symbols and intervals through the lab's grid and
walk-forward, ranks them by out-of-sample evidence, and writes each verdict into the brain.
Nothing here promises consistency: most scans end in "no edge", and the scan exists to say so
before money does.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import deque
from typing import Callable

from bigbrain import lab
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import Bar

Entry = Callable[[int], "tuple[int, float, float] | None"]  # candle index -> (side, stop, take) or nothing

SCALP_RULES = ("vwap_fade", "sweep", "squeeze")

# the settings a scan chooses from: small on purpose, every extra setting is one more chance to fit noise
SCAN_GRIDS = {
    "vwap_fade": "window=30,60,120;entry_z=2,2.5,3;sl_atr=1,2",
    "sweep": "lookback=20,40,80;tp_r=1,1.5,2;trend=0,200",
    "squeeze": "squeeze_pct=0.1,0.2;tp_atr=1,2,3;sl_atr=1,1.5",
}


# ------------------------------------------------------------- series
_CACHE: dict[tuple, list] = {}


def _cached(bars: list[Bar], name: str, period: int, make: Callable[[], list]) -> list:
    """A grid runs the same candles through many settings; compute each series once."""
    key = (bars[0].date, bars[-1].date, len(bars), bars[-1].close, name, period)
    hit = _CACHE.get(key)
    if hit is None:
        if len(_CACHE) > 256:
            _CACHE.clear()
        hit = _CACHE[key] = make()
    return hit


def _atr(bars: list[Bar], period: int = 14) -> list:
    return _cached(bars, "atr", period, lambda: ind.atr([b.high for b in bars], [b.low for b in bars], [b.close for b in bars], period))


def _ema(bars: list[Bar], period: int) -> list:
    return _cached(bars, "ema", period, lambda: ind.ema([b.close for b in bars], period))


def rolling_vwap_z(bars: list[Bar], window: int) -> tuple[list, list]:
    """Rolling VWAP of the typical price over ``window`` candles, and how many standard deviations the close sits from it.

    Running sums keep it linear in the number of candles, which matters on a month of one-minute data."""

    def make() -> list:
        vwap: list = [None] * len(bars)
        z: list = [None] * len(bars)
        pv = vol = 0.0
        devs: deque = deque()  # the last `window` deviations of the close from the VWAP
        dev_sum = dev_sq = 0.0
        for i, b in enumerate(bars):
            pv += (b.high + b.low + b.close) / 3 * b.volume
            vol += b.volume
            if i >= window:
                old = bars[i - window]
                pv -= (old.high + old.low + old.close) / 3 * old.volume
                vol -= old.volume
            if i < window - 1 or vol <= 0:
                continue
            vwap[i] = pv / vol
            d = b.close - vwap[i]
            devs.append(d)
            dev_sum += d
            dev_sq += d * d
            if len(devs) > window:
                gone = devs.popleft()
                dev_sum -= gone
                dev_sq -= gone * gone
            if len(devs) == window:
                mean = dev_sum / window
                sd = math.sqrt(max(dev_sq / window - mean * mean, 0.0))
                z[i] = d / sd if sd > 0 else None
        return [vwap, z]

    vwap, z = _cached(bars, "vwap_z", window, make)
    return vwap, z


def bandwidth(bars: list[Bar], period: int = 20, width: float = 2.0) -> tuple[list, list, list]:
    """Bollinger bands and their width relative to the middle band."""

    def make() -> list:
        upper, mid, lower = ind.bollinger([b.close for b in bars], period, width)
        bw = [(u - l) / m if u is not None and m else None for u, m, l in zip(upper, mid, lower)]
        return [upper, lower, bw]

    upper, lower, bw = _cached(bars, "bands", period, make)
    return upper, lower, bw


def _volume_avg(bars: list[Bar], period: int = 20) -> list:
    return _cached(bars, "vol_sma", period, lambda: ind.sma([b.volume for b in bars], period))


def random_walk(symbol: str = "WALK", n: int = 3000, seed: int = 7, vol: float = 0.002, steps: int = 60) -> list[Bar]:
    """A driftless random walk whose candles are built from ``steps`` ticks each, so the high and the low lie on the path.

    Scalping setups need this: on candles whose wicks are drawn independently of the path (or that have none), a stop
    that is crossed is a stop the price keeps running through, and filling it at its level books a phantom edge
    of several basis points per trade. On this walk nothing can earn anything, which is what a demo should show."""
    rng = random.Random(seed)
    bars, price = [], 100.0
    step_vol = vol / math.sqrt(steps)
    for i in range(n):
        o = high = low = price
        for _ in range(steps):
            price *= math.exp(rng.gauss(-step_vol * step_vol / 2, step_vol))  # the -sigma^2/2 keeps the price itself a martingale
            high, low = max(high, price), min(low, price)
        bars.append(Bar(f"D{i:05d}", o, high, low, price, 1_000_000 * (0.5 + rng.random()) * (1 + abs(math.log(price / o)) / vol)))  # big candles trade more, as they do
    return bars


# ------------------------------------------------------------- bracket
def bracketed(bars: list[Bar], entry: Entry, max_bars: int) -> list[dict]:
    """Turn entry signals into decisions with a resting stop and target and a time limit.

    A decision made after candle i acts from candle i+1, where the simulator checks the stop first and then the
    target; this walks the same candles in the same order, so the rule is flat exactly when the simulator is."""
    out: list[dict] = []
    pos: dict | None = None
    for i, b in enumerate(bars):
        if pos:
            pos["held"] += 1
            side = pos["side"]
            stopped = b.low <= pos["stop"] if side == 1 else b.high >= pos["stop"]
            taken = b.high >= pos["take"] if side == 1 else b.low <= pos["take"]
            timed_out = not (stopped or taken) and max_bars and pos["held"] >= max_bars
            if stopped or taken or timed_out:
                pos = None
            if timed_out:  # a time exit is a market order at the next open: the decision must say flat, so no re-entry this candle
                out.append(lab._flat())
                continue
        if pos is None and i > 0:
            sig = entry(i)
            if sig:
                side, stop, take = sig
                close = b.close
                if side and (stop - close) * side < 0 < (take - close) * side:  # stop behind the price, target ahead of it
                    pos = {"side": side, "stop": stop, "take": take, "held": 0}
        if pos:
            out.append({"target": pos["side"], "stop": pos["stop"], "take": pos["take"], "reverse": False})
        else:
            out.append(lab._flat())
    return out


def _allowed(bars: list[Bar], i: int, side: int, params: dict) -> bool:
    """The side and trend filters every setup shares."""
    wanted = int(params.get("side", 0))
    if wanted and side != wanted:
        return False
    period = int(params.get("trend", 0))
    if period:
        e = _ema(bars, period)[i]
        if e is None or (bars[i].close - e) * side <= 0:
            return False
    return True


# --------------------------------------------------------------- setups
@lab.rule("vwap_fade", {"window": 60, "entry_z": 2.5, "sl_atr": 1.5, "max_bars": 30, "trend": 0, "side": 0},
          "scalp: fade a close stretched entry_z deviations from the rolling VWAP once it turns back, target the VWAP, stop sl_atr ATRs away, out after max_bars",
          hint="1m-15m, liquid pairs; read the gross edge against the cost")
def vwap_fade_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    vwap, z = rolling_vwap_z(bars, int(params["window"]))
    atr = _atr(bars)
    entry_z, sl = params["entry_z"], params["sl_atr"]

    def entry(i: int):
        if z[i] is None or z[i - 1] is None or not atr[i]:
            return None
        close = bars[i].close
        for side in (1, -1):
            stretched = -side * z[i - 1] >= entry_z  # a long needs a close far below the VWAP, a short far above
            turning = side * (z[i] - z[i - 1]) > 0
            if stretched and turning and _allowed(bars, i, side, params):
                return side, close - side * sl * atr[i], vwap[i]
        return None

    return bracketed(bars, entry, int(params["max_bars"]))


@lab.rule("sweep", {"lookback": 40, "tp_r": 1.5, "buffer_atr": 0.1, "max_bars": 20, "trend": 0, "side": 0},
          "scalp: a candle runs the stops beyond the last `lookback` candles' low (high) and closes back inside; enter against the sweep, stop past the wick, target tp_r times the risk",
          hint="1m-15m; the liquidity-grab reversal")
def sweep_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    lookback = int(params["lookback"])
    atr = _atr(bars)
    tp_r, buffer = params["tp_r"], params["buffer_atr"]
    lows = _cached(bars, "min_low", lookback, lambda: _rolling_extreme([b.low for b in bars], lookback, min))
    highs = _cached(bars, "max_high", lookback, lambda: _rolling_extreme([b.high for b in bars], lookback, max))

    def entry(i: int):
        b, a = bars[i], atr[i]
        if not a or lows[i] is None or b.high <= b.low:
            return None
        where = (b.close - b.low) / (b.high - b.low)  # 0 at the low of the candle, 1 at the high
        if b.low < lows[i] < b.close and where >= 0.5 and _allowed(bars, i, 1, params):
            stop = b.low - buffer * a
            risk = b.close - stop
            if risk >= 0.25 * a:
                return 1, stop, b.close + tp_r * risk
        if b.high > highs[i] > b.close and where <= 0.5 and _allowed(bars, i, -1, params):
            stop = b.high + buffer * a
            risk = stop - b.close
            if risk >= 0.25 * a:
                return -1, stop, b.close - tp_r * risk
        return None

    return bracketed(bars, entry, int(params["max_bars"]))


def _rolling_extreme(values: list[float], n: int, pick) -> list:
    """The extreme of the ``n`` values before each index (excluding it), or None before there are n."""
    out: list = [None] * len(values)
    window: deque = deque()  # indices, values monotone so the front is the extreme
    better = (lambda a, b: a <= b) if pick is min else (lambda a, b: a >= b)
    for i, v in enumerate(values):
        if i >= n:
            out[i] = values[window[0]]
        while window and better(v, values[window[-1]]):
            window.pop()
        window.append(i)
        if window[0] <= i - n:
            window.popleft()
    return out


@lab.rule("squeeze", {"lookback": 120, "squeeze_pct": 0.2, "vol_mult": 1.5, "sl_atr": 1.0, "tp_atr": 2.0, "max_bars": 30, "trend": 0, "side": 0},
          "scalp: after a Bollinger squeeze (bandwidth in its lowest squeeze_pct of the last `lookback` candles), go with a close outside the band on vol_mult times average volume",
          hint="5m-1h; the volatility breakout")
def squeeze_rule(bars: list[Bar], params: dict, extras: dict | None = None) -> list[dict]:
    lookback, pct = int(params["lookback"]), params["squeeze_pct"]
    upper, lower, bw = bandwidth(bars)
    vol_avg, atr = _volume_avg(bars), _atr(bars)
    rank = _cached(bars, "bw_rank", lookback, lambda: _rolling_rank(bw, lookback))

    def entry(i: int):
        b, a = bars[i], atr[i]
        if not a or upper[i] is None or rank[i - 1] is None or not vol_avg[i - 1]:
            return None
        if rank[i - 1] > pct or b.volume < params["vol_mult"] * vol_avg[i - 1]:
            return None
        side = 1 if b.close > upper[i] else -1 if b.close < lower[i] else 0
        if side and _allowed(bars, i, side, params):
            return side, b.close - side * params["sl_atr"] * a, b.close + side * params["tp_atr"] * a
        return None

    return bracketed(bars, entry, int(params["max_bars"]))


def _rolling_rank(values: list, n: int) -> list:
    """Where each value sits among the ``n`` known values before it (0 = below all of them, 1 = above all)."""
    out: list = [None] * len(values)
    order: deque = deque()  # the last n known values, oldest first
    ranked: list[float] = []  # the same values, sorted
    for i, v in enumerate(values):
        if v is None:
            continue
        if len(order) == n:
            out[i] = bisect.bisect_left(ranked, v) / n
            ranked.pop(bisect.bisect_left(ranked, order.popleft()))
        order.append(v)
        bisect.insort(ranked, v)
    return out


# --------------------------------------------------------------- costs
def cost_edge(trades: list[lab.Trade], costs: lab.Costs) -> dict:
    """Split each trade's result into what the setup captured and what trading it cost, in basis points per trade.

    The gross move is measured between the quoted prices (before half the spread and slippage on each fill); the
    cost is fees, spread, slippage and funding. ``breakeven_bps`` is the round-trip cost at which the setup stops
    paying: if you cannot trade cheaper than that, there is nothing to scalp."""
    if not trades:
        return {"gross_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0, "breakeven_bps": 0.0, "gross_win": 0.0, "gross_t": 0.0}
    per_fill = (costs.spread_bps / 2 + costs.slippage_bps) / 1e4
    gross = []
    for t in trades:
        quoted_entry = t.entry / (1 + t.side * per_fill)
        quoted_exit = t.exit / (1 - t.side * per_fill)
        gross.append(t.side * (quoted_exit / quoted_entry - 1))
    g = sum(gross) / len(gross)
    net = sum(t.ret for t in trades) / len(trades)
    se = math.sqrt(sum((x - g) ** 2 for x in gross) / (len(gross) - 1) / len(gross)) if len(gross) > 1 else 0.0
    return {"gross_bps": g * 1e4, "cost_bps": (g - net) * 1e4, "net_bps": net * 1e4, "breakeven_bps": g * 1e4,
            "gross_win": sum(1 for x in gross if x > 0) / len(gross), "gross_t": g / se if se > 0 else 0.0}


# ---------------------------------------------------------------- scan
def scan(markets_by_interval: dict[str, lab.Markets], costs: lab.Costs, rules=SCALP_RULES, grids: dict[str, str] | None = None,
         long_only: bool = False, folds: int = 6, workers: int = 4, extras_by_interval: dict | None = None, log=None) -> list[dict]:
    """Every setup on every interval: grid, plateau, walk-forward and verdict, plus where the edge went.

    Returns one row per (rule, interval), best evidence first."""
    rows = []
    for interval, markets in markets_by_interval.items():
        data = next(iter(markets.values())) if len(markets) == 1 else markets
        for name in rules:
            spec = lab.RULES[name]
            grid = lab.parse_grid((grids or {}).get(name, SCAN_GRIDS.get(name, "")), spec["defaults"])
            if long_only and "side" in grid:
                grid["side"] = [1]
            if log:
                log(f"  {name} on {interval}: {len(lab.grid_points(grid))} settings, {folds}-window walk-forward ...")
            report = lab.run(name, data, grid, costs, interval, folds=folds, workers=workers, extras=(extras_by_interval or {}).get(interval))
            wf = report["walk_forward"]
            oos = wf["out_of_sample"] if wf else {}
            trades = wf["oos_trades"] if wf else []
            days = _span_days(markets)
            rows.append({"rule": name, "interval": interval, "report": report, "verdict": report["verdict"], "oos": oos,
                         "edge": cost_edge(trades, costs), "per_day": len(trades) / max(days * (folds - 1) / folds, 1e-9) if days else 0.0,
                         "params": wf["steps"][-1]["params"] if wf else report["best_fit"]["params"]})
    order = {"worth a look": 0, "noise": 1, "no edge": 2, "insufficient": 3}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), -r["oos"].get("tstat", 0.0)))
    return rows


def _span_days(markets: lab.Markets) -> float:
    dates = sorted({b.date for bars in markets.values() for b in bars})
    try:
        return (lab._ms(dates[-1]) - lab._ms(dates[0])) / 86_400_000
    except (ValueError, IndexError):
        return 0.0  # synthetic candles: no calendar


def format_scan(rows: list[dict], costs: lab.Costs, label: str) -> str:
    rt = 2 * costs.fee * 1e4 + costs.spread_bps + 2 * costs.slippage_bps
    lines = [f"scalp scan: {label}; you pay about {rt:.1f} bp per round trip ({costs.fee:.3%} fee per side, {costs.spread_bps:g} bp spread, {costs.slippage_bps:g} bp slippage per fill)",
             "every number below is out of sample: parameters chosen on the past only, judged on the window that followed", "",
             f"  {'setup':10} {'tf':>4} {'trades':>6} {'/day':>5} {'win':>5} {'PF':>5} {'net/tr':>8} {'gross':>7} {'cost':>6} {'t':>5}  {'verdict':13} last chosen"]
    for r in rows:
        o, e = r["oos"], r["edge"]
        n = o.get("trades", 0)
        per_day = f"{r['per_day']:.1f}" if r["per_day"] else "-"  # synthetic candles have no calendar
        lines.append(f"  {r['rule']:10} {r['interval']:>4} {n:6} {per_day:>5} {(o.get('wins', 0) / n if n else 0):5.0%} {min(o.get('profit_factor', 0.0), 99):5.2f} "
                     f"{e['net_bps']:+7.1f}bp {e['gross_bps']:+6.1f}bp {e['cost_bps']:5.1f}bp {o.get('tstat', 0.0):5.1f}  {r['verdict']:13} {lab.describe_params(r['params'])}")
    lines += ["", "reading it: 'gross' is what the setup caught between quoted prices, 'cost' is fees, spread, slippage and funding."]
    best = rows[0] if rows else None
    if best and best["verdict"] == "worth a look":
        lines.append(f"the best candidate is {best['rule']} on {best['interval']}: paper-trade it small before believing it. Out-of-sample "
                     "edges shrink again live, and one good month is one sample.")
    else:
        edges = [r for r in rows if r["edge"]["gross_t"] >= 2.0 and r["oos"].get("trades", 0) >= 30]  # a real move before costs, not a lucky one
        if edges:
            top = max(edges, key=lambda r: r["edge"]["gross_bps"])
            lines.append(f"nothing survived after costs. The strongest edge before costs was {top['rule']} on {top['interval']} at {top['edge']['gross_bps']:+.1f} bp per trade "
                         f"({top['edge']['gross_t']:.1f} standard errors from zero) against "
                         f"{top['edge']['cost_bps']:.1f} bp of cost: it would only pay below about {top['edge']['breakeven_bps']:.1f} bp round trip (maker fees, a rebate tier, "
                         "a tighter market), or on a slower timeframe where each trade moves further.")
        else:
            lines.append("nothing survived, and no setup caught a move distinguishable from zero even before costs: on these markets and timeframes these setups are noise.")
    return "\n".join(lines)


def learn_scan(brain, rows: list[dict], label: str, days: int) -> list[str]:
    """Each verdict, and where its edge went, becomes a lesson the brain can recall."""
    titles = []
    for r in rows:
        title = lab.learn_result(brain, r["report"], label, r["interval"], days)
        e = r["edge"]
        cost_title = f"Scalp cost check: {r['rule']} on {label} {r['interval']}"
        brain.forget(title=cost_title)
        brain.learn("lesson", cost_title,
                    f"Out of sample the {r['rule']} scalp setup on {label} {r['interval']} caught {e['gross_bps']:+.1f} basis points per trade between quoted prices "
                    f"and paid {e['cost_bps']:.1f} basis points in fees, spread, slippage and funding, netting {e['net_bps']:+.1f} bp per trade over "
                    f"{r['oos'].get('trades', 0)} trades ({r['per_day']:.1f} a day). It breaks even at about {e['breakeven_bps']:.1f} bp round-trip cost. "
                    f"For scalping, the cost per round trip decides whether a small edge survives. Verdict: {r['verdict']}.",
                    source="lab", extra_concepts=["scalping", "transaction costs", "market microstructure", r["rule"]])
        titles.append(title)
    brain.commit()
    return titles
