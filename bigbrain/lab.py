"""The lab: test a trading hypothesis the hard way, so the brain never has to pay for it live.

A hypothesis is a *rule*: a function of past candles and a few parameters that says, after
each candle closes, what position to hold next and where its stop sits. The lab runs a rule
over months of candles with the same honesty as the live trader:

* decisions after the close, market orders filled at the next candle's open;
* stops and stop-entries are resting orders that fill at their level, or at the open when
  a candle gaps through it;
* every fill pays the fee, half the spread and slippage, both ways.

Then it does the three things a backtest screenshot never does:

* **grid**: every parameter combination, with a *plateau* score. A real edge is a region
  where neighbouring settings all work; a lone spike is one lucky fit to one stretch of noise.
* **walk-forward**: parameters are chosen on the past only and judged on the window that
  follows, window after window. The stitched out-of-sample record is the only number that
  says anything about the future.
* **memory**: the verdict becomes a lesson cell, so the brain can answer "does parabolic SAR
  work on one-minute EURUSD" from evidence instead of folklore.

Nothing here can tune a rule to catch tops and bottoms. Anything that appears to has been fitted
to the past, and the walk-forward exists to show that.
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

from bigbrain.brain import Brain
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import Bar
from bigbrain.watch import INTERVAL_SECONDS

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
# A rule maps (bars, params) to one decision per candle, made after that candle closed and
# acting in the next one: {"target": +1|-1|0, "stop": level or None, "reverse": bool}.
# With reverse=True a stop that fills also opens the opposite side at the same level (a
# stop-and-reverse system such as parabolic SAR).

RULES: dict[str, dict] = {}


def rule(name: str, defaults: dict, doc: str):
    def wrap(fn):
        RULES[name] = {"fn": fn, "defaults": defaults, "doc": doc}
        return fn

    return wrap


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
def psar_rule(bars: list[Bar], params: dict) -> list[dict]:
    series = psar_series(bars, params["start"], params["increment"], params["maximum"])
    out = []
    for i, (sar, up, nxt) in enumerate(series):
        if i == 0 or math.isnan(nxt):
            out.append({"target": 0, "stop": None, "reverse": False})
        else:
            out.append({"target": 1 if up else -1, "stop": nxt, "reverse": True})
    return out


@rule("sma_cross", {"fast": 20, "slow": 50}, "moving-average crossover: long above, short below, market orders at the next open")
def sma_cross_rule(bars: list[Bar], params: dict) -> list[dict]:
    closes = [b.close for b in bars]
    fast, slow = ind.sma(closes, int(params["fast"])), ind.sma(closes, int(params["slow"]))
    out = []
    for f, s in zip(fast, slow):
        target = 0 if f is None or s is None else (1 if f > s else -1)
        out.append({"target": target, "stop": None, "reverse": False})
    return out


@rule("rsi_reversion", {"period": 14, "entry": 30.0, "exit": 55.0, "stop_pct": 0.02},
      "long when RSI falls below entry, out when it recovers past exit, with a fixed stop")
def rsi_reversion_rule(bars: list[Bar], params: dict) -> list[dict]:
    closes = [b.close for b in bars]
    r = ind.rsi(closes, int(params["period"]))
    out, holding, entry_px = [], False, 0.0
    for i, v in enumerate(r):
        if v is None:
            out.append({"target": 0, "stop": None, "reverse": False})
            continue
        if holding and v >= params["exit"]:
            holding = False
        elif not holding and v <= params["entry"]:
            holding, entry_px = True, closes[i]
        out.append({"target": 1 if holding else 0, "stop": entry_px * (1 - params["stop_pct"]) if holding else None, "reverse": False})
    return out


# -------------------------------------------------------------- simulator
@dataclass
class Trade:
    side: int
    entry_i: int
    entry: float  # fill, costs included
    exit_i: int
    exit: float
    ret: float  # net return on notional, fees included
    reason: str


@dataclass
class Result:
    trades: int = 0
    wins: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # mean net return per trade
    net_return: float = 0.0  # compounded, position = `size` x equity
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    tstat: float = 0.0  # expectancy / standard error: how many standard errors the mean is from zero
    log: list = field(default_factory=list)  # Trade records

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "log"}


def simulate(bars: list[Bar], decisions: list[dict], costs: Costs, interval: str = "15m", size: float = 1.0, start: int = 0) -> Result:
    """Run one rule's decisions through the market, strictly in order. ``start``: ignore trades entered before this index."""
    pos: dict | None = None
    equity, curve, trades = 1.0, [], []

    def close_at(price: float, i: int, reason: str) -> None:
        nonlocal pos, equity
        fill = costs.fill(price, -pos["side"])
        gross = pos["side"] * (fill / pos["entry"] - 1)
        ret = gross - costs.fee - costs.fee * fill / pos["entry"]
        if pos["entry_i"] >= start:
            trades.append(Trade(pos["side"], pos["entry_i"], pos["entry"], i, fill, ret, reason))
            equity *= 1 + size * ret
        pos = None

    def open_at(price: float, i: int, side: int) -> None:
        nonlocal pos
        pos = {"side": side, "entry": costs.fill(price, side), "entry_i": i}

    for i in range(1, len(bars)):
        d, bar = decisions[i - 1], bars[i]
        target = d.get("target", 0)
        if pos and target != pos["side"]:
            close_at(bar.open, i, "signal")
        if pos is None and target:
            open_at(bar.open, i, target)
        stop = d.get("stop")
        if pos and stop is not None:
            hit = bar.low <= stop if pos["side"] == 1 else bar.high >= stop
            if hit:
                gapped = bar.open < stop if pos["side"] == 1 else bar.open > stop
                level = bar.open if gapped else stop
                side = pos["side"]
                close_at(level, i, "stop")
                if d.get("reverse"):
                    open_at(level, i, -side)
        mark = equity
        if pos and pos["entry_i"] >= start:
            mark = equity * (1 + size * pos["side"] * (bar.close / pos["entry"] - 1))
        curve.append(mark)
    if pos and pos["entry_i"] >= start:
        close_at(bars[-1].close, len(bars) - 1, "end")
    return summarize(trades, curve, interval)


def summarize(trades: list[Trade], curve: list[float], interval: str) -> Result:
    r = Result(log=trades, trades=len(trades))
    if not trades:
        return r
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
    r.net_return = curve[-1] - 1 if curve else 0.0
    r.max_drawdown = ind.max_drawdown(curve) if curve else 0.0
    per_bar = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve)) if curve[i - 1] > 0]
    r.sharpe = ind.sharpe(per_bar, periods_per_year=int(365 * 86400 / INTERVAL_SECONDS[interval]))
    return r


def evaluate(rule_name: str, bars: list[Bar], params: dict, costs: Costs, interval: str = "15m", start: int = 0) -> Result:
    spec = RULES[rule_name]
    return simulate(bars, spec["fn"](bars, {**spec["defaults"], **params}), costs, interval, start=start)


# ------------------------------------------------------------------- grid
def parse_grid(text: str, defaults: dict) -> dict[str, list]:
    """'start=0.01,0.02;maximum=0.1,0.2' -> {'start': [0.01, 0.02], 'maximum': [0.1, 0.2]}; missing parameters keep their default."""
    grid: dict[str, list] = {k: [v] for k, v in defaults.items()}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        key, _, values = part.partition("=")
        key = key.strip()
        if key not in defaults:
            raise ValueError(f"unknown parameter '{key}'; this rule has {sorted(defaults)}")
        grid[key] = sorted({type(defaults[key])(v) for v in values.split(",") if v.strip()})
    return grid


def grid_points(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def grid_search(rule_name: str, bars: list[Bar], grid: dict[str, list], costs: Costs, interval: str = "15m", workers: int = 4, start: int = 0) -> list[tuple[dict, Result]]:
    points = grid_points(grid)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda p: evaluate(rule_name, bars, p, costs, interval, start), points))
    return list(zip(points, results))


def _score(r: Result) -> float:
    """What a fit is judged by: how many standard errors the mean trade is above zero (few lucky trades score low)."""
    return r.tstat if r.trades >= 20 else -abs(r.tstat) - 1.0


def plateau(rows: list[tuple[dict, Result]], grid: dict[str, list]) -> list[dict]:
    """For every grid point: its own profit factor, its neighbours' mean, and the plateau score (the lower of the two).

    Neighbours differ in exactly one parameter by one step. A setting whose neighbours fail is a spike, not an edge."""
    index = {tuple(p[k] for k in grid): r for p, r in rows}
    keys = list(grid)
    out = []
    for p, r in rows:
        neighbours = []
        for k in keys:
            values = grid[k]
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
def walk_forward(rule_name: str, bars: list[Bar], grid: dict[str, list], costs: Costs, interval: str = "15m", folds: int = 6, workers: int = 4) -> dict:
    """Anchored walk-forward: for each window after the first, choose parameters on everything before it,
    then trade that window with them. The stitched out-of-sample trades are the honest record."""
    if folds < 2 or len(bars) < folds * 50:
        raise ValueError("walk-forward needs at least two windows of fifty candles each")
    window = len(bars) // folds
    steps, oos_trades, oos_curve_parts = [], [], []
    for k in range(1, folds):
        train_end, test_end = k * window, (k + 1) * window if k < folds - 1 else len(bars)
        fitted = grid_search(rule_name, bars[:train_end], grid, costs, interval, workers)
        best_params, best = max(fitted, key=lambda pr: _score(pr[1]))
        test = evaluate(rule_name, bars[:test_end], best_params, costs, interval, start=train_end)  # history for warm-up, trades only in the window
        steps.append({"window": k, "train_candles": train_end, "test_candles": test_end - train_end, "params": best_params,
                      "in_sample": best.as_dict(), "out_of_sample": test.as_dict()})
        oos_trades += test.log
    stitched = summarize(oos_trades, _stitch_curve(oos_trades), interval)
    return {"folds": folds, "steps": steps, "out_of_sample": stitched.as_dict(), "oos_trades": oos_trades}


def _stitch_curve(trades: list[Trade]) -> list[float]:
    equity, curve = 1.0, [1.0]
    for t in sorted(trades, key=lambda t: t.exit_i):
        equity *= 1 + t.ret
        curve.append(equity)
    return curve


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
def fetch_history(symbol: str, interval: str, days: int, market: str = "spot", cache_dir: str | Path | None = None, sleep: float = 0.25) -> list[Bar]:
    """``days`` of Binance candles, paginated (1,000 per request on spot, 1,500 on perps), cached on disk per day."""
    from bigbrain.ingest.market import BINANCE_HOSTS, FUTURES_HOST, parse_binance_klines
    from bigbrain.net import http_get

    cache = None
    if cache_dir:
        cache = Path(cache_dir) / f"{symbol.upper()}-{market}-{interval}-{days}d-{datetime.now(timezone.utc):%Y%m%d}.json"
        if cache.exists():
            return [Bar(**row) for row in json.loads(cache.read_text())]
    step = INTERVAL_SECONDS[interval] * 1000
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86400 * 1000
    limit = 1500 if market == "perps" else 1000
    base = f"{FUTURES_HOST}/fapi/v1/klines" if market == "perps" else f"{BINANCE_HOSTS[0]}/api/v3/klines"
    bars: list[Bar] = []
    cursor = start_ms
    while cursor < now_ms:
        url = f"{base}?symbol={symbol.upper()}&interval={interval}&startTime={cursor}&limit={limit}"
        chunk = parse_binance_klines(json.loads(http_get(url, headers={"Accept": "application/json"}).decode("utf-8")))
        if not chunk:
            break
        bars += chunk
        cursor = int(datetime.strptime(chunk[-1].date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000) + step
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


# ----------------------------------------------------------------- report
def run(rule_name: str, bars: list[Bar], grid: dict[str, list], costs: Costs, interval: str, folds: int = 6, workers: int = 4) -> dict:
    """Grid, plateau and walk-forward in one go."""
    rows = grid_search(rule_name, bars, grid, costs, interval, workers)
    table = plateau(rows, grid)
    table.sort(key=lambda e: (-e["plateau"], -e["result"].tstat))
    best_fit = max(rows, key=lambda pr: _score(pr[1]))
    wf = walk_forward(rule_name, bars, grid, costs, interval, folds, workers) if len(grid_points(grid)) and len(bars) >= folds * 50 else None
    label, sentence = verdict(wf["out_of_sample"], best_fit[1].as_dict()) if wf else ("insufficient", "not enough candles for a walk-forward.")
    return {"rule": rule_name, "grid": grid, "candles": len(bars), "costs": costs.__dict__, "table": table, "best_fit": {"params": best_fit[0], "result": best_fit[1].as_dict()},
            "walk_forward": wf, "verdict": label, "sentence": sentence}


def describe_params(p: dict) -> str:
    return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def learn_result(brain: Brain, report: dict, symbol: str, interval: str, days: int) -> str:
    """The brain remembers what the lab found, as evidence it can recall."""
    best, wf = report["best_fit"], report["walk_forward"]
    top = report["table"][0]
    text = (
        f"Lab test of the '{report['rule']}' rule ({RULES[report['rule']]['doc']}) on {symbol} {interval} candles over {days} days ({report['candles']} candles), "
        f"with costs of {report['costs']['fee']:.2%} fee per side, {report['costs']['spread_bps']:g} bp spread and {report['costs']['slippage_bps']:g} bp slippage. "
        f"Across {len(report['table'])} parameter settings the best in-sample fit was {describe_params(best['params'])} with profit factor {min(best['result']['profit_factor'], 99):.2f} "
        f"over {best['result']['trades']} trades; the most robust setting (its neighbours work too) was {describe_params(top['params'])} with plateau score {top['plateau']:.2f}. "
    )
    if wf:
        oos = wf["out_of_sample"]
        text += (f"Walk-forward over {wf['folds']} windows, choosing parameters on the past only: out-of-sample profit factor {min(oos['profit_factor'], 99):.2f}, "
                 f"expectancy {oos['expectancy']:+.3%} per trade over {oos['trades']} trades, {oos['tstat']:.1f} standard errors from zero. ")
    text += f"Verdict: {report['verdict']}. {report['sentence']} A profit factor near one after costs means the rule has no edge on this market and timeframe, whatever a single week's chart suggests."
    title = f"Lab: {report['rule']} on {symbol} {interval} ({report['verdict']})"
    brain.forget(title=title)
    cell, _ = brain.learn("lesson", title, text, source="lab", extra_concepts=["backtesting", "overfitting", "walk-forward", "profit factor", symbol.lower(), report["rule"]])
    brain.commit()
    return cell.title


def format_report(report: dict, symbol: str, interval: str, top: int = 10) -> str:
    lines = [f"lab: {report['rule']} on {symbol} {interval}, {report['candles']} candles, costs fee {report['costs']['fee']:.2%}/side, spread {report['costs']['spread_bps']:g} bp, slippage {report['costs']['slippage_bps']:g} bp",
             "", f"grid: {len(report['table'])} settings, ranked by plateau score (own profit factor or its neighbours' mean, whichever is lower)",
             f"  {'params':40} {'trades':>6} {'win':>5} {'PF':>6} {'expect':>8} {'net':>8} {'maxDD':>7} {'t':>5} {'nbrs':>6} {'plateau':>8}"]
    for e in report["table"][:top]:
        r = e["result"]
        pf = f"{min(r.profit_factor, 99):6.2f}"
        lines.append(f"  {describe_params(e['params']):40} {r.trades:6} {r.wins / r.trades if r.trades else 0:5.0%} {pf} {r.expectancy:+8.3%} {r.net_return:+8.1%} {r.max_drawdown:7.1%} {r.tstat:5.1f} {e['neighbours']:6.2f} {e['plateau']:8.2f}")
    b = report["best_fit"]
    lines.append(f"  best in-sample fit: {describe_params(b['params'])}  PF {min(b['result']['profit_factor'], 99):.2f} over {b['result']['trades']} trades (this is the number a screenshot shows; it is not evidence)")
    wf = report["walk_forward"]
    if wf:
        lines += ["", f"walk-forward: {wf['folds']} windows, parameters chosen on the past only, judged on the window that follows",
                  f"  {'window':>6} {'chosen':40} {'IS PF':>6} {'IS n':>5} {'OOS PF':>7} {'OOS n':>5} {'OOS expect':>10}"]
        for s in wf["steps"]:
            i, o = s["in_sample"], s["out_of_sample"]
            lines.append(f"  {s['window']:6} {describe_params(s['params']):40} {min(i['profit_factor'], 99):6.2f} {i['trades']:5} {min(o['profit_factor'], 99):7.2f} {o['trades']:5} {o['expectancy']:+10.3%}")
        o = wf["out_of_sample"]
        lines.append(f"  stitched out-of-sample: PF {min(o['profit_factor'], 99):.2f}, expectancy {o['expectancy']:+.3%}/trade, {o['trades']} trades, maxDD {o['max_drawdown']:.1%}, {o['tstat']:.1f} standard errors from zero")
    lines += ["", f"verdict: {report['verdict']}. {report['sentence']}"]
    return "\n".join(lines)
