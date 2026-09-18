"""Exit learning: the brain tests alternative exits on every trade it closes.

A position keeps its bar path (high, low, close since entry). When it closes,
``simulate`` replays that path under several exit variants and records what
each would have returned net of costs:

    rule          the signal's own exit (what actually happened)
    tp_1R         take profit at one R (one stop distance in favour)
    tp_2R         take profit at two R
    breakeven_1R  after +1R, move the stop to entry
    trail_1R      after +1R, trail the stop one R behind the best price

Per signal, the running averages become an *exit policy*: once a variant
beats the rule by a clear margin over enough trades, the trader applies it
live for that signal; if it stops winning, the policy reverts. The same
step function drives both the replay and the live position, so what the
brain measured is exactly what it then does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from bigbrain.brain import Brain

VARIANTS = ("rule", "tp_1R", "tp_2R", "breakeven_1R", "trail_1R")
MIN_TRADES_FOR_POLICY = 20
MIN_EDGE = 0.001  # a variant must beat the rule by 0.1% average net return to be adopted


@dataclass
class ExitState:
    """Mutable exit bookkeeping for one position under one variant."""

    stop: float
    best: float  # best price seen in the trade's favour
    armed: bool = False  # +1R reached
    target: float | None = None


def init_state(variant: str, entry: float, side: int, risk_pct: float) -> ExitState:
    st = ExitState(stop=entry * (1 - side * risk_pct), best=entry)
    if variant == "tp_1R":
        st.target = entry * (1 + side * risk_pct)
    elif variant == "tp_2R":
        st.target = entry * (1 + side * 2 * risk_pct)
    return st


def step(variant: str, st: ExitState, entry: float, side: int, risk_pct: float, high: float, low: float) -> tuple[str | None, float | None]:
    """Advance one bar. Returns (exit reason, exit price) if the variant exits on this bar.

    Stops are checked before targets, the conservative assumption when both are touched in one bar."""
    stop_hit = (low <= st.stop) if side == 1 else (high >= st.stop)
    if stop_hit:
        return ("stop" if not st.armed else "trail"), st.stop
    if st.target is not None and ((high >= st.target) if side == 1 else (low <= st.target)):
        return "target", st.target
    favourable = high if side == 1 else low
    if (side == 1 and favourable > st.best) or (side == -1 and favourable < st.best):
        st.best = favourable
    one_r = entry * (1 + side * risk_pct)
    reached_1r = (st.best >= one_r) if side == 1 else (st.best <= one_r)
    if reached_1r and variant in ("breakeven_1R", "trail_1R"):
        st.armed = True
        if variant == "breakeven_1R":
            st.stop = max(st.stop, entry) if side == 1 else min(st.stop, entry)
        else:
            trail = st.best * (1 - side * risk_pct)
            st.stop = max(st.stop, trail) if side == 1 else min(st.stop, trail)
    return None, None


def simulate(path: list, entry: float, side: int, risk_pct: float, actual_exit: float, cost_ratio: float) -> dict[str, float]:
    """Net return each variant would have produced on this trade's path.

    ``path`` is [(high, low, close), ...] for every bar after entry; ``actual_exit`` is
    the price the rule exited at (used when a variant never triggers); ``cost_ratio``
    is fees plus slippage as a fraction of notional, charged to every variant alike."""
    out: dict[str, float] = {}
    for variant in VARIANTS:
        if variant == "rule":
            price = actual_exit
        else:
            st = init_state(variant, entry, side, risk_pct)
            price = actual_exit
            for high, low, _close in path:
                reason, px = step(variant, st, entry, side, risk_pct, high, low)
                if reason:
                    price = px
                    break
        out[variant] = side * (price / entry - 1) - cost_ratio
    return out


# ------------------------------------------------------------------ policy
def record(brain: Brain, book: str, signal: str, results: dict[str, float]) -> None:
    for variant, ret in results.items():
        brain.db.execute(
            "INSERT INTO exit_stats (book, signal, variant, n, sum_ret) VALUES (?, ?, ?, 1, ?)"
            " ON CONFLICT(book, signal, variant) DO UPDATE SET n = n + 1, sum_ret = sum_ret + excluded.sum_ret",
            (book, signal, variant, ret),
        )


def stats(brain: Brain, book: str, signal: str) -> dict[str, tuple[int, float]]:
    rows = brain.db.execute("SELECT variant, n, sum_ret FROM exit_stats WHERE book = ? AND signal = ?", (book, signal)).fetchall()
    return {r["variant"]: (r["n"], r["sum_ret"] / r["n"] if r["n"] else 0.0) for r in rows}


def policy_key(book: str) -> str:
    return f"exit_policy:{book}"


def current_policy(brain: Brain, book: str) -> dict[str, str]:
    return brain.get_state(policy_key(book), {}) or {}


def update_policy(brain: Brain, book: str, signal: str) -> tuple[str, str] | None:
    """Re-evaluate the exit policy for a signal. Returns (old, new) if it changed."""
    st = stats(brain, book, signal)
    if "rule" not in st or st["rule"][0] < MIN_TRADES_FOR_POLICY:
        return None
    rule_avg = st["rule"][1]
    best_variant, best_avg = "rule", rule_avg
    for variant, (n, avg) in st.items():
        if n >= MIN_TRADES_FOR_POLICY and avg > best_avg + MIN_EDGE:
            best_variant, best_avg = variant, avg
    policy = current_policy(brain, book)
    old = policy.get(signal, "rule")
    if best_variant == old:
        return None
    policy[signal] = best_variant
    brain.set_state(policy_key(book), policy)
    title = f"Exit policy: {signal.replace('_', ' ')} ({book})"
    brain.forget(title=title)
    ranking = ", ".join(f"{v} {avg:+.2%} over {n} trades" for v, (n, avg) in sorted(st.items(), key=lambda kv: -kv[1][1]))
    text = (
        f"From its own closed trades on signal '{signal.replace('_', ' ')}', the brain compared five exit styles on every trade's actual path: {ranking}. "
        f"It now exits {signal.replace('_', ' ')} positions with '{best_variant}' instead of '{old}' because it beat the alternative by at least {MIN_EDGE:.1%} net per trade. "
        "This is measured on the brain's own out-of-sample trades and is re-evaluated as more close; a trailing or profit-taking exit that stops winning is dropped again."
    )
    brain.learn("lesson", title, text, source=f"trade:{book}", extra_concepts=["paper trading", "walk-forward", "stop loss", "take profit"])
    brain.db.commit()
    return old, best_variant


def describe(brain: Brain, book: str) -> list[dict]:
    policy = current_policy(brain, book)
    signals = {r["signal"] for r in brain.db.execute("SELECT DISTINCT signal FROM exit_stats WHERE book = ?", (book,))}
    out = []
    for signal in sorted(signals):
        st = stats(brain, book, signal)
        rule_n, rule_avg = st.get("rule", (0, 0.0))
        best = max(st.items(), key=lambda kv: kv[1][1]) if st else ("rule", (0, 0.0))
        out.append({"signal": signal, "policy": policy.get(signal, "rule"), "trades": rule_n, "rule_avg": rule_avg, "best": best[0], "best_avg": best[1][1], "stats": st})
    return out
