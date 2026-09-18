"""Post-mortems: the brain studies every closed trade and updates its beliefs.

A closed trade carries the context it was entered in (regime, volatility
bucket, indicator readings), what happened while it was open (best and
worst excursion), and how it ended (exit reason, gross and net return).
The *lenses* below turn that into findings a trader would write in a
journal: "fought the trend", "stop inside the noise", "gave back an open
gain". Findings update the belief table that the trader consults before
its next entry, so a mistake made in a context is less likely to be
repeated in that context, and a success there is more likely to be
repeated. Instructive trades also become post-mortem cells the brain can
recall when asked why it lost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from bigbrain.brain import Brain

PRIOR_WINS = 2.0  # pseudo-counts: the textbook prior before any evidence
PRIOR_LOSSES = 2.0
MIN_SAMPLES = 8  # below this the trader is still exploring the (signal, context) cell
NOTABLE_RETURN = 0.01  # trades beyond +-1% net, or stopped out, get their own post-mortem cell


@dataclass
class Trade:
    book: str
    symbol: str
    signal: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str  # stop | signal | time | opposite
    qty: float
    notional: float
    gross_ret: float
    net_ret: float
    pnl: float
    fees: float
    slippage: float
    bars_held: int
    mfe: float  # best unrealized return while open
    mae: float  # worst unrealized return while open
    context: dict = field(default_factory=dict)
    explore: bool = False


# ------------------------------------------------------------------- lenses
def lenses(t: Trade) -> list[tuple[str, str]]:
    """Return (tag, sentence) findings for a trade. Applies to wins and losses alike."""
    ctx, out = t.context, []
    won = t.net_ret > 0
    r = ctx.get("risk_pct") or 0.0  # stop distance as a fraction of price; one R
    mean_rev = t.signal in ("rsi_oversold", "below_lower_band")
    trend_sig = t.signal in ("above_upper_band", "golden_cross", "macd_bullish")
    regime = ctx.get("regime", "unknown")

    if not won:
        if mean_rev and regime == "downtrend":
            out.append(("fought_the_trend", "entered a mean reversion long while the 50-period average was below the 200-period: buying dips in a downtrend, where stretched moves keep stretching."))
        if trend_sig and regime == "downtrend":
            out.append(("breakout_against_regime", "took a bullish breakout signal inside a downtrend; breakouts against the higher-timeframe trend fail more often than they follow through."))
        if t.exit_reason == "stop" and ctx.get("vol_bucket") == "high":
            out.append(("stop_inside_noise", f"stopped out in a high-volatility regime: the stop sat {r:.2%} away while typical bars moved {ctx.get('atr_pct', 0):.2%}, so noise alone could reach it."))
        if r and t.mfe >= r and t.exit_reason != "stop":
            out.append(("gave_back_open_gain", f"the trade was up {t.mfe:.2%} (more than one R) before closing at {t.net_ret:+.2%}: no partial exit or trailing stop protected the gain."))
        if t.bars_held <= 2 and t.exit_reason in ("stop", "opposite"):
            out.append(("whipsaw", "reversed within two bars of entry: the signal fired on a spike that immediately unwound."))
        if t.exit_reason == "time" and abs(t.mae) < (r or 0.005) * 0.5 and t.mfe < (r or 0.005) * 0.5:
            out.append(("thesis_never_developed", "price went nowhere for the whole holding period; the setup produced neither the expected move nor a clear failure, which suggests the signal carried no information here."))
        if t.gross_ret > 0 >= t.net_ret:
            out.append(("costs_ate_the_edge", f"gross return was {t.gross_ret:+.2%} but fees and slippage ({(t.fees + t.slippage) / t.notional:.2%} of notional) turned it into a loss: the move was too small for the cost of trading it."))
        if mean_rev and ctx.get("rsi") is not None and ctx["rsi"] > 26 and t.exit_reason == "stop":
            out.append(("shallow_oversold", f"RSI was {ctx['rsi']:.0f} at entry, barely oversold; deeper readings mark exhaustion more reliably."))
        if not out:
            out.append(("unexplained_loss", "no single lens explains this loss; it is within the normal variance of a positive-expectancy setup, if the setup has one."))
    else:
        if mean_rev and regime == "uptrend":
            out.append(("dip_in_uptrend", "bought an oversold dip inside an uptrend, where mean reversion has the trend on its side."))
        if trend_sig and regime == "uptrend":
            out.append(("trend_aligned", "the breakout followed the higher-timeframe trend, the setting where continuation is most likely."))
        if r and t.mfe > 2 * r and t.net_ret > 1.5 * r:
            out.append(("rode_the_move", f"held through a {t.mfe:.2%} favourable excursion and kept most of it; the exit rule let the winner run."))
        if t.bars_held <= 3:
            out.append(("fast_resolution", "the expected move arrived within three bars; the signal marked a genuine turning point."))
        if r and abs(t.mae) > 0.8 * r:
            out.append(("near_miss", f"the trade came within {abs(t.mae):.2%} of the stop before working: the outcome depended on stop placement more than on the signal."))
        if not out:
            out.append(("clean_win", "the trade worked as the signal predicted, with no notable stress."))
    return out


# --------------------------------------------------------------- beliefs
def update_belief(brain: Brain, t: Trade) -> None:
    ctx = t.context
    won = t.net_ret > 0
    brain.db.execute(
        "INSERT INTO beliefs (book, signal, regime, vol_bucket, wins, losses, sum_ret, sum_win, sum_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(book, signal, regime, vol_bucket) DO UPDATE SET wins = wins + excluded.wins, losses = losses + excluded.losses,"
        " sum_ret = sum_ret + excluded.sum_ret, sum_win = sum_win + excluded.sum_win, sum_loss = sum_loss + excluded.sum_loss",
        (t.book, t.signal, ctx.get("regime", "unknown"), ctx.get("vol_bucket", "mid"), int(won), int(not won), t.net_ret, t.net_ret if won else 0.0, t.net_ret if not won else 0.0),
    )


def belief(brain: Brain, book: str, signal: str, regime: str, vol_bucket: str) -> dict:
    """What the brain believes about a signal in a context: win probability, expectancy, and how much evidence it has.

    Falls back from the exact context to the signal across all contexts, then to the prior."""
    exact = brain.db.execute(
        "SELECT wins, losses, sum_ret, sum_win, sum_loss FROM beliefs WHERE book = ? AND signal = ? AND regime = ? AND vol_bucket = ?",
        (book, signal, regime, vol_bucket),
    ).fetchone()
    broad = brain.db.execute(
        "SELECT SUM(wins) AS wins, SUM(losses) AS losses, SUM(sum_ret) AS sum_ret, SUM(sum_win) AS sum_win, SUM(sum_loss) AS sum_loss FROM beliefs WHERE book = ? AND signal = ?",
        (book, signal),
    ).fetchone()
    ew, el, er = (exact["wins"], exact["losses"], exact["sum_ret"]) if exact else (0, 0, 0.0)
    bw, bl, br = (broad["wins"] or 0, broad["losses"] or 0, broad["sum_ret"] or 0.0)
    n_exact, n_broad = ew + el, bw + bl
    # Blend: the exact context dominates once it has samples; the broad record fills in before that.
    w_exact = min(1.0, n_exact / MIN_SAMPLES)
    wins = PRIOR_WINS + w_exact * ew + (1 - w_exact) * (bw * min(1.0, n_broad / MIN_SAMPLES))
    losses = PRIOR_LOSSES + w_exact * el + (1 - w_exact) * (bl * min(1.0, n_broad / MIN_SAMPLES))
    p_win = wins / (wins + losses)
    n_eff = n_exact if n_exact >= MIN_SAMPLES else max(n_exact, min(n_broad, MIN_SAMPLES - 1))
    expectancy = (er / n_exact) if n_exact >= MIN_SAMPLES else (br / n_broad if n_broad else 0.0)
    return {"p_win": p_win, "expectancy": expectancy, "samples": n_eff, "exact_samples": n_exact, "broad_samples": n_broad}


# ---------------------------------------------------------------- records
def record(brain: Brain, t: Trade, findings: list[tuple[str, str]]) -> None:
    brain.db.execute(
        "INSERT INTO trades (book, symbol, signal, entry_time, entry_price, exit_time, exit_price, exit_reason, qty, notional, gross_ret, net_ret, pnl,"
        " fees, slippage, bars_held, mfe, mae, context, findings, explore) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (t.book, t.symbol, t.signal, t.entry_time, t.entry_price, t.exit_time, t.exit_price, t.exit_reason, t.qty, t.notional, t.gross_ret, t.net_ret,
         t.pnl, t.fees, t.slippage, t.bars_held, t.mfe, t.mae, json.dumps(t.context), json.dumps([tag for tag, _ in findings]), int(t.explore)),
    )


def narrative(t: Trade, findings: list[tuple[str, str]]) -> str:
    ctx = t.context
    verdict = "profit" if t.net_ret > 0 else "loss"
    head = (
        f"{t.symbol} {t.signal.replace('_', ' ')} entered {t.entry_time} UTC at {t.entry_price:.6g} and closed {t.exit_time} at {t.exit_price:.6g} "
        f"by {t.exit_reason} after {t.bars_held} bars: a {verdict} of {t.net_ret:+.2%} net ({t.gross_ret:+.2%} gross, {t.pnl:+.2f} USDT on {t.notional:.0f} notional). "
        f"Context at entry: {ctx.get('regime', 'unknown')} regime, {ctx.get('vol_bucket', 'mid')} volatility ({ctx.get('vol20', 0):.0%} annualized), "
        f"RSI {ctx.get('rsi', 0):.0f}, stop {ctx.get('risk_pct', 0):.2%} away. While open the trade reached {t.mfe:+.2%} at best and {t.mae:+.2%} at worst."
    )
    why = " ".join(f"Finding ({tag.replace('_', ' ')}): {text}" for tag, text in findings)
    lesson = "Lesson: " + ("avoid this signal in this context, or demand a deeper reading and a wider stop." if t.net_ret <= 0 else "this signal in this context is worth taking again; keep the exit rule that captured it.")
    return f"{head} {why} {lesson}"


def learn_postmortem(brain: Brain, t: Trade, findings: list[tuple[str, str]]) -> str | None:
    """Write a post-mortem cell when the trade is instructive. Returns the cell title, or None."""
    notable = abs(t.net_ret) >= NOTABLE_RETURN or t.exit_reason == "stop" or any(tag != "clean_win" and tag != "unexplained_loss" for tag, _ in findings)
    if not notable:
        return None
    title = f"{t.symbol}: {'win' if t.net_ret > 0 else 'loss'} on {t.signal.replace('_', ' ')} {t.entry_time}"
    cell, _ = brain.learn(
        "postmortem", title, narrative(t, findings), source=f"trade:{t.book}",
        extra_concepts=[t.symbol.lower(), "paper trading", "post-mortem", t.context.get("regime", "unknown")] + [tag.replace("_", " ") for tag, _ in findings],
    )
    return cell.title


def learn_summary(brain: Brain, book: str, signal: str) -> None:
    """Rewrite the brain's summary of what it has learned about a signal, across contexts."""
    rows = brain.db.execute(
        "SELECT regime, vol_bucket, wins, losses, sum_ret FROM beliefs WHERE book = ? AND signal = ? ORDER BY wins + losses DESC", (book, signal)
    ).fetchall()
    if not rows or sum(r["wins"] + r["losses"] for r in rows) < MIN_SAMPLES:
        return
    tags = brain.db.execute("SELECT findings FROM trades WHERE book = ? AND signal = ?", (book, signal)).fetchall()
    counts: dict[str, int] = {}
    for r in tags:
        for tag in json.loads(r["findings"]):
            counts[tag] = counts.get(tag, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    parts = []
    for r in rows:
        n = r["wins"] + r["losses"]
        parts.append(f"in a {r['regime']} with {r['vol_bucket']} volatility it won {r['wins']} of {n} ({r['wins'] / n:.0%}) for an average {r['sum_ret'] / n:+.2%} net")
    text = (
        f"What the brain has learned about '{signal.replace('_', ' ')}' from its own paper trades ({book}): " + "; ".join(parts) + ". "
        + ("The most common findings were " + ", ".join(f"{tag.replace('_', ' ')} ({n})" for tag, n in top) + ". " if top else "")
        + "Contexts with a negative average are now avoided unless the brain is deliberately re-testing them; contexts with a positive average get full size. "
        "This is live, out-of-sample evidence and it keeps updating."
    )
    title = f"Learned: {signal.replace('_', ' ')} ({book})"
    brain.forget(title=title)
    brain.learn("lesson", title, text, source=f"trade:{book}", extra_concepts=["paper trading", "walk-forward", "post-mortem"])
