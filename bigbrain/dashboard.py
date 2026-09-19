"""A live terminal dashboard for a trading book.

Reads the trader's state from the brain every few seconds, fetches live
prices for open positions in one request, and redraws: equity and drawdown,
an equity sparkline, open positions with live unrealized P&L, what the
brain currently believes, the latest closed trades with their findings,
and the trade feed. Run it in a second terminal next to ``bigbrain trade``.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone

from bigbrain.brain import Brain
from bigbrain.ingest.market import FUTURES_HOST, BINANCE_HOSTS
from bigbrain.net import http_get

BLOCKS = "▁▂▃▄▅▆▇█"
CLEAR = "\x1b[2J\x1b[H"
BOLD, DIM, GREEN, RED, YELLOW, CYAN, RESET = "\x1b[1m", "\x1b[2m", "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[36m", "\x1b[0m"


def sparkline(values: list[float], width: int) -> str:
    if len(values) < 2:
        return ""
    step = max(1, len(values) // width)
    pts = values[-width * step :: step][-width:]
    lo, hi = min(pts), max(pts)
    if hi == lo:
        return BLOCKS[3] * len(pts)
    return "".join(BLOCKS[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in pts)


def live_prices(market: str) -> dict[str, float]:
    url = f"{FUTURES_HOST}/fapi/v1/ticker/price" if market == "perps" else f"{BINANCE_HOSTS[0]}/api/v3/ticker/price"
    rows = json.loads(http_get(url, headers={"Accept": "application/json"}).decode("utf-8"))
    return {r["symbol"]: float(r["price"]) for r in rows}


def colour(x: float, text: str) -> str:
    return f"{GREEN if x > 0 else RED if x < 0 else ''}{text}{RESET}"


def render(brain: Brain, book: str, prices: dict[str, float] | None = None, width: int | None = None, height: int | None = None) -> str:
    """Build the dashboard text. Pure apart from reading the brain, so it is testable."""
    width = width or shutil.get_terminal_size((120, 50)).columns
    height = height or shutil.get_terminal_size((120, 50)).lines
    state = brain.get_state(f"trader:{book}")
    if not state:
        return f"No trading book '{book}' yet. Run `bigbrain trade` first."
    prices = prices or {}
    market = state.get("market", "perps")
    positions = [dict(p) for p in state["positions"].values()]
    for p in positions:
        p["close_mark"] = p["mark"]
        if p["symbol"] in prices:
            p["mark"] = prices[p["symbol"]]
    basis = "live prices" if prices else "last candle close (no live prices)"
    cash = state["cash"]
    margin = sum(p.get("margin") or p["notional"] for p in positions)
    # two bases: the trader's own marks (last closed candle) and live ticker prices
    unreal_close = sum(p.get("side", 1) * p["qty"] * ((p["close_mark"] or p["entry_price"]) - p["entry_price"]) for p in positions)
    unreal = sum(p.get("side", 1) * p["qty"] * ((p["mark"] or p["entry_price"]) - p["entry_price"]) for p in positions)
    equity_close = cash + margin + unreal_close
    equity = cash + margin + unreal
    last_candle = max(state.get("last_bar", {}).values(), default="")
    start = state["start"]
    gross = sum(p["qty"] * (p["mark"] or p["entry_price"]) for p in positions)
    open_risk = sum(p["notional"] * (p["context"].get("risk_pct") or 0) for p in positions)
    peak = max(state.get("peak", start), equity)
    dd = equity / peak - 1 if peak else 0.0
    closed = brain.db.execute("SELECT COUNT(*) AS n, SUM(pnl) AS pnl, SUM(CASE WHEN net_ret > 0 THEN 1 ELSE 0 END) AS wins, SUM(fees) AS fees, SUM(slippage) AS slip, SUM(funding) AS funding FROM trades WHERE book = ?", (book,)).fetchone()
    n_closed = closed["n"] or 0
    realized, fees, slip, funding = closed["pnl"] or 0.0, closed["fees"] or 0.0, closed["slip"] or 0.0, closed["funding"] or 0.0
    win_rate = (closed["wins"] or 0) / n_closed if n_closed else 0.0
    lev = gross / equity if equity else 0.0
    risk_share = open_risk / equity if equity else 0.0
    max_dd = state.get("max_drawdown", 0.0)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    from bigbrain.watch import INTERVAL_SECONDS

    interval = state.get("interval", "15m")
    last_tick = state.get("last_tick", "")
    status = f"{YELLOW}status unknown{RESET}"
    if last_tick:
        age = (datetime.now(timezone.utc) - datetime.strptime(last_tick, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)).total_seconds()
        if age <= INTERVAL_SECONDS.get(interval, 900) * 1.5:
            status = f"{GREEN}trader running{RESET} (last tick {age / 60:.0f} min ago)"
        else:
            status = f"{RED}TRADER STOPPED{RESET} (last tick {age / 3600:.1f} h ago: positions are frozen and stops cannot fire until `bigbrain trade` runs again)"

    lines = []
    lines.append(f"{BOLD}{CYAN}BIG BRAIN TIME{RESET}  book {BOLD}{book}{RESET} on Binance {market} {interval}   {DIM}{now}{RESET}   {status}")
    ret_text = colour(equity - start, f"{equity / start - 1:+.2%} ({equity - start:+.2f})")
    unreal_text = colour(unreal, f"{unreal:+.2f}")
    dd_colour = RED if dd < -0.02 else ""
    realized_text = colour(realized, f"{realized:+.2f}")
    close_text = colour(unreal_close, f"{unreal_close:+.2f}")
    lines.append(f"equity {BOLD}{equity:,.2f}{RESET} USDT at {basis}  {ret_text}   drawdown {dd_colour}{dd:.2%}{RESET} (max {max_dd:.2%})")
    lines.append(f"at last candle close ({last_candle or 'n/a'} UTC, the trader's basis): equity {equity_close:,.2f}  unrealized {close_text}   |   live: unrealized {unreal_text}   cash {cash:,.2f}   margin {margin:,.2f}")
    lines.append(f"open {len(positions)}   gross exposure {gross:,.0f} ({lev:.2f}x)   risk at stops {open_risk:,.2f} ({risk_share:.1%})   "
                 f"closed {n_closed}   realized {realized_text}   win {win_rate:.0%}   fees {fees:.2f}  slip {slip:.2f}  funding {funding:+.2f}")
    curve = [v for _, v in state.get("curve", [])]
    if curve:
        lines.append(f"{DIM}equity{RESET} {sparkline(curve + [equity], min(width - 12, 100))}")
    lines.append("")

    # open positions
    lines.append(f"{BOLD}OPEN POSITIONS{RESET}  {DIM}side symbol signal entry -> live mark, unrealized at live, bars held, stop{RESET}")
    positions.sort(key=lambda p: -abs(p.get("side", 1) * p["qty"] * ((p["mark"] or p["entry_price"]) - p["entry_price"])))
    max_pos = max(5, (height - 22) // 2)
    for p in positions[:max_pos]:
        side = "long " if p.get("side", 1) == 1 else "short"
        mark = p["mark"] or p["entry_price"]
        u = p.get("side", 1) * (mark / p["entry_price"] - 1)
        upnl = p.get("side", 1) * p["qty"] * (mark - p["entry_price"])
        tag = f"{DIM}explore{RESET}" if p.get("explore") else ""
        pnl_text = colour(u, f"{u:+6.2%} {upnl:+7.2f}")
        lines.append(f"  {side} {p['symbol']:12} {p['signal']:18} {p['entry_price']:<11.6g} {mark:<11.6g} {pnl_text}  {p['bars_held']:3}  {p['stop']:<10.6g} {tag}")
    if len(positions) > max_pos:
        lines.append(f"  {DIM}... and {len(positions) - max_pos} more{RESET}")
    if not positions:
        lines.append(f"  {DIM}none{RESET}")
    lines.append("")

    # beliefs
    beliefs = brain.db.execute("SELECT signal, regime, vol_bucket, wins, losses, sum_ret FROM beliefs WHERE book = ? ORDER BY wins + losses DESC LIMIT 6", (book,)).fetchall()
    lines.append(f"{BOLD}WHAT THE BRAIN BELIEVES{RESET}  {DIM}signal / regime / vol: n trades over d days, win rate, avg net; the verdict the trader acts on{RESET}")
    if beliefs:
        from bigbrain import postmortem as pm

        shades = {"full": f"{GREEN}full size{RESET}", "half": f"{GREEN}half size{RESET}", "avoid": f"{RED}avoid{RESET}", "explore": f"{YELLOW}exploring{RESET}"}
        for b in beliefs:
            n = b["wins"] + b["losses"]
            avg = b["sum_ret"] / n
            belief = pm.belief(brain, book, b["signal"], b["regime"], b["vol_bucket"])
            avg_text = colour(avg, f"{avg:+.2%}")
            wr = b["wins"] / n
            lines.append(f"  {b['signal']:18} {b['regime']:9} {b['vol_bucket']:4}  n {n:3} / {belief['blocks']:2}d  win {wr:4.0%}  avg {avg_text}  {shades[pm.verdict(belief)]}")
    else:
        lines.append(f"  {DIM}no closed trades yet{RESET}")
    lines.append("")

    # exit policies
    from bigbrain import exits as _exits

    pol = [p for p in _exits.describe(brain, book) if p["trades"] >= 5]
    if pol:
        lines.append(f"{BOLD}EXIT LEARNING{RESET}  {DIM}per signal: live exit style, and the best measured alternative{RESET}")
        for p in pol[:4]:
            best_text = colour(p["best_avg"], f"{p['best']} {p['best_avg']:+.2%}")
            rule_text = colour(p["rule_avg"], f"{p['rule_avg']:+.2%}")
            lines.append(f"  {p['signal']:18} exits by {p['policy']:13} rule avg {rule_text}  best {best_text}  ({p['trades']} trades)")
        lines.append("")

    # recent closed trades
    lines.append(f"{BOLD}RECENT CLOSED TRADES{RESET}")
    rows = brain.db.execute("SELECT symbol, signal, side, exit_time, exit_reason, net_ret, pnl, findings FROM trades WHERE book = ? ORDER BY id DESC LIMIT 6", (book,)).fetchall()
    for r in rows:
        side = "long " if r["side"] == 1 else "short"
        pnl_text = colour(r["pnl"], f"{r['net_ret']:+6.2%} {r['pnl']:+7.2f}")
        tags = ", ".join(json.loads(r["findings"]))
        when = r["exit_time"][5:]
        lines.append(f"  {when} {side} {r['symbol']:12} {r['signal']:18} {r['exit_reason']:6} {pnl_text}  {DIM}{tags}{RESET}")
    if not rows:
        lines.append(f"  {DIM}none yet{RESET}")
    lines.append("")

    # feed
    remaining = max(3, height - len(lines) - 2)
    lines.append(f"{BOLD}TRADE FEED{RESET}")
    for line in state.get("log", [])[-remaining:]:
        lines.append(f"  {line[: width - 3]}")
    return "\n".join(line[: width + 40] for line in lines)  # +40 allows for colour codes


def run(brain: Brain, book: str = "main", refresh: float = 10.0, once: bool = False, out=print, sleep=time.sleep) -> None:
    import sqlite3

    while True:
        try:
            state = brain.get_state(f"trader:{book}") or {}
            prices: dict[str, float] = {}
            if state.get("positions"):
                try:
                    prices = live_prices(state.get("market", "perps"))
                except Exception:
                    prices = {}
            out(CLEAR + render(brain, book, prices))
        except sqlite3.OperationalError as exc:
            out(f"{CLEAR}{YELLOW}database busy ({exc}); retrying in {refresh:.0f}s{RESET}")
        if once:
            return
        sleep(refresh)
