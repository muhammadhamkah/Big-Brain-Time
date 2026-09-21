"""Command line interface: ``bigbrain <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bigbrain import __version__
from bigbrain.brain import Brain

DEFAULT_DB = os.environ.get("BIGBRAIN_DB", ".brain/brain.db")


def open_brain(path: str) -> Brain:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Brain(path)


def cmd_seed(args: argparse.Namespace) -> int:
    from bigbrain.ingest.textbook import seed

    brain = open_brain(args.db)
    n = seed(brain, packs=not args.no_packs)
    print(f"Seeded {n} new concept cells. Brain now has {brain.count_cells()} cells and {brain.count_synapses()} synapses.")
    return 0


def cmd_learn_text(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    content = args.content if args.content else sys.stdin.read()
    cell, synapses = brain.learn(args.kind, args.title, content.strip(), source=args.source or "")
    print(f"Learned '{cell.title}' [{cell.kind}] concepts={cell.concepts}")
    _print_links(brain, synapses)
    return 0


def cmd_learn_file(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    for path in args.paths:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        cell, synapses = brain.learn(args.kind, p.stem.replace("_", " ").replace("-", " "), text.strip(), source=str(p))
        print(f"Learned '{cell.title}' from {p} (+{len(synapses)} links)")
    return 0


def cmd_learn_papers(args: argparse.Namespace) -> int:
    from bigbrain.ingest.papers import fetch, learn_papers

    brain = open_brain(args.db)
    print(f"Fetching up to {args.max} papers from arXiv" + (f" about '{args.query}'" if args.query else "") + " ...")
    try:
        papers = fetch(args.query, max_results=args.max)
    except Exception as exc:  # network errors are the common failure here
        print(f"Could not reach arXiv: {exc}", file=sys.stderr)
        print("arXiv sometimes throttles with HTTP 406; wait a minute and try again.", file=sys.stderr)
        return 1
    before = brain.count_synapses()
    titles = learn_papers(brain, papers)
    for t in titles:
        print(f"  + {t}")
    print(f"Learned {len(titles)} papers, grew {brain.count_synapses() - before} synapses.")
    return 0


def cmd_learn_market(args: argparse.Namespace) -> int:
    from bigbrain.ingest.market import learn_bars, load_csv, synthetic
    from bigbrain.ingest.strategies import learn_backtests

    from bigbrain.ingest.market import fetch_binance, fetch_stooq

    brain = open_brain(args.db)
    if args.csv:
        if not Path(args.csv).is_file():
            print(f"No such file: {args.csv}. Point --csv at a CSV with date,open,high,low,close,volume columns.", file=sys.stderr)
            return 1
        try:
            bars = load_csv(args.csv)
        except (ValueError, KeyError) as exc:
            print(f"Could not read {args.csv}: {exc}", file=sys.stderr)
            return 1
        symbol = args.symbol or Path(args.csv).stem.upper()
        source = f"csv:{args.csv}"
    elif getattr(args, "source", None) == "binance":
        symbol = (args.symbol or "BTCUSDT").upper()
        print(f"Fetching {args.limit} {args.interval} candles of {symbol} from Binance ...")
        try:
            bars = fetch_binance(symbol, interval=args.interval, limit=args.limit)
        except Exception as exc:
            print(f"Could not fetch from Binance: {exc}", file=sys.stderr)
            return 1
        source = f"binance:{symbol}:{args.interval}"
    elif getattr(args, "source", None) == "stooq":
        if not args.symbol:
            print("Stooq needs --symbol, e.g. aapl.us, spy.us, ^spx, btc.v", file=sys.stderr)
            return 1
        print(f"Fetching daily history of {args.symbol} from Stooq ...")
        try:
            bars = fetch_stooq(args.symbol)
        except Exception as exc:
            print(f"Could not fetch from Stooq: {exc}", file=sys.stderr)
            return 1
        symbol = args.symbol.upper()
        source = f"stooq:{args.symbol}"
    else:
        symbol = args.symbol or "SYNTH"
        bars = synthetic(symbol, n=args.bars, seed=args.seed)
        source = "synthetic data"
    before = brain.count_synapses()
    titles = learn_bars(brain, symbol, bars, source=source)
    for t in titles:
        print(f"  + {t}")
    if not args.no_backtest:
        results = learn_backtests(brain, symbol, bars, source=f"backtest on {source}")
        for r in results:
            print(f"  + {r.strategy} on {symbol}: return {r.total_return:+.1%}, Sharpe {r.sharpe:.2f}, maxDD {r.max_drawdown:.1%}, trades {r.trades}")
    print(f"Learned from {len(bars)} bars of {symbol}, grew {brain.count_synapses() - before} synapses.")
    return 0


def cmd_learn_reddit(args: argparse.Namespace) -> int:
    from bigbrain.ingest.reddit import DEFAULT_SUBREDDITS, fetch_posts, learn_posts

    brain = open_brain(args.db)
    subs = tuple(args.sub) if args.sub else DEFAULT_SUBREDDITS
    total = 0
    for sub in subs:
        print(f"Reading r/{sub} (top of the {args.time}) ...")
        try:
            posts = fetch_posts(sub, time=args.time, limit=args.limit, min_score=args.min_score, with_comments=not args.no_comments)
        except Exception as exc:
            print(f"  could not read r/{sub}: {exc}", file=sys.stderr)
            continue
        titles = learn_posts(brain, posts)
        for t in titles:
            print(f"  + {t}")
        total += len(titles)
    print(f"Learned {total} discussions. Brain now has {brain.count_cells()} cells and {brain.count_synapses()} synapses.")
    return 0


def cmd_learn_github(args: argparse.Namespace) -> int:
    from bigbrain.ingest.github import DEFAULT_QUERIES, learn_repos, search

    brain = open_brain(args.db)
    queries = tuple(args.query) if args.query else DEFAULT_QUERIES
    if not os.environ.get("GITHUB_TOKEN"):
        print("(no GITHUB_TOKEN set: limited to 60 requests an hour)")
    total = 0
    for q in queries:
        print(f"Searching GitHub for {q!r} ...")
        try:
            repos = search(q, args.max)
            titles = learn_repos(brain, repos, with_readme=not args.no_readme)
        except Exception as exc:
            print(f"  GitHub error: {exc}", file=sys.stderr)
            continue
        for t in titles:
            print(f"  + {t}")
        total += len(titles)
    print(f"Learned {total} repositories. Brain now has {brain.count_cells()} cells and {brain.count_synapses()} synapses.")
    return 0


def cmd_learn_url(args: argparse.Namespace) -> int:
    from bigbrain.ingest.web import learn_url

    brain = open_brain(args.db)
    ok = 0
    for url in args.urls:
        try:
            title, links = learn_url(brain, url, kind=args.kind)
            print(f"  + {title} (+{links} links)")
            ok += 1
        except Exception as exc:
            print(f"  could not learn {url}: {exc}", file=sys.stderr)
    return 0 if ok else 1


def cmd_learn_feed(args: argparse.Namespace) -> int:
    from bigbrain.ingest.web import learn_feed
    from bigbrain.sources import DEFAULT_FEEDS

    brain = open_brain(args.db)
    feeds = args.urls or DEFAULT_FEEDS
    total = 0
    for url in feeds:
        print(f"Reading feed {url} ...")
        try:
            titles = learn_feed(brain, url, max_items=args.max, follow_links=args.full)
        except Exception as exc:
            print(f"  could not read {url}: {exc}", file=sys.stderr)
            continue
        for t in titles:
            print(f"  + {t}")
        total += len(titles)
    print(f"Learned {total} articles. Brain now has {brain.count_cells()} cells and {brain.count_synapses()} synapses.")
    return 0


def cmd_feed(args: argparse.Namespace) -> int:
    """Go online and learn from every source the brain knows."""
    from bigbrain.sources import run_all

    brain = open_brain(args.db)
    before = (brain.count_cells(), brain.count_synapses())
    report = run_all(brain, quick=args.quick, workers=args.workers, log=print)
    after = (brain.count_cells(), brain.count_synapses())
    print()
    for source, (n, err) in report.items():
        status = f"{n} learned" if err is None else f"failed: {err}"
        print(f"  {source:12} {status}")
    print(f"\nBrain grew from {before[0]} to {after[0]} cells and from {before[1]} to {after[1]} synapses.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from bigbrain import cortex

    brain = open_brain(args.db)
    use_claude = True if args.claude else False if args.offline else None
    text, recalls = cortex.answer(brain, args.question, k=args.k, use_claude=use_claude, model=args.model)
    print(text)
    if args.show_cells:
        print("\nRecalled cells:")
        for r in recalls:
            print(f"  {r.score:.2f}  {r.cell.title}  [{r.cell.kind}]")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    for r in brain.recall(args.query, k=args.k, reinforce=not args.no_reinforce):
        via = f" via {len(r.via)} link(s)" if r.via else ""
        print(f"{r.score:5.2f}  {r.cell.title}  [{r.cell.kind}]{via}")
        if args.verbose:
            print(f"       {r.cell.summary(200)}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    matches = brain.find(args.title)
    if not matches:
        print(f"No cell with a title containing '{args.title}'.")
        return 1
    cell = matches[0]
    print(f"{cell.title}  [{cell.kind}]  id={cell.id}  activations={cell.activations}")
    print(f"source: {cell.source or 'n/a'}")
    print(f"concepts: {', '.join(cell.concepts) or 'none'}")
    print()
    print(cell.content)
    print()
    print("Connected to:")
    for other, syn in brain.neighbors(cell.id, limit=args.limit):
        print(f"  {syn.weight:.2f}  {other.title}  [{other.kind}]  ({syn.reason})")
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    try:
        n = brain.forget(source=args.source, kind=args.kind, title=args.title)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Forgot {n} cells. Brain now has {brain.count_cells()} cells and {brain.count_synapses()} synapses.")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from bigbrain.watch import Watcher

    brain = open_brain(args.db)
    w = Watcher(brain, symbol=args.symbol, interval=args.interval, horizon=args.horizon, paper=not args.no_paper)
    if not args.once:
        extra = "" if args.no_paper else f" Paper trading {len(w.paper.strategies)} strategies with virtual accounts."
        print(f"Watching {w.symbol} on the {args.interval} chart; grading each signal {args.horizon} bars later.{extra} Ctrl-C to stop.")
    try:
        w.run(once=args.once)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    from bigbrain.paper import STARTING_EQUITY, PaperTrader

    brain = open_brain(args.db)
    rows = brain.db.execute("SELECT key FROM state WHERE key LIKE 'paper:%'").fetchall()
    if not rows:
        print("No paper trading yet. Run `bigbrain watch` first.")
        return 0
    for r in rows:
        _, symbol, interval = r["key"].split(":")
        if args.symbol and symbol != args.symbol.upper():
            continue
        pt = PaperTrader(brain, symbol, interval)
        live = brain.get_state(f"watch:{symbol}:{interval}", {})
        price = live.get("close")
        print(f"{symbol} {interval}  (virtual accounts start at {STARTING_EQUITY:,.0f}; last close {price:.2f})" if price else f"{symbol} {interval}")
        for s in pt.report(price):
            pos = f"long from {s['entry_price']:.2f}" if s["in_position"] else "flat"
            wr = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else "  -"
            print(f"  {s['strategy']:20} equity {s['equity']:>10,.2f}  {s['return']:+7.2%}  maxDD {s['max_drawdown']:6.1%}  trades {s['trades']:3}  win {wr:>4}  {pos}")
        if args.recent:
            print("  recent trades:")
            for t in brain.db.execute(
                "SELECT strategy, entry_time, entry_price, exit_time, exit_price, ret FROM paper_trades WHERE symbol = ? AND interval = ? ORDER BY entry_time DESC LIMIT ?",
                (symbol, interval, args.recent),
            ):
                status = f"closed {t['exit_time']} @ {t['exit_price']:.2f} ({t['ret']:+.2%})" if t["exit_time"] else "open"
                print(f"    {t['strategy']:20} in {t['entry_time']} @ {t['entry_price']:.2f}  {status}")
    return 0


def cmd_trade(args: argparse.Namespace) -> int:
    from bigbrain.trader import Trader

    brain = open_brain(args.db)
    t = Trader(brain, book=args.book, interval=args.interval, wallet=args.wallet, top=args.top, market=args.market, frozen=args.frozen)
    companions = ()
    if args.with_frozen:
        if args.frozen:
            print("--with-frozen runs a frozen companion beside an adaptive book; drop --frozen.", file=sys.stderr)
            return 2
        # the baseline: same market, interval, universe, starting capital, risk limits and inputs; no learning
        companions = (Trader(brain, book=f"{args.book}-frozen", interval=args.interval, wallet=t.wallet.start, top=args.top, market=t.market, frozen=True),)
    if args.reset:
        for b in (t, *companions):
            r = b.reset(wallet=args.wallet)
            print(f"Reset book '{b.book}': closed {r['positions_closed']} positions, wallet back to {r['wallet']:.0f} USDT (learning kept).")
    if not args.once:
        sides = "long and short, real funding rates charged" if t.market == "perps" else "long only"
        mode = "FROZEN baseline: fixed rules, fixed size, no learning" if t.frozen else "adaptive: beliefs, exit learning, post-mortems"
        print(f"Trading book '{args.book}' on Binance {t.market}: top {args.top} USDT pairs on {args.interval} candles, wallet {t.wallet.start:.0f} USDT, "
              f"VIP0 fees plus slippage, fills at the live bid/ask, {sides}. {mode}. Ctrl-C to stop.")
        if companions:
            print(f"Companion book '{companions[0].book}' runs frozen on the same candles, quotes and funding: the baseline the learning must beat.")
    try:
        t.run(once=args.once, companions=companions)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        for b in (t, *companions):
            b.release_lock()
        print(f"\nstopped. {len(t.wallet.positions)} positions stay open in the book; run `bigbrain trade` again to manage them.")
        sys.stdout.flush()
        os._exit(0)  # skip waiting on fetch threads still in flight; state was saved after every symbol
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    from bigbrain import backup

    db = Path(args.db)
    if not db.exists():
        print(f"no database at {db}", file=sys.stderr)
        return 1
    out_dir = Path(args.dir) if args.dir else db.parent / "backups"
    repo = Path(__file__).resolve().parent.parent
    backup.run_periodic(db, out_dir, repo, every_hours=args.every, keep=args.keep, do_push=args.push, once=not args.every)
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    from bigbrain import backup

    db = Path(args.db)
    out_dir = db.parent / "backups"
    repo = Path(__file__).resolve().parent.parent
    if args.from_github:
        try:
            snap = backup.fetch(repo, out_dir)
        except Exception as exc:
            print(f"could not fetch the backup branch: {exc}", file=sys.stderr)
            return 1
    elif args.file:
        snap = Path(args.file)
    else:
        snap = backup.latest(out_dir)
        if snap is None:
            print(f"no snapshots in {out_dir}", file=sys.stderr)
            return 1
    print(f"restoring {snap} -> {db}  (stop the trader first; the current file is kept as {db.name}.before-restore)")
    backup.restore(snap, db)
    print("done")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    before = brain.size_report()["bytes"]
    n = brain.prune(args.kind, args.days)
    brain.checkpoint()
    brain.db.execute("VACUUM")
    after = brain.size_report()["bytes"]
    print(f"forgot {n} {args.kind} cells older than {args.days} days; database {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    """Remove duplicate trade records (already done on open) and rebuild beliefs from the trade log."""
    from bigbrain.trader import Trader

    brain = open_brain(args.db)
    r = Trader(brain, book=args.book).rebuild_stats()
    print(f"book '{args.book}': beliefs and exit statistics rebuilt from {r['trades']} trades.")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    from bigbrain.trader import Trader

    brain = open_brain(args.db)
    if brain.get_state(f"trader:{args.book}") is None:
        print(f"No trading book '{args.book}' to reset.")
        return 0
    r = Trader(brain, book=args.book).reset(wallet=args.wallet, forget=args.forget)
    kept = f"history forgotten ({r['forgot_cells']} post-mortem cells removed)" if args.forget else "trades, beliefs and post-mortems kept"
    print(f"Reset book '{args.book}': closed {r['positions_closed']} positions, wallet back to {r['wallet']:.0f} USDT, {kept}.")
    print("If the trader is running, stop it (Ctrl-C) and start it again so it picks up the fresh book.")
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    from bigbrain.trader import Trader

    brain = open_brain(args.db)
    if brain.get_state(f"trader:{args.book}") is None:
        print(f"No trading book '{args.book}' yet. Run `bigbrain trade` first.")
        return 0
    t = Trader(brain, book=args.book)
    r = t.report()
    cfg = brain.get_state(f"strategy:{args.book}")
    if cfg:
        from bigbrain.strategy import StrategyTrader

        t = StrategyTrader(brain, book=args.book, rule=cfg["rule"], symbols=cfg["symbols"], grid=cfg.get("grid"), refit_days=cfg.get("refit_days", 30))
        r = t.report()
        bench = r["benchmark"]
        print(f"strategy {r['rule']} with {lab_params(r['params'])} (re-chosen {r['last_refit'] or 'never'}); control: holding the same {bench['symbols']} coins since {bench['since'] or '-'}: {bench['return']:+.2%}")
    mode = ", frozen baseline" if r["frozen"] else ""
    print(f"book {args.book} ({r['market']}{mode}): equity {r['equity']:.2f} USDT ({r['return']:+.2%} on {r['start']:.0f}), cash {r['cash']:.2f}, max drawdown {r['max_drawdown']:.1%}, closed trades {r['closed']}"
          + (f", {r['pending']} orders queued for the next open" if r["pending"] else "") + (f", {r['shadows']} shadows scoring exits" if r["shadows"] else ""))
    if r["open"]:
        print("open positions:")
        for p in r["open"]:
            fund = f"  funding {p['funding']:+.2f}" if p["funding"] else ""
            leaving = f"  exiting at next open ({p['pending_exit']})" if p.get("pending_exit") else ""
            print(f"  {p['side']:5} {p['symbol']:12} {p['signal']:18} in @ {p['entry']:.6g}  now {p['mark']:.6g} ({p['unrealized']:+.2%})  {p['bars']} bars  stop {p['stop']:.6g}{fund}{'  exploring' if p['explore'] else ''}{leaving}")
    if r["by_signal"]:
        print("closed trades by signal:")
        for b in r["by_signal"]:
            print(f"  {b['signal']:18} trades {b['trades']:4}  win {b['win_rate']:.0%}  avg {b['avg_ret']:+.2%}  pnl {b['pnl']:+.2f}  fees {b['fees']:.2f}  slippage {b['slippage']:.2f}  funding {b['funding']:+.2f}")
    if args.recent:
        print("recent closed trades:")
        for row in brain.db.execute(
            "SELECT symbol, signal, entry_time, exit_time, exit_reason, net_ret, pnl, findings FROM trades WHERE book = ? ORDER BY id DESC LIMIT ?", (args.book, args.recent)
        ):
            print(f"  {row['symbol']:12} {row['signal']:18} {row['entry_time']} -> {row['exit_time']} {row['exit_reason']:6} {row['net_ret']:+.2%} ({row['pnl']:+.2f})  {', '.join(json.loads(row['findings']))}")
    return 0


def lab_params(p: dict | None) -> str:
    from bigbrain.lab import describe_params

    return describe_params(p) if p else "default parameters"


def cmd_dashboard(args: argparse.Namespace) -> int:
    from bigbrain import dashboard

    brain = open_brain(args.db)
    try:
        dashboard.run(brain, book=args.book, refresh=args.refresh, once=args.once)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_beliefs(args: argparse.Namespace) -> int:
    from bigbrain.trader import Trader

    from bigbrain import postmortem as pm

    brain = open_brain(args.db)
    rows = Trader(brain, book=args.book).beliefs()
    if not rows:
        print("No beliefs yet: the brain has not closed a trade.")
        return 0
    print(f"full size needs {pm.FULL_SIZE_SAMPLES} trades over {pm.MIN_BLOCKS}+ separate days with the mean minus one standard error above zero (days, not trades, are the independent evidence)")
    print(f"{'signal':18} {'regime':10} {'vol':5} {'n':>4} {'days':>4} {'win':>5} {'avg':>8} {'se':>7}  verdict")
    for b in rows:
        se = f"{b['stderr']:7.2%}" if b["stderr"] != float("inf") else "      -"
        print(f"{b['signal']:18} {b['regime']:10} {b['vol']:5} {b['n']:4} {b['blocks']:4} {b['win_rate']:5.0%} {b['avg_ret']:+8.2%} {se}  {b['verdict']}")
    from bigbrain import exits

    pol = exits.describe(brain, args.book)
    if pol:
        print("\nexit policies (net average per trade under each exit style, measured on the brain's own trades):")
        for p in pol:
            ranking = "  ".join(f"{v} {avg:+.2%}" for v, (n, avg) in sorted(p["stats"].items(), key=lambda kv: -kv[1][1]))
            print(f"  {p['signal']:18} uses {p['policy']:13} ({p['trades']} trades)  {ranking}")
    log = brain.get_state(f"policy_log:{args.book}", []) or []
    if log:
        print("\npolicy changes (every adoption and reversal, with the evidence at the time):")
        for e in log[-12:]:
            print(f"  {e['at']}  {e['signal']:18} {e['from']} -> {e['to']}  rule {e['rule_avg']:+.2%} vs {e['best_avg']:+.2%} over {e['trades']} trades")
    return 0


def cmd_strategy(args: argparse.Namespace) -> int:
    from bigbrain import lab
    from bigbrain.strategy import StrategyTrader

    brain = open_brain(args.db)
    if args.rule not in lab.RULES:
        print(f"unknown rule '{args.rule}'; rules: {', '.join(lab.RULES)}", file=sys.stderr)
        return 2
    try:
        grid = lab.parse_grid(args.grid or "", lab.RULES[args.rule]["defaults"])
        symbols = lab.resolve_symbols(args.symbols, args.market)
        t = StrategyTrader(brain, book=args.book, rule=args.rule, symbols=symbols, interval=args.interval, wallet=args.wallet, market=args.market,
                           grid=grid, target_vol=args.target_vol, refit_days=args.refit_days)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.reset:
        r = t.reset(wallet=args.wallet)
        t.hold_base, t.hold_start, t.params, t.last_refit = {}, "", None, ""
        t._save_config()
        print(f"Reset book '{args.book}': closed {r['positions_closed']} positions, wallet back to {r['wallet']:.0f} USDT; the control restarts with it.")
    if args.rebase_control:
        t.hold_base, t.hold_start = {}, ""  # the next tick starts the control again at the prices the book can trade at now
        t._save_config()
        print("the buy-and-hold control restarts at the next tick, at current quotes; positions are untouched.")
    if not args.once:
        print(f"Strategy book '{args.book}': {args.rule} on {len(symbols)} symbols ({', '.join(symbols[:6])}{', ...' if len(symbols) > 6 else ''}), {args.interval} candles, "
              f"Binance {args.market}, wallet {t.wallet.start:.0f} USDT. Parameters re-chosen every {args.refit_days} days from {len(lab.grid_points(grid))} settings; "
              f"a buy-and-hold control of the same coins runs beside it. No beliefs, no exit learning: the rule decides. Ctrl-C to stop.")
    try:
        t.run(once=args.once)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        t.release_lock()
        print(f"\nstopped. {len(t.wallet.positions)} positions stay open in the book; run the same command again to manage them.")
        sys.stdout.flush()
        os._exit(0)
    return 0


def cmd_lab(args: argparse.Namespace) -> int:
    from bigbrain import lab

    if args.rule not in lab.RULES:
        print(f"unknown rule '{args.rule}'; rules:", file=sys.stderr)
        for k, v in lab.RULES.items():
            print(f"  {k:14} {v['doc']}" + (f" [{v['hint']}]" if v["hint"] else ""), file=sys.stderr)
        return 2
    spec = lab.RULES[args.rule]
    try:
        grid = lab.parse_grid(args.grid or "", spec["defaults"])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    market = args.market or ("perps" if "funding" in spec["needs"] else "spot")
    if "funding" in spec["needs"] and market != "perps":
        print(f"the {args.rule} rule needs funding rates, which only perps have; add --market perps", file=sys.stderr)
        return 2
    fee = args.fee if args.fee is not None else (0.0005 if market == "perps" else 0.001)
    costs = lab.Costs(fee=fee, spread_bps=args.spread_bps, slippage_bps=args.slippage_bps)
    brain = open_brain(args.db)
    cache = Path(brain.path).parent / "lab" if brain.path != ":memory:" else None
    try:
        spec_symbols = args.symbols or args.symbol
        if args.rank_at_start and spec_symbols.startswith("top:") and "-" not in spec_symbols:
            wanted = int(spec_symbols[4:])
            spec_symbols = f"top:{min(3 * wanted, 150)}"  # a wider pool, ranked by volume at the start of the history below
        else:
            wanted = None
        symbols = lab.resolve_symbols(spec_symbols, market)
    except Exception as exc:
        print(f"could not resolve the universe: {exc}", file=sys.stderr)
        return 1
    if spec["portfolio"] and len(symbols) < 4:
        print(f"the {args.rule} rule ranks a universe; give it at least four symbols, e.g. --symbols top:30", file=sys.stderr)
        return 2
    print(f"fetching {args.days} days of {args.interval} candles for {len(symbols)} symbol{'s' if len(symbols) != 1 else ''} from Binance {market}"
          + (" with funding history" if "funding" in spec["needs"] else "") + " ...", flush=True)
    markets, extras = lab.fetch_universe(symbols, args.interval, args.days, market, cache_dir=cache, workers=args.workers, funding="funding" in spec["needs"], log=print)
    if not markets:
        print("no candles fetched", file=sys.stderr)
        return 1
    if wanted:
        markets = lab.rank_at_start(markets, wanted)
        extras = {k: {s: v[s] for s in markets if s in v} for k, v in extras.items()} if extras else extras
        print(f"universe ranked by volume over the first 30 days of the history, not today: {len(markets)} symbols kept ({', '.join(sorted(markets)[:8])}, ...)")
    data = markets[symbols[0]] if len(markets) == 1 else markets
    n = len(lab.grid_points(grid))
    print(f"{sum(len(b) for b in markets.values())} candles across {len(markets)} symbols; testing {n} parameter setting{'s' if n != 1 else ''} "
          f"with {args.workers} workers, then a {args.folds}-window walk-forward ...", flush=True)
    report = lab.run(args.rule, data, grid, costs, args.interval, folds=args.folds, workers=args.workers, extras=extras)
    label = symbols[0] if len(markets) == 1 else f"{len(markets)} pairs"
    print(lab.format_report(report, label, args.interval, top=args.top))
    if not args.no_learn:
        title = lab.learn_result(brain, report, label, args.interval, args.days)
        print(f"\nthe brain remembers this as '{title}'")
    return 0


def cmd_calls(args: argparse.Namespace) -> int:
    from bigbrain.watch import Watcher

    brain = open_brain(args.db)
    rows = brain.db.execute("SELECT DISTINCT symbol, interval FROM calls ORDER BY symbol, interval").fetchall()
    if not rows:
        print("No calls yet. Run `bigbrain watch` first.")
        return 0
    for r in rows:
        if args.symbol and r["symbol"] != args.symbol.upper():
            continue
        w = Watcher(brain, r["symbol"], r["interval"])
        print(f"{r['symbol']} {r['interval']}")
        for c in w.scorecard():
            rate = f"{c['hit_rate']:.0%}" if c["hit_rate"] is not None else "  -"
            avg = f"{c['avg_return']:+.2%}" if c["avg_return"] is not None else "   -  "
            print(f"  {c['signal']:18} fired {c['fired']:3}  graded {c['graded']:3}  hit rate {rate:>4}  avg move {avg}  pending {c['pending']}")
        if args.recent:
            print("  recent calls:")
            for c in brain.db.execute(
                "SELECT signal, bar_time, price, hit, outcome_return FROM calls WHERE symbol = ? AND interval = ? ORDER BY bar_time DESC LIMIT ?",
                (r["symbol"], r["interval"], args.recent),
            ):
                status = "pending" if c["hit"] is None else ("hit" if c["hit"] else "miss") + f" ({c['outcome_return']:+.2%})"
                print(f"    {c['bar_time']}  {c['signal']:18} @ {c['price']:.2f}  {status}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    s = brain.stats()
    print(f"cells: {s['cells']}   synapses: {s['synapses']}   avg weight: {s['avg_synapse_weight']}")
    print("kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(s["kinds"].items())))
    print("top concepts: " + ", ".join(f"{k} ({v})" for k, v in s["top_concepts"].items()))
    print("hubs: " + ", ".join(f"{t} ({d})" for t, d in s["hubs"]))
    size = brain.size_report()
    rows = size["rows"]
    print(f"database: {size['bytes'] / 1e6:.1f} MB on disk; trades {rows['trades']}, post-mortems {size['cells_by_kind'].get('postmortem', 0)}, "
          f"index rows {rows['cell_tokens']}, synapses {rows['synapses']}")
    if args.journal:
        print("\nrecent activity:")
        for _, event, detail in brain.journal(args.journal):
            print(f"  {event:7} {detail}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    out = brain.export_dot() if args.format == "dot" else json.dumps(brain.export_graph(), indent=2)
    if args.output:
        Path(args.output).write_text(out)
        print(f"Wrote {args.output}")
    else:
        print(out)
    return 0


def _print_links(brain: Brain, synapses) -> None:
    for syn in synapses[:10]:
        a, b = brain.get(syn.a), brain.get(syn.b)
        if a and b:
            print(f"  linked ({syn.weight:.2f}, {syn.reason}): {a.title} <-> {b.title}")
    if len(synapses) > 10:
        print(f"  ... and {len(synapses) - 10} more links")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bigbrain", description="Big Brain Time: a self-linking trading knowledge brain.")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"brain database path (default: {DEFAULT_DB}, env BIGBRAIN_DB)")
    parser.add_argument("--version", action="version", version=f"bigbrain {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed", help="teach the brain the curriculum and every knowledge pack")
    p.add_argument("--no-packs", action="store_true", help="only the 51 core lessons")
    p.set_defaults(func=cmd_seed)

    learn = sub.add_parser("learn", help="feed the brain knowledge").add_subparsers(dest="source", required=True)

    p = learn.add_parser("text", help="learn a piece of text")
    p.add_argument("title")
    p.add_argument("content", nargs="?", help="text to learn (reads stdin if omitted)")
    p.add_argument("--kind", default="note")
    p.add_argument("--source", default="")
    p.set_defaults(func=cmd_learn_text)

    p = learn.add_parser("file", help="learn one or more text files")
    p.add_argument("paths", nargs="+")
    p.add_argument("--kind", default="note")
    p.set_defaults(func=cmd_learn_file)

    p = learn.add_parser("papers", help="read recent research papers from arXiv")
    p.add_argument("--query", "-q", default=None, help="search terms, e.g. 'momentum crypto'")
    p.add_argument("--max", type=int, default=10)
    p.set_defaults(func=cmd_learn_papers)

    p = learn.add_parser("market", help="learn from OHLCV data: indicators plus strategy backtests")
    p.add_argument("--csv", help="CSV with date,open,high,low,close,volume columns (oldest first)")
    p.add_argument("--from", dest="source", choices=["binance", "stooq"], help="fetch history online: binance (crypto, no key) or stooq (stocks, indices)")
    p.add_argument("--interval", default="1d", help="binance candle size: 1m 5m 15m 30m 1h 4h 1d 1w (default 1d)")
    p.add_argument("--limit", type=int, default=1000, help="binance candles to fetch (max 1000)")
    p.add_argument("--symbol", help="symbol: BTCUSDT for binance, aapl.us / spy.us / ^spx for stooq, or a name for a CSV")
    p.add_argument("--bars", type=int, default=400, help="bars of synthetic data when no CSV is given")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-backtest", action="store_true")
    p.set_defaults(func=cmd_learn_market)

    p = learn.add_parser("reddit", help="read what traders discuss on Reddit")
    p.add_argument("--sub", action="append", help="subreddit (repeatable); default is a curated set")
    p.add_argument("--time", default="week", choices=["day", "week", "month", "year", "all"])
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--min-score", type=int, default=5)
    p.add_argument("--no-comments", action="store_true", help="skip fetching top comments (fewer requests)")
    p.set_defaults(func=cmd_learn_reddit)

    p = learn.add_parser("github", help="read trading repositories on GitHub")
    p.add_argument("--query", "-q", action="append", help="search query (repeatable), e.g. 'topic:backtesting'")
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--no-readme", action="store_true")
    p.set_defaults(func=cmd_learn_github)

    p = learn.add_parser("url", help="read any web page (blog post, TradingView idea, docs)")
    p.add_argument("urls", nargs="+")
    p.add_argument("--kind", default="article")
    p.set_defaults(func=cmd_learn_url)

    p = learn.add_parser("feed", help="read RSS/Atom feeds (defaults to curated trading blogs)")
    p.add_argument("urls", nargs="*")
    p.add_argument("--max", type=int, default=20)
    p.add_argument("--full", action="store_true", help="fetch each linked article, not just the summary")
    p.set_defaults(func=cmd_learn_feed)

    p = sub.add_parser("feed", help="go online and learn from every source: papers, Reddit, GitHub, blogs")
    p.add_argument("--quick", action="store_true", help="fewer items per source")
    p.add_argument("--workers", type=int, default=10, help="parallel fetchers (default 10)")
    p.set_defaults(func=cmd_feed)

    p = sub.add_parser("ask", help="ask the brain a question")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=10, help="how many cells to recall")
    p.add_argument("--claude", action="store_true", help="force the Claude-powered cortex")
    p.add_argument("--offline", action="store_true", help="force the offline cortex")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--show-cells", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("recall", help="show which cells a query activates")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--no-reinforce", action="store_true", help="do not strengthen links between recalled cells")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("explain", help="show a cell and everything it is connected to")
    p.add_argument("title", help="part of the cell title")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("watch", help="watch a market at each candle close, record signals as calls, grade them")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="15m", help="1m 5m 15m 30m 1h 4h 1d 1w")
    p.add_argument("--horizon", type=int, default=12, help="bars to wait before grading a call (12 x 15m = 3 hours)")
    p.add_argument("--once", action="store_true", help="one pass, then exit (good for cron)")
    p.add_argument("--no-paper", action="store_true", help="only grade signals; do not paper trade the strategies")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("trade", help="the brain trades the top USDT pairs with a virtual wallet and learns from every trade")
    p.add_argument("--book", default="main", help="name of the trading book (separate wallets and beliefs)")
    p.add_argument("--top", type=int, default=100, help="how many USDT pairs by volume")
    p.add_argument("--interval", default="15m")
    p.add_argument("--wallet", type=float, default=1000.0, help="starting USDT (only used when the book is new)")
    p.add_argument("--market", choices=["perps", "spot"], default="perps", help="perps: long and short with funding (default); spot: long only")
    p.add_argument("--reset", action="store_true", help="close all positions and restart the wallet before trading (learning kept)")
    p.add_argument("--frozen", action="store_true", help="baseline book: same signals and rules, fixed size, no beliefs, no exit learning")
    p.add_argument("--with-frozen", action="store_true", help="also run a frozen companion book (<book>-frozen) on identical inputs, to measure what the learning adds")
    p.add_argument("--once", action="store_true")
    p.set_defaults(func=cmd_trade)

    p = sub.add_parser("backup", help="consistent snapshot of the brain; --push publishes it to the brain-backup branch on GitHub")
    p.add_argument("--dir", help="where snapshots go (default .brain/backups)")
    p.add_argument("--keep", type=int, default=14, help="local snapshots to keep")
    p.add_argument("--push", action="store_true", help="also push the snapshot to GitHub (branch brain-backup, one commit, replaced each time)")
    p.add_argument("--every", type=float, default=0, metavar="HOURS", help="keep running and back up every N hours")
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser("restore", help="replace the brain with a snapshot (latest local, a file, or --from-github)")
    p.add_argument("file", nargs="?")
    p.add_argument("--from-github", action="store_true")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("prune", help="forget old post-mortem cells and compact the database (the trader does this daily at 60 days)")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--kind", default="postmortem")
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("repair", help="rebuild a book's beliefs and exit statistics from its trade log")
    p.add_argument("--book", default="main")
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("reset", help="close all positions and restart a book's wallet; --forget also erases its history and beliefs")
    p.add_argument("--book", default="main")
    p.add_argument("--wallet", type=float, default=None, help="new starting USDT (default: the book's original)")
    p.add_argument("--forget", action="store_true", help="also erase trades, beliefs and post-mortem cells for this book")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("portfolio", help="the trading book: equity, open positions, closed trades")
    p.add_argument("--book", default="main")
    p.add_argument("--recent", type=int, default=0, metavar="N")
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("dashboard", help="live dashboard for a trading book: equity, positions with live prices, beliefs, trades, feed")
    p.add_argument("--book", default="main")
    p.add_argument("--refresh", type=float, default=10.0, help="seconds between redraws")
    p.add_argument("--once", action="store_true")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("beliefs", help="what the brain believes about each signal in each context, from its own trades")
    p.add_argument("--book", default="main")
    p.set_defaults(func=cmd_beliefs)

    p = sub.add_parser("strategy", help="trade a lab rule live in its own book: the rule decides, the trader executes, a buy-and-hold control runs beside it")
    p.add_argument("--rule", default="trend_vt")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,SOLUSDT", help="a list, or top:N")
    p.add_argument("--book", default="trend")
    p.add_argument("--interval", default="1d")
    p.add_argument("--market", default="perps", choices=("spot", "perps"))
    p.add_argument("--wallet", type=float, default=1000.0)
    p.add_argument("--grid", default="sma=30,50,100,200;short=0,1", help="settings the monthly refit chooses from (the lab's selection rule)")
    p.add_argument("--target-vol", type=float, default=0.15, help="annualized volatility target per coin (default 0.15)")
    p.add_argument("--refit-days", type=int, default=30)
    p.add_argument("--reset", action="store_true", help="close every position, restart the wallet and the control")
    p.add_argument("--rebase-control", action="store_true", help="restart only the buy-and-hold control at current quotes (positions untouched)")
    p.add_argument("--once", action="store_true")
    p.set_defaults(func=cmd_strategy)

    p = sub.add_parser("lab", help="test a trading rule honestly: parameter grid with plateau score, real costs, walk-forward, verdict")
    p.add_argument("--rule", default="psar", help="psar | sma_cross | rsi_reversion | playbook | trend_vt | funding_carry | xs_momentum (default psar)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--symbols", default="", help="a universe: 'BTCUSDT,ETHUSDT,...' or 'top:30' (by 24h volume); trades are pooled and judged on the same dates")
    p.add_argument("--rank-at-start", action="store_true", help="with --symbols top:N, choose the N by volume at the START of the history (what was knowable then), not today's winners")
    p.add_argument("--interval", default="15m")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--market", default=None, choices=("spot", "perps"), help="default spot, or perps for rules that need funding")
    p.add_argument("--grid", default="", help='parameter grid, e.g. "start=0.01,0.02,0.04;increment=0.01,0.02,0.04;maximum=0.1,0.2,0.4"; missing parameters keep defaults')
    p.add_argument("--fee", type=float, default=None, help="fee per side as a fraction (default: 0.1%% spot, 0.05%% perps; use 0 for a spread-only FX broker)")
    p.add_argument("--spread-bps", type=float, default=1.0, help="full bid-ask spread in basis points, half paid per fill (default 1; EURUSD at 0.4 pip is about 0.35)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="extra slippage per fill in basis points")
    p.add_argument("--folds", type=int, default=6, help="walk-forward windows (default 6)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--top", type=int, default=10, help="grid rows to print")
    p.add_argument("--no-learn", action="store_true", help="do not write the verdict into the brain")
    p.set_defaults(func=cmd_lab)

    p = sub.add_parser("paper", help="paper trading accounts: equity, drawdown, trades per strategy")
    p.add_argument("--symbol")
    p.add_argument("--recent", type=int, default=0, metavar="N", help="also list the last N trades")
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("calls", help="scorecard of live signals and how they turned out")
    p.add_argument("--symbol")
    p.add_argument("--recent", type=int, default=0, metavar="N", help="also list the last N calls")
    p.set_defaults(func=cmd_calls)

    p = sub.add_parser("forget", help="remove cells by source, kind or title, e.g. --source synthetic")
    p.add_argument("--source", help="substring of the source, e.g. 'synthetic', 'reddit', 'binance:BTCUSDT'")
    p.add_argument("--kind", help="cell kind: concept, paper, observation, lesson, note, article, code, discussion")
    p.add_argument("--title", help="substring of the title")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("stats", help="how big and how connected the brain is")
    p.add_argument("--journal", type=int, default=0, metavar="N", help="also show the last N events")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("graph", help="export the knowledge graph")
    p.add_argument("--format", choices=["json", "dot"], default="json")
    p.add_argument("--output", "-o")
    p.set_defaults(func=cmd_graph)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
