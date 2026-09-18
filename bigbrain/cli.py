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
        results = learn_backtests(brain, symbol, bars)
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


def cmd_stats(args: argparse.Namespace) -> int:
    brain = open_brain(args.db)
    s = brain.stats()
    print(f"cells: {s['cells']}   synapses: {s['synapses']}   avg weight: {s['avg_synapse_weight']}")
    print("kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(s["kinds"].items())))
    print("top concepts: " + ", ".join(f"{k} ({v})" for k, v in s["top_concepts"].items()))
    print("hubs: " + ", ".join(f"{t} ({d})" for t, d in s["hubs"]))
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
    p.add_argument("-k", type=int, default=8, help="how many cells to recall")
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
