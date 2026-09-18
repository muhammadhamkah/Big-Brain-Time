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
    n = seed(brain)
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

    brain = open_brain(args.db)
    if args.csv:
        bars = load_csv(args.csv)
        symbol = args.symbol or Path(args.csv).stem.upper()
        source = f"csv:{args.csv}"
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

    p = sub.add_parser("seed", help="teach the brain the foundational trading curriculum")
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
    p.add_argument("--symbol", help="symbol name (defaults to the file name)")
    p.add_argument("--bars", type=int, default=400, help="bars of synthetic data when no CSV is given")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-backtest", action="store_true")
    p.set_defaults(func=cmd_learn_market)

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
