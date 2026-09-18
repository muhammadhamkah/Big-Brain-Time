"""The brain's diet: every online source it feeds from, and the one-shot run.

``bigbrain feed`` calls ``run_all`` which walks each source in turn, keeps
going when one fails, and reports what was learned. Add sources here.
"""

from __future__ import annotations

import time
from typing import Callable

from bigbrain.brain import Brain

PAPER_TOPICS = (
    "momentum", "mean reversion", "trend following", "volatility forecasting", "market microstructure",
    "portfolio optimization", "risk management", "reinforcement learning trading", "cryptocurrency", "options pricing",
)

# Public RSS/Atom feeds from quantitative trading blogs and aggregators.
DEFAULT_FEEDS = (
    "https://quantocracy.com/feed/",
    "https://www.quantstart.com/feed/",
    "https://robotwealth.com/feed/",
    "https://alphaarchitect.com/feed/",
)


def run_all(brain: Brain, quick: bool = False, log: Callable[[str], None] = lambda s: None) -> dict[str, tuple[int, str | None]]:
    """Feed the brain from every source. Returns {source: (items learned, error or None)}."""
    from bigbrain.ingest import github, papers, reddit, textbook, web

    report: dict[str, tuple[int, str | None]] = {}
    n_papers = 5 if quick else 10
    n_items = 10 if quick else 25

    log("Curriculum ...")
    report["curriculum"] = (textbook.seed(brain), None)

    log("arXiv papers ...")
    count, err = 0, None
    for topic in PAPER_TOPICS[: 4 if quick else len(PAPER_TOPICS)]:
        try:
            titles = papers.learn_papers(brain, papers.fetch(topic, max_results=n_papers, retries=2))
            log(f"  {topic}: {len(titles)} papers")
            count += len(titles)
        except Exception as exc:
            err = str(exc).splitlines()[0]
            log(f"  {topic}: failed ({err})")
        time.sleep(1)
    report["papers"] = (count, err if count == 0 else None)

    log("Reddit ...")
    count, err = 0, None
    subs = reddit.DEFAULT_SUBREDDITS[: 4 if quick else len(reddit.DEFAULT_SUBREDDITS)]
    for sub in subs:
        try:
            titles = reddit.learn_posts(brain, reddit.fetch_posts(sub, limit=n_items, with_comments=not quick))
            log(f"  r/{sub}: {len(titles)} discussions")
            count += len(titles)
        except Exception as exc:
            err = str(exc).splitlines()[0]
            log(f"  r/{sub}: failed ({err})")
    report["reddit"] = (count, err if count == 0 else None)

    log("GitHub ...")
    count, err = 0, None
    for q in github.DEFAULT_QUERIES[: 2 if quick else len(github.DEFAULT_QUERIES)]:
        try:
            titles = github.learn_repos(brain, github.search(q, 5 if quick else 10), with_readme=True)
            log(f"  {q}: {len(titles)} repositories")
            count += len(titles)
        except Exception as exc:
            err = str(exc).splitlines()[0]
            log(f"  {q}: failed ({err})")
    report["github"] = (count, err if count == 0 else None)

    log("Blogs and feeds ...")
    count, err = 0, None
    for url in DEFAULT_FEEDS[: 3 if quick else len(DEFAULT_FEEDS)]:
        try:
            titles = web.learn_feed(brain, url, max_items=n_items)
            log(f"  {url}: {len(titles)} articles")
            count += len(titles)
        except Exception as exc:
            err = str(exc).splitlines()[0]
            log(f"  {url}: failed ({err})")
    report["feeds"] = (count, err if count == 0 else None)
    return report
