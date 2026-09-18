"""The brain's diet: every online source it feeds from, and the one-shot run.

``bigbrain feed`` calls ``run_all``: it builds one fetch job per topic,
subreddit, repository query, feed and reference page, runs them on a pool of
workers (politeness per host is enforced in ``bigbrain.net``), and learns
each job's items on the main thread as they arrive. A failing job never
stops the others. Knowledge packs (``bigbrain.knowledge``) contribute their
own curated sources on top of the defaults below.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from bigbrain.brain import Brain
from bigbrain.ingest import Item, learn_items

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


@dataclass
class Job:
    group: str  # papers | reddit | github | feeds | pages
    name: str
    fetch: Callable[[], list[Item]]


@dataclass
class Report:
    learned: dict[str, int] = field(default_factory=dict)
    failed: dict[str, list[str]] = field(default_factory=dict)
    jobs: int = 0

    def summary(self) -> dict[str, tuple[int, str | None]]:
        groups = sorted(set(self.learned) | set(self.failed))
        out = {}
        for g in groups:
            n = self.learned.get(g, 0)
            errs = self.failed.get(g, [])
            out[g] = (n, None if n or not errs else errs[-1])
        return out


def _dedupe(*seqs) -> list[str]:
    seen: list[str] = []
    for seq in seqs:
        for v in seq:
            if v not in seen:
                seen.append(v)
    return seen


def build_jobs(quick: bool = False) -> list[Job]:
    from bigbrain.ingest import github, papers, reddit, web
    from bigbrain.knowledge import merged_sources

    packs = merged_sources()
    n_papers = 5 if quick else 10
    n_items = 10 if quick else 25
    jobs: list[Job] = []

    topics = _dedupe(PAPER_TOPICS, packs["arxiv_topics"])
    for topic in topics[: 4 if quick else len(topics)]:
        jobs.append(Job("papers", topic, lambda t=topic: papers.items(papers.fetch(t, max_results=n_papers, retries=2))))

    subs = _dedupe(reddit.DEFAULT_SUBREDDITS, packs["subreddits"])
    for sub in subs[: 4 if quick else len(subs)]:
        jobs.append(Job("reddit", f"r/{sub}", lambda s=sub: reddit.items(reddit.fetch_posts(s, limit=n_items, with_comments=not quick))))

    queries = _dedupe(github.DEFAULT_QUERIES, packs["github_queries"])
    for q in queries[: 2 if quick else len(queries)]:
        jobs.append(Job("github", q, lambda q=q: github.items(github.fetch_repos(q, 5 if quick else 10, with_readme=True))))

    feeds = _dedupe(DEFAULT_FEEDS, packs["feeds"])
    for url in feeds[: 3 if quick else len(feeds)]:
        jobs.append(Job("feeds", url, lambda u=url: web.feed_items(u, max_items=n_items)))

    for url in ([] if quick else packs["urls"]):
        jobs.append(Job("pages", url, lambda u=url: [web.url_item(u)]))
    return jobs


def run_all(brain: Brain, quick: bool = False, workers: int = 10, log: Callable[[str], None] = lambda s: None) -> dict[str, tuple[int, str | None]]:
    """Feed the brain from every source. Returns {group: (items learned, error or None)}."""
    from bigbrain.ingest import textbook

    log("Curriculum and knowledge packs ...")
    seeded = textbook.seed(brain)
    log(f"  {seeded} new lessons")

    jobs = build_jobs(quick)
    report = Report(jobs=len(jobs))
    report.learned["curriculum"] = seeded
    log(f"Fetching from {len(jobs)} sources with {workers} workers ...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(job.fetch): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                items = future.result()
            except Exception as exc:
                msg = str(exc).splitlines()[0][:160]
                report.failed.setdefault(job.group, []).append(f"{job.name}: {msg}")
                log(f"  {job.group:7} {job.name}: failed ({msg})")
                continue
            titles = learn_items(brain, items)
            report.learned[job.group] = report.learned.get(job.group, 0) + len(titles)
            log(f"  {job.group:7} {job.name}: {len(titles)} learned")
    return report.summary()
