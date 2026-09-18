"""Research paper ingestion from arXiv (quantitative finance and friends).

Uses the public arXiv Atom API, so it needs network access but no key. Each
paper becomes one cell built from its title and abstract; concept extraction
then wires it to every strategy, indicator and observation it relates to.
"""

from __future__ import annotations

import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from bigbrain.brain import Brain

# arXiv serves the same API from both hosts; its edge intermittently rejects one
# with HTTP 406 for minutes at a time, so we alternate between them.
ARXIV_HOSTS = ("https://export.arxiv.org/api/query", "https://arxiv.org/api/query")
ARXIV_API = ARXIV_HOSTS[0]

# arXiv's edge returns 406 to some request shapes while a bare curl succeeds.
# The minimal profile mirrors curl; later ones are tried if the edge rejects it.
HEADER_PROFILES: tuple[dict[str, str], ...] = (
    {"User-Agent": "bigbrain/0.1", "Accept": "*/*"},
    {"User-Agent": "Mozilla/5.0 (compatible; bigbrain/0.1; +https://github.com/muhammadhamkah/Big-Brain-Time)", "Accept": "*/*"},
    {"User-Agent": "bigbrain/0.1", "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.5"},
)
ATOM = "{http://www.w3.org/2005/Atom}"

# Categories the brain cares about: quantitative finance, plus stat/ML finance crossovers.
DEFAULT_CATEGORIES = ("q-fin.TR", "q-fin.PM", "q-fin.ST", "q-fin.CP", "q-fin.RM")


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    url: str


def build_query(search: str | None, categories: tuple[str, ...] = DEFAULT_CATEGORIES) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    if search:
        return f"({cats}) AND all:{search}"
    return cats


def parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        pid = (entry.findtext(f"{ATOM}id") or "").strip()
        papers.append(
            Paper(
                id=pid.rsplit("/", 1)[-1],
                title=" ".join((entry.findtext(f"{ATOM}title") or "").split()),
                abstract=" ".join((entry.findtext(f"{ATOM}summary") or "").split()),
                authors=[" ".join((a.findtext(f"{ATOM}name") or "").split()) for a in entry.findall(f"{ATOM}author")],
                published=(entry.findtext(f"{ATOM}published") or "")[:10],
                url=pid,
            )
        )
    return papers


def fetch(
    search: str | None = None,
    max_results: int = 10,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    timeout: float = 30.0,
    retries: int = 6,
    hosts: tuple[str, ...] = ARXIV_HOSTS,
) -> list[Paper]:
    """Query the arXiv API, alternating hosts and backing off on throttling.

    arXiv answers 406 to some request shapes and, when shedding load, to every
    client on a host for minutes at a time. Each attempt rotates the host and
    the header profile; after the first two attempts the category filter is
    dropped as a further fallback in case the combined query is rejected.
    """
    delay = 2.0
    last_error: str | None = None
    for attempt in range(retries + 1):
        host = hosts[attempt % len(hosts)]
        headers = HEADER_PROFILES[attempt % len(HEADER_PROFILES)]
        cats = categories if attempt < 2 or not search else ()
        query = build_query(search, cats) if cats else (f"all:{search}" if search else build_query(None, categories))
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{host}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return parse_atom(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read(600).decode("utf-8", errors="replace").strip()
            last_error = f"HTTP {exc.code} from {url}\n  response headers: {dict(exc.headers)}\n  response body: {body!r}"
            if exc.code not in (403, 406, 415, 429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = f"{exc} from {url}"
        if attempt == retries:
            break
        time.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 2, 30.0)
    raise RuntimeError(f"arXiv rejected every attempt on {len(hosts)} hosts; last error: {last_error}")


def learn_papers(brain: Brain, papers: list[Paper]) -> list[str]:
    """Teach the brain each paper. Returns the titles learned."""
    learned = []
    for p in papers:
        by = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        content = f"{p.abstract}\n\nAuthors: {by}. Published {p.published}. {p.url}"
        cell, _ = brain.learn("paper", p.title, content, source=f"arXiv:{p.id}")
        learned.append(cell.title)
    return learned
