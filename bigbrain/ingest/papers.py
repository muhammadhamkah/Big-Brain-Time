"""Research paper ingestion from arXiv (quantitative finance and friends).

Uses the public arXiv Atom API, so it needs network access but no key. Each
paper becomes one cell built from its title and abstract; concept extraction
then wires it to every strategy, indicator and observation it relates to.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from bigbrain.brain import Brain

ARXIV_API = "https://export.arxiv.org/api/query"
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
    retries: int = 4,
) -> list[Paper]:
    """Query the arXiv API.

    arXiv answers 406 both to requests without an ``Accept`` header and, when it
    is shedding load, as a throttle. So we always send the header and retry
    406/429/5xx with a growing pause.
    """
    params = {
        "search_query": build_query(search, categories),
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    headers = {
        "User-Agent": "bigbrain/0.1 (https://github.com/muhammadhamkah/Big-Brain-Time)",
        "Accept": "application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
    }
    delay = 3.0
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return parse_atom(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (406, 429, 500, 502, 503, 504) or attempt == retries:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"arXiv request failed: {last_error}")


def learn_papers(brain: Brain, papers: list[Paper]) -> list[str]:
    """Teach the brain each paper. Returns the titles learned."""
    learned = []
    for p in papers:
        by = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        content = f"{p.abstract}\n\nAuthors: {by}. Published {p.published}. {p.url}"
        cell, _ = brain.learn("paper", p.title, content, source=f"arXiv:{p.id}")
        learned.append(cell.title)
    return learned
