"""Research paper ingestion from arXiv (quantitative finance and friends).

Uses the public arXiv Atom API, so it needs network access but no key. Each
paper becomes one cell built from its title and abstract; concept extraction
then wires it to every strategy, indicator and observation it relates to.
"""

from __future__ import annotations

import http.client
import random
import ssl
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from bigbrain.brain import Brain

# arXiv serves the same API from both hosts; its edge intermittently rejects one
# with HTTP 406 for minutes at a time, so we alternate between them.
ARXIV_HOSTS = ("https://export.arxiv.org/api/query", "https://arxiv.org/api/query")
ARXIV_API = ARXIV_HOSTS[0]

# Headers are written in this exact order. arXiv's edge (Fastly) answers 406 on a
# cache miss to the request shape Python's urllib writes (Accept-Encoding before
# Host, plus Connection: close), while this curl-like shape reaches the origin.
REQUEST_HEADERS: tuple[tuple[str, str], ...] = (
    ("User-Agent", "bigbrain/0.1"),
    ("Accept", "*/*"),
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


def search_term(search: str) -> str:
    """Turn a human search into an arXiv query term.

    Multi-word searches become a quoted phrase, which arXiv's grammar requires
    (an unquoted second word is an invalid term and the edge answers 406).
    Anything that already looks like arXiv syntax (a field prefix or quotes)
    is passed through unchanged, so power users can write ``ti:momentum``.
    """
    search = " ".join(search.split())
    if ":" in search or search.startswith('"'):
        return search
    if " " in search:
        return f'all:"{search}"'
    return f"all:{search}"


def build_query(search: str | None, categories: tuple[str, ...] = DEFAULT_CATEGORIES) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    if search and categories:
        return f"({cats}) AND {search_term(search)}"
    if search:
        return search_term(search)
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


class HTTPStatusError(Exception):
    def __init__(self, status: int, url: str, headers: dict[str, str], body: bytes) -> None:
        super().__init__(f"HTTP {status} from {url}")
        self.status, self.url, self.headers, self.body = status, url, headers, body


def http_get(url: str, timeout: float = 30.0, max_redirects: int = 3) -> bytes:
    """GET ``url`` writing headers curl-style: Host first, then REQUEST_HEADERS, nothing else."""
    for _ in range(max_redirects + 1):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            raise ValueError(f"only https URLs are supported: {url}")
        conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout, context=ssl.create_default_context())
        try:
            path = parts.path + (f"?{parts.query}" if parts.query else "")
            conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", parts.hostname)
            for name, value in REQUEST_HEADERS:
                conn.putheader(name, value)
            conn.endheaders()
            resp = conn.getresponse()
            body = resp.read()
            headers = {k: v for k, v in resp.getheaders()}
        finally:
            conn.close()
        if resp.status in (301, 302, 303, 307, 308) and resp.getheader("Location"):
            url = urllib.parse.urljoin(url, resp.getheader("Location"))
            continue
        if resp.status != 200:
            raise HTTPStatusError(resp.status, url, headers, body)
        return body
    raise HTTPStatusError(310, url, {}, b"too many redirects")


RETRY_STATUSES = frozenset({403, 406, 415, 429, 500, 502, 503, 504})


def fetch(
    search: str | None = None,
    max_results: int = 10,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    timeout: float = 30.0,
    retries: int = 6,
    hosts: tuple[str, ...] = ARXIV_HOSTS,
) -> list[Paper]:
    """Query the arXiv API, alternating hosts and backing off on throttling.

    arXiv answers 406 to requests shaped like Python's default client (see
    ``http_get``) and, when shedding load, to every client on a host for
    minutes at a time. Each attempt rotates the host; after the first two
    attempts the category filter is dropped in case the combined query is what
    is rejected.
    """
    delay = 3.0  # arXiv asks for at least three seconds between requests
    last_error: str | None = None
    for attempt in range(retries + 1):
        host = hosts[attempt % len(hosts)]
        cats = categories if attempt < 2 or not search else ()
        query = build_query(search, cats) if (cats or search) else build_query(None, categories)
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{host}?{urllib.parse.urlencode(params)}"
        try:
            return parse_atom(http_get(url, timeout=timeout).decode("utf-8"))
        except HTTPStatusError as exc:
            body = exc.body[:600].decode("utf-8", errors="replace").strip()
            last_error = f"HTTP {exc.status} from {url}\n  response headers: {exc.headers}\n  response body: {body!r}"
            if exc.status not in RETRY_STATUSES:
                raise
        except (OSError, TimeoutError) as exc:
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
