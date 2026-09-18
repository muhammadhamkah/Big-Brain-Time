"""GitHub: where trading systems, backtesters and indicator libraries live.

Uses GitHub's public REST API. Without a token you get 60 requests an hour,
which covers a couple of searches with READMEs; set ``GITHUB_TOKEN`` for
5000. Each repository becomes a *code* cell built from its description,
topics and README, so a question about "backtesting frameworks" can reach
the tools people actually use.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from bigbrain.brain import Brain
from bigbrain.net import HTTPStatusError, http_get, http_json

API = "https://api.github.com"
DEFAULT_QUERIES = (
    "topic:algorithmic-trading",
    "topic:trading-strategies",
    "topic:backtesting",
    "topic:quantitative-finance",
    "topic:technical-indicators",
)
MAX_README = 5000


@dataclass
class Repo:
    full_name: str
    description: str
    stars: int
    language: str
    topics: list[str] = field(default_factory=list)
    readme: str = ""
    url: str = ""


def _headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def parse_search(payload: dict) -> list[Repo]:
    repos = []
    for item in payload.get("items", []):
        repos.append(Repo(
            full_name=item.get("full_name", ""),
            description=(item.get("description") or "").strip(),
            stars=int(item.get("stargazers_count") or 0),
            language=item.get("language") or "",
            topics=list(item.get("topics") or []),
            url=item.get("html_url") or "",
        ))
    return repos


def search(query: str, max_results: int = 10) -> list[Repo]:
    from urllib.parse import quote

    url = f"{API}/search/repositories?q={quote(query)}&sort=stars&order=desc&per_page={max_results}"
    return parse_search(http_json(url, headers=_headers()))


def fetch_readme(repo: Repo) -> str:
    headers = _headers()
    headers["Accept"] = "application/vnd.github.raw+json"
    try:
        raw = http_get(f"{API}/repos/{repo.full_name}/readme", headers=headers).decode("utf-8", errors="replace")
    except HTTPStatusError:
        return ""
    return clean_markdown(raw)[:MAX_README]


def clean_markdown(text: str) -> str:
    """Strip badges, images, HTML and link syntax so the README reads as prose."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[#*`>|_-]{2,}", " ", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    lines = [" ".join(l.split()) for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


def render(repo: Repo) -> str:
    head = f"{repo.description} ({repo.stars} stars, {repo.language or 'unknown language'})."
    if repo.topics:
        head += " Topics: " + ", ".join(repo.topics) + "."
    body = f"{head}\n\n{repo.readme}" if repo.readme else head
    return f"{body}\n\nRepository: {repo.url}"


def learn_repos(brain: Brain, repos: list[Repo], with_readme: bool = True) -> list[str]:
    learned = []
    for repo in repos:
        if with_readme and not repo.readme:
            repo.readme = fetch_readme(repo)
        body = render(repo)
        if len(body) < 150:
            continue
        cell, _ = brain.learn("code", repo.full_name, body, source=repo.url or f"github:{repo.full_name}", extra_concepts=["open source"])
        learned.append(cell.title)
    return learned


def learn_github(brain: Brain, queries: tuple[str, ...] = DEFAULT_QUERIES, max_results: int = 10, with_readme: bool = True) -> dict[str, list[str]]:
    return {q: learn_repos(brain, search(q, max_results), with_readme=with_readme) for q in queries}
