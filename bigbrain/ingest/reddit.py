"""Reddit: what traders are actually discussing.

Uses Reddit's public JSON listings, which need no account. Each post that
carries enough substance (its own text plus the best comments) becomes a
*discussion* cell. Forum knowledge is noisy, so the brain trusts it less
than papers when recalling (see ``Brain.KIND_TRUST``), but it is where
practical wisdom, tooling tips and strategy failures get talked about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bigbrain.brain import Brain
from bigbrain.net import http_json

DEFAULT_SUBREDDITS = ("algotrading", "quant", "quantfinance", "Daytrading", "options", "stocks", "Forex", "CryptoCurrency", "investing")

# Sorting windows Reddit accepts for /top
TIMES = ("hour", "day", "week", "month", "year", "all")


@dataclass
class Post:
    id: str
    subreddit: str
    title: str
    text: str
    score: int
    num_comments: int
    url: str
    comments: list[tuple[int, str]] = field(default_factory=list)  # (score, body)

    @property
    def permalink(self) -> str:
        return f"https://www.reddit.com/r/{self.subreddit}/comments/{self.id}/"


def listing_url(subreddit: str, sort: str = "top", time: str = "week", limit: int = 25) -> str:
    return f"https://www.reddit.com/r/{subreddit}/{sort}.json?t={time}&limit={limit}&raw_json=1"


def comments_url(post: Post, limit: int = 10) -> str:
    return f"https://www.reddit.com/r/{post.subreddit}/comments/{post.id}.json?sort=top&limit={limit}&depth=1&raw_json=1"


def parse_listing(payload: dict) -> list[Post]:
    posts = []
    for child in payload.get("data", {}).get("children", []):
        d = child.get("data", {})
        if child.get("kind") != "t3" or d.get("stickied") or d.get("over_18"):
            continue
        posts.append(Post(
            id=d.get("id", ""),
            subreddit=d.get("subreddit", ""),
            title=" ".join((d.get("title") or "").split()),
            text=(d.get("selftext") or "").strip(),
            score=int(d.get("score") or 0),
            num_comments=int(d.get("num_comments") or 0),
            url=d.get("url") or "",
        ))
    return posts


def parse_comments(payload: list) -> list[tuple[int, str]]:
    if len(payload) < 2:
        return []
    out = []
    for child in payload[1].get("data", {}).get("children", []):
        d = child.get("data", {})
        body = (d.get("body") or "").strip()
        if child.get("kind") == "t1" and len(body) >= 80 and body not in ("[deleted]", "[removed]"):
            out.append((int(d.get("score") or 0), body))
    return sorted(out, key=lambda kv: -kv[0])


def fetch_posts(subreddit: str, time: str = "week", limit: int = 25, min_score: int = 5, with_comments: bool = True, max_comments: int = 5) -> list[Post]:
    posts = [p for p in parse_listing(http_json(listing_url(subreddit, "top", time, limit))) if p.score >= min_score]
    if with_comments:
        for post in posts:
            if post.num_comments == 0:
                continue
            try:
                post.comments = parse_comments(http_json(comments_url(post)))[:max_comments]
            except Exception:
                post.comments = []
    return posts


def render(post: Post) -> str:
    parts = []
    if post.text:
        parts.append(re.sub(r"\n{3,}", "\n\n", post.text)[:3000])
    if post.comments:
        parts.append("Top comments:")
        for score, body in post.comments:
            parts.append(f"- ({score} points) {' '.join(body.split())[:800]}")
    parts.append(f"r/{post.subreddit}, {post.score} points, {post.num_comments} comments. {post.permalink}")
    return "\n\n".join(parts)


def learn_posts(brain: Brain, posts: list[Post], min_chars: int = 200) -> list[str]:
    learned = []
    for post in posts:
        body = render(post)
        if len(body) < min_chars:
            continue
        cell, _ = brain.learn("discussion", post.title[:200], body, source=f"reddit:r/{post.subreddit}")
        learned.append(cell.title)
    return learned


def learn_subreddits(brain: Brain, subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS, time: str = "week", limit: int = 25, min_score: int = 5, with_comments: bool = True) -> dict[str, list[str]]:
    result = {}
    for sub in subreddits:
        result[sub] = learn_posts(brain, fetch_posts(sub, time=time, limit=limit, min_score=min_score, with_comments=with_comments))
    return result
