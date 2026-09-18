"""Reddit: what traders are actually discussing.

Reddit serves every subreddit and every thread as an RSS/Atom feed, which
needs no account and is not blocked for scripts the way the JSON listings
are. The JSON route is tried first (it carries scores); on 403 the brain
falls back to RSS. Each post that carries enough substance (its own text
plus the best comments) becomes a *discussion* cell. Forum knowledge is noisy, so the brain trusts it less
than papers when recalling (see ``Brain.KIND_TRUST``), but it is where
practical wisdom, tooling tips and strategy failures get talked about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bigbrain.brain import Brain
from bigbrain.net import HTTPStatusError, http_json, http_text

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


def rss_listing_url(subreddit: str, sort: str = "top", time: str = "week", limit: int = 25) -> str:
    return f"https://www.reddit.com/r/{subreddit}/{sort}/.rss?t={time}&limit={limit}"


def rss_comments_url(post: Post, limit: int = 10) -> str:
    return f"{post.permalink}.rss?sort=top&limit={limit}&depth=1"


_ID_RE = re.compile(r"/comments/([a-z0-9]+)/")


def parse_rss_listing(xml_text: str, subreddit: str) -> list[Post]:
    """Posts from a subreddit RSS feed. Scores are not in RSS, so they read as unknown (-1)."""
    from bigbrain.ingest.web import parse_feed

    posts = []
    for item in parse_feed(xml_text):
        m = _ID_RE.search(item.link)
        if not m:
            continue
        text = item.summary
        # Reddit appends "submitted by /u/x [link] [comments]" boilerplate to every entry
        text = re.sub(r"submitted by\s+/u/\S+.*$", "", text).strip()
        posts.append(Post(id=m.group(1), subreddit=subreddit, title=item.title, text=text, score=-1, num_comments=-1, url=item.link))
    return posts


def parse_rss_comments(xml_text: str, post_id: str) -> list[tuple[int, str]]:
    """Comments from a thread RSS feed: every entry except the post itself. Scores are unknown."""
    from bigbrain.ingest.web import parse_feed

    out = []
    for item in parse_feed(xml_text):
        if f"/comments/{post_id}/" in item.link and item.link.rstrip("/").endswith(post_id):
            continue  # the submission itself
        body = re.sub(r"/u/\S+\s*$", "", item.summary).strip()
        if len(body) >= 80 and body not in ("[deleted]", "[removed]"):
            out.append((0, body))
    return out


def fetch_posts(subreddit: str, time: str = "week", limit: int = 25, min_score: int = 5, with_comments: bool = True, max_comments: int = 5) -> list[Post]:
    """Top posts of a subreddit, via JSON when allowed and RSS otherwise."""
    try:
        posts = [p for p in parse_listing(http_json(listing_url(subreddit, "top", time, limit))) if p.score >= min_score]
        via_rss = False
    except HTTPStatusError as exc:
        if exc.status not in (403, 429):
            raise
        posts = parse_rss_listing(http_text(rss_listing_url(subreddit, "top", time, limit)), subreddit)
        via_rss = True
    if with_comments:
        for post in posts:
            if post.num_comments == 0:
                continue
            try:
                if via_rss:
                    post.comments = parse_rss_comments(http_text(rss_comments_url(post)), post.id)[:max_comments]
                else:
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
            tag = f"({score} points) " if score > 0 else ""
            parts.append(f"- {tag}{' '.join(body.split())[:800]}")
    meta = f"r/{post.subreddit}"
    if post.score >= 0:
        meta += f", {post.score} points, {post.num_comments} comments"
    parts.append(f"{meta}. {post.permalink}")
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
