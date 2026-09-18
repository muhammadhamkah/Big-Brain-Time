"""Read any web page or RSS/Atom feed and learn it.

The page reader keeps the readable text (title, headings, paragraphs, list
items) and drops scripts, styles, navigation and boilerplate. It is enough
for blog posts, documentation, forum threads and idea pages such as
TradingView's, when their content is in the HTML rather than loaded later
by JavaScript.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urlsplit

from bigbrain.brain import Brain
from bigbrain.net import http_text

MAX_CONTENT = 6000  # characters kept per learned page


@dataclass
class Page:
    url: str
    title: str
    text: str
    description: str = ""


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "svg", "form", "iframe", "template"}
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "section", "article", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.description = ""
        self.chunks: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = dict(attrs)
            if a.get("name") in ("description", "twitter:description") or a.get("property") == "og:description":
                if not self.description and a.get("content"):
                    self.description = a["content"].strip()
        elif tag in self.BLOCK:
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.chunks.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip_depth:
            self.chunks.append(data)


def extract(url: str, raw_html: str) -> Page:
    parser = _TextExtractor()
    parser.feed(raw_html)
    lines = [" ".join(line.split()) for line in "".join(parser.chunks).splitlines()]
    # Drop tiny fragments (menu items, buttons) so what remains reads like prose.
    text = "\n".join(line for line in lines if len(line) >= 40 or line.endswith((".", ":", "?", "!")) and len(line) >= 20)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    title = " ".join("".join(parser.title_parts).split()) or urlsplit(url).netloc
    return Page(url=url, title=html.unescape(title), text=text, description=html.unescape(parser.description))


def fetch_page(url: str) -> Page:
    return extract(url, http_text(url, headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}))


def learn_page(brain: Brain, page: Page, kind: str = "article") -> tuple[str, int]:
    """Teach the brain a page. Returns (title, links created)."""
    body = page.text[:MAX_CONTENT]
    if page.description and page.description not in body:
        body = f"{page.description}\n\n{body}"
    if len(body) < 200:
        raise ValueError(f"{page.url} has too little readable text (is it rendered by JavaScript?)")
    cell, synapses = brain.learn(kind, page.title[:200], f"{body}\n\nSource: {page.url}", source=page.url)
    return cell.title, len(synapses)


def learn_url(brain: Brain, url: str, kind: str = "article") -> tuple[str, int]:
    return learn_page(brain, fetch_page(url), kind=kind)


# ------------------------------------------------------------------ RSS / Atom
@dataclass
class FeedItem:
    title: str
    link: str
    summary: str
    published: str = ""


def parse_feed(xml_text: str) -> list[FeedItem]:
    root = ET.fromstring(xml_text)
    items: list[FeedItem] = []
    atom = "{http://www.w3.org/2005/Atom}"
    for entry in root.iter(f"{atom}entry"):
        link = ""
        for l in entry.findall(f"{atom}link"):
            if l.get("rel", "alternate") == "alternate":
                link = l.get("href", "")
                break
        items.append(FeedItem(
            title=" ".join((entry.findtext(f"{atom}title") or "").split()),
            link=link,
            summary=_strip_tags(entry.findtext(f"{atom}summary") or entry.findtext(f"{atom}content") or ""),
            published=(entry.findtext(f"{atom}updated") or entry.findtext(f"{atom}published") or "")[:10],
        ))
    for item in root.iter("item"):
        content = item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or item.findtext("description") or ""
        items.append(FeedItem(
            title=" ".join((item.findtext("title") or "").split()),
            link=(item.findtext("link") or "").strip(),
            summary=_strip_tags(content),
            published=(item.findtext("pubDate") or "")[:25],
        ))
    return [i for i in items if i.title]


def _strip_tags(text: str) -> str:
    parser = _TextExtractor()
    parser.feed(text)
    return " ".join("".join(parser.chunks).split())


def learn_feed(brain: Brain, feed_url: str, max_items: int = 20, follow_links: bool = False) -> list[str]:
    """Learn every item in a feed. With ``follow_links`` the full article is fetched too."""
    items = parse_feed(http_text(feed_url, headers={"Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*;q=0.5"}))
    learned: list[str] = []
    for item in items[:max_items]:
        body = item.summary
        if follow_links and item.link:
            try:
                body = fetch_page(item.link).text[:MAX_CONTENT] or body
            except Exception:
                pass
        if len(body) < 120:
            continue
        cell, _ = brain.learn("article", item.title[:200], f"{body}\n\nSource: {item.link or feed_url}", source=item.link or feed_url)
        learned.append(cell.title)
    return learned
