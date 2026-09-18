"""Parsers and learners for the online senses, tested against fixtures (no network)."""

import json
import unittest
from unittest import mock

from bigbrain import Brain
from bigbrain.ingest import github, reddit, web
from bigbrain.ingest.textbook import seed

LISTING = {"data": {"children": [
    {"kind": "t3", "data": {"id": "abc", "subreddit": "algotrading", "title": "My mean reversion bot blew up", "selftext": "I ran an RSI mean reversion strategy on crypto with 10x leverage and no stop loss. " * 4, "score": 120, "num_comments": 40, "url": "https://www.reddit.com/r/algotrading/comments/abc/x/"}},
    {"kind": "t3", "data": {"id": "low", "subreddit": "algotrading", "title": "low score", "selftext": "meh", "score": 1, "num_comments": 0, "url": ""}},
    {"kind": "t3", "data": {"id": "sticky", "subreddit": "algotrading", "title": "Weekly thread", "selftext": "", "score": 500, "num_comments": 3, "url": "", "stickied": True}},
]}}
COMMENTS = [{"data": {}}, {"data": {"children": [
    {"kind": "t1", "data": {"body": "Position sizing is everything. Risk 1% per trade and your drawdown stays survivable; leverage without stops is how accounts die.", "score": 55}},
    {"kind": "t1", "data": {"body": "lol", "score": 900}},
    {"kind": "t1", "data": {"body": "[deleted]", "score": 3}},
]}}]

SEARCH = {"items": [
    {"full_name": "someone/backtester", "description": "A fast event-driven backtesting framework for trading strategies", "stargazers_count": 4200, "language": "Python", "topics": ["backtesting", "trading"], "html_url": "https://github.com/someone/backtester"},
]}
README = """# backtester\n\n![badge](https://img.shields.io/x.svg)\n\nAn event-driven **backtesting** engine. Supports [slippage](docs/slippage.md) and commission models, walk-forward optimization, and position sizing by volatility.\n\n```python\nimport backtester\n```\n\n<img src='x.png'>\n"""

HTML = """<html><head><title>Why Trend Following Works &amp; When It Fails</title>
<meta name="description" content="A look at trend following across regimes."></head>
<body><nav><a href="/">Home</a><a href="/about">About</a></nav>
<article><h1>Why Trend Following Works</h1>
<p>Trend following profits from persistent moves and suffers in sideways markets, which is why position sizing and a trailing stop matter more than the entry signal.</p>
<p>Across futures markets the strategy shows positive skew: many small losses and a few large gains during crises.</p>
<script>var x = 1;</script></article>
<footer>Copyright</footer></body></html>"""

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>Blog</title>
<item><title>Volatility targeting explained</title><link>https://example.com/vol</link><description>&lt;p&gt;Scaling position size inversely with realized volatility keeps risk constant across regimes and improves the Sharpe ratio of most strategies.&lt;/p&gt;</description><pubDate>Mon, 01 Sep 2026 00:00:00 GMT</pubDate></item>
<item><title>short</title><link>https://example.com/s</link><description>tiny</description></item>
</channel></rss>"""


class RedditTests(unittest.TestCase):
    def test_parse_listing_filters_stickies(self):
        posts = reddit.parse_listing(LISTING)
        self.assertEqual([p.id for p in posts], ["abc", "low"])

    def test_parse_comments_keeps_substantive(self):
        comments = reddit.parse_comments(COMMENTS)
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0][0], 55)

    def test_learn_posts_links_to_curriculum(self):
        brain = Brain()
        seed(brain)
        with mock.patch.object(reddit, "http_json", side_effect=[LISTING, COMMENTS]):
            posts = reddit.fetch_posts("algotrading", min_score=5)
        self.assertEqual(len(posts), 1)
        titles = reddit.learn_posts(brain, posts)
        cell = brain.find(titles[0])[0]
        self.assertEqual(cell.kind, "discussion")
        self.assertIn("mean reversion", cell.concepts)
        self.assertIn("leverage", cell.concepts)
        self.assertIn("Top comments:", cell.content)
        neighbors = [c.title for c, _ in brain.neighbors(cell.id, limit=30)]
        self.assertIn("Mean reversion", neighbors)

    def test_discussions_rank_below_concepts(self):
        brain = Brain()
        seed(brain)
        with mock.patch.object(reddit, "http_json", side_effect=[LISTING, COMMENTS]):
            reddit.learn_posts(brain, reddit.fetch_posts("algotrading"))
        results = brain.recall("mean reversion with leverage and no stop loss", k=5)
        kinds = [r.cell.kind for r in results]
        self.assertIn("discussion", kinds)
        self.assertEqual(kinds[0], "concept")


REDDIT_RSS = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Walk-forward results for my SMA crossover bot</title>
<link href="https://www.reddit.com/r/algotrading/comments/xyz1/walkforward_results/" />
<content type="html">&lt;div&gt;&lt;p&gt;Ran a 20/50 SMA crossover with walk-forward optimization on 12 futures. Sharpe 0.8 out of sample, max drawdown 18%, transaction costs included. Position sizing by ATR.&lt;/p&gt;&lt;/div&gt; submitted by /u/someone &lt;a href="x"&gt;[link]&lt;/a&gt; &lt;a href="y"&gt;[comments]&lt;/a&gt;</content>
<updated>2026-09-15T10:00:00+00:00</updated></entry>
<entry><title>Not a post</title><link href="https://www.reddit.com/r/algotrading/" /><content type="html">x</content></entry>
</feed>"""

REDDIT_THREAD_RSS = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Walk-forward results for my SMA crossover bot</title><link href="https://www.reddit.com/r/algotrading/comments/xyz1/walkforward_results/" /><content type="html">&lt;p&gt;the post body again&lt;/p&gt;</content></entry>
<entry><title>/u/quant_guy on Walk-forward results</title><link href="https://www.reddit.com/r/algotrading/comments/xyz1/walkforward_results/abc123/" /><content type="html">&lt;p&gt;An 18% drawdown on Sharpe 0.8 is about what you should expect; halve your size if you cannot stomach it, and remember out-of-sample drawdowns tend to be deeper than backtested ones.&lt;/p&gt; /u/quant_guy</content></entry>
<entry><title>/u/lol on Walk-forward results</title><link href="https://www.reddit.com/r/algotrading/comments/xyz1/walkforward_results/def456/" /><content type="html">&lt;p&gt;nice&lt;/p&gt; /u/lol</content></entry>
</feed>"""


class RedditRSSTests(unittest.TestCase):
    def test_parse_rss_listing(self):
        posts = reddit.parse_rss_listing(REDDIT_RSS, "algotrading")
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].id, "xyz1")
        self.assertIn("walk-forward optimization", posts[0].text)
        self.assertNotIn("submitted by", posts[0].text)

    def test_parse_rss_comments_skips_post_and_short(self):
        comments = reddit.parse_rss_comments(REDDIT_THREAD_RSS, "xyz1")
        self.assertEqual(len(comments), 1)
        self.assertIn("halve your size", comments[0][1])

    def test_fetch_posts_falls_back_to_rss_on_403(self):
        from bigbrain.net import HTTPStatusError

        with mock.patch.object(reddit, "http_json", side_effect=HTTPStatusError(403, "u", {}, b"")), \
             mock.patch.object(reddit, "http_text", side_effect=[REDDIT_RSS, REDDIT_THREAD_RSS]):
            posts = reddit.fetch_posts("algotrading")
        self.assertEqual(len(posts), 1)
        self.assertEqual(len(posts[0].comments), 1)
        brain = Brain()
        seed(brain)
        titles = reddit.learn_posts(brain, posts)
        cell = brain.find(titles[0])[0]
        self.assertEqual(cell.kind, "discussion")
        self.assertIn("walk-forward", cell.concepts)
        self.assertIn("Top comments:", cell.content)
        self.assertNotIn("points", cell.content.split("Top comments:")[1].splitlines()[1])


class GitHubTests(unittest.TestCase):
    def test_clean_markdown(self):
        text = github.clean_markdown(README)
        self.assertNotIn("shields.io", text)
        self.assertNotIn("import backtester", text)
        self.assertIn("slippage and commission models", text)

    def test_learn_repos(self):
        brain = Brain()
        seed(brain)
        with mock.patch.object(github, "http_json", return_value=SEARCH), mock.patch.object(github, "http_get", return_value=README.encode()):
            repos = github.search("topic:backtesting")
            titles = github.learn_repos(brain, repos)
        cell = brain.find("someone/backtester")[0]
        self.assertEqual(cell.kind, "code")
        self.assertIn("backtesting", cell.concepts)
        self.assertIn("open source", cell.concepts)
        self.assertIn("4200 stars", cell.content)
        neighbors = [c.title for c, _ in brain.neighbors(cell.id, limit=30)]
        self.assertIn("Backtesting", neighbors)


class WebTests(unittest.TestCase):
    def test_extract_drops_boilerplate(self):
        page = web.extract("https://example.com/tf", HTML)
        self.assertEqual(page.title, "Why Trend Following Works & When It Fails")
        self.assertIn("positive skew", page.text)
        self.assertNotIn("Home", page.text)
        self.assertNotIn("var x", page.text)
        self.assertNotIn("Copyright", page.text)
        self.assertEqual(page.description, "A look at trend following across regimes.")

    def test_learn_url(self):
        brain = Brain()
        seed(brain)
        with mock.patch.object(web, "http_text", return_value=HTML):
            title, links = web.learn_url(brain, "https://example.com/tf")
        self.assertGreater(links, 0)
        cell = brain.find(title)[0]
        self.assertIn("trend", cell.concepts)
        self.assertIn("stop loss", cell.concepts)

    def test_learn_url_rejects_empty_pages(self):
        with mock.patch.object(web, "http_text", return_value="<html><body><script>app()</script></body></html>"):
            with self.assertRaises(ValueError):
                web.learn_url(Brain(), "https://example.com/spa")

    def test_parse_and_learn_feed(self):
        items = web.parse_feed(RSS)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].link, "https://example.com/vol")
        self.assertIn("Scaling position size", items[0].summary)
        brain = Brain()
        seed(brain)
        with mock.patch.object(web, "http_text", return_value=RSS):
            titles = web.learn_feed(brain, "https://example.com/feed")
        self.assertEqual(titles, ["Volatility targeting explained"])
        self.assertIn("volatility", brain.find(titles[0])[0].concepts)


class SourcesTests(unittest.TestCase):
    def test_run_all_keeps_going_when_a_source_fails(self):
        from bigbrain import sources
        from bigbrain.ingest import Item
        brain = Brain()
        with mock.patch("bigbrain.ingest.papers.fetch", side_effect=RuntimeError("arxiv down")), \
             mock.patch("bigbrain.ingest.reddit.fetch_posts", return_value=[]), \
             mock.patch("bigbrain.ingest.github.fetch_repos", return_value=[]), \
             mock.patch("bigbrain.ingest.web.feed_items", return_value=[Item("article", "Vol targeting", "Scaling position size inversely with volatility keeps risk constant. " * 3, "https://x")]):
            report = sources.run_all(brain, quick=True, workers=4)
        self.assertGreaterEqual(report["curriculum"][0], 51)
        self.assertEqual(report["papers"][0], 0)
        self.assertIn("arxiv down", report["papers"][1])
        self.assertEqual(report["reddit"], (0, None))
        self.assertEqual(report["feeds"][0], 3)
        self.assertEqual(len(brain.find("Vol targeting")), 1, "identical items from several feeds merge into one cell")

    def test_build_jobs_merges_pack_sources(self):
        from bigbrain import sources
        with mock.patch("bigbrain.knowledge.merged_sources", return_value={"feeds": ["https://example.com/feed"], "subreddits": ["algotrading", "newsub"], "github_queries": [], "arxiv_topics": ["momentum"], "urls": ["https://example.com/ref"]}):
            jobs = sources.build_jobs(quick=False)
        names = [(j.group, j.name) for j in jobs]
        self.assertIn(("feeds", "https://example.com/feed"), names)
        self.assertIn(("reddit", "r/newsub"), names)
        self.assertIn(("pages", "https://example.com/ref"), names)
        self.assertEqual(sum(1 for g, n in names if g == "papers" and n == "momentum"), 1)
        self.assertEqual(sum(1 for g, n in names if g == "reddit" and n == "r/algotrading"), 1)


if __name__ == "__main__":
    unittest.main()
