import unittest

from bigbrain import Brain
from bigbrain.cortex import answer_offline
from bigbrain.ingest import indicators as ind
from bigbrain.ingest.market import learn_bars, load_csv, snapshot, synthetic
from bigbrain.ingest.papers import learn_papers, parse_atom
from bigbrain.ingest.strategies import STRATEGIES, backtest, learn_backtests, sma_crossover
from bigbrain.ingest.textbook import seed


class IndicatorTests(unittest.TestCase):
    def test_sma(self):
        self.assertEqual(ind.sma([1, 2, 3, 4, 5], 3), [None, None, 2.0, 3.0, 4.0])

    def test_ema_seeds_with_sma(self):
        out = ind.ema([1, 2, 3, 4, 5], 3)
        self.assertEqual(out[:3], [None, None, 2.0])
        self.assertAlmostEqual(out[3], 3.0)

    def test_rsi_extremes(self):
        rising = list(range(1, 30))
        self.assertEqual(ind.rsi(rising)[-1], 100.0)
        falling = list(range(30, 1, -1))
        self.assertEqual(ind.rsi(falling)[-1], 0.0)

    def test_bollinger_brackets_price(self):
        closes = [100 + (i % 5) for i in range(40)]
        upper, mid, lower = ind.bollinger(closes)
        self.assertLess(lower[-1], mid[-1])
        self.assertGreater(upper[-1], mid[-1])

    def test_macd_lengths(self):
        closes = [100 + i * 0.5 for i in range(60)]
        line, sig, hist = ind.macd(closes)
        self.assertEqual(len(line), len(sig))
        self.assertEqual(len(hist), 60)
        self.assertGreater(hist[-1] if hist[-1] is not None else 0, -1)

    def test_max_drawdown(self):
        self.assertAlmostEqual(ind.max_drawdown([1.0, 1.5, 0.75, 1.0]), -0.5)

    def test_sharpe_zero_for_constant(self):
        self.assertEqual(ind.sharpe([0.01] * 10), 0.0)


class MarketTests(unittest.TestCase):
    def test_synthetic_is_deterministic(self):
        a, b = synthetic(n=50, seed=1), synthetic(n=50, seed=1)
        self.assertEqual([x.close for x in a], [x.close for x in b])

    def test_snapshot_and_observations(self):
        bars = synthetic(n=300)
        snap = snapshot("TEST", bars)
        self.assertIsNotNone(snap.rsi14)
        self.assertIsNotNone(snap.sma200)
        obs = snap.observations()
        self.assertTrue(any("RSI" in title for title, _ in obs))

    def test_learn_bars_creates_linked_observations(self):
        brain = Brain()
        seed(brain)
        titles = learn_bars(brain, "TEST", synthetic(n=300))
        self.assertGreaterEqual(len(titles), 4)
        cell = brain.find("TEST: RSI")[0]
        self.assertIn("test", cell.concepts)
        neighbor_titles = [c.title for c, _ in brain.neighbors(cell.id)]
        self.assertIn("RSI (Relative Strength Index)", neighbor_titles)

    def test_load_csv(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.csv")
            with open(path, "w") as fh:
                fh.write("Date,Open,High,Low,Close,Volume\n")
                for i in range(40):
                    fh.write(f"2024-01-{i+1:02d},{100+i},{101+i},{99+i},{100.5+i},1000\n")
            bars = load_csv(path)
            self.assertEqual(len(bars), 40)
            self.assertEqual(bars[0].close, 100.5)

    def test_too_few_bars(self):
        with self.assertRaises(ValueError):
            learn_bars(Brain(), "X", synthetic(n=10))


class StrategyTests(unittest.TestCase):
    def test_backtest_buy_and_hold_matches_price_change(self):
        bars = synthetic(n=120, seed=3)
        result = backtest("buy and hold", "X", bars, STRATEGIES["buy and hold"][0])
        expected = bars[-1].close / bars[0].close - 1.0
        # always long, so the result is the price change minus one 0.1% entry cost
        self.assertAlmostEqual(result.total_return, expected, delta=0.005)
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.exposure, 1.0)

    def test_crossover_positions_are_binary(self):
        closes = [b.close for b in synthetic(n=100)]
        self.assertTrue(set(sma_crossover(closes)) <= {0, 1})

    def test_learn_backtests_creates_lessons(self):
        brain = Brain()
        seed(brain)
        results = learn_backtests(brain, "SYN", synthetic(n=300))
        self.assertEqual(len(results), len(STRATEGIES))
        lessons = brain.cells(kind="lesson")
        self.assertEqual(len(lessons), len(STRATEGIES))
        titles = [c.title for c, _ in brain.neighbors(brain.find("sma crossover on SYN")[0].id)]
        self.assertTrue(any("Trend" in t or "Moving averages" in t for t in titles))


ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Momentum   and Mean Reversion in
      Crypto Markets</title>
    <summary>We study momentum and mean reversion in bitcoin using RSI and find
      transaction costs matter.</summary>
    <author><name>A. Author</name></author>
    <author><name>B. Author</name></author>
  </entry>
</feed>"""


class PaperTests(unittest.TestCase):
    def test_parse_atom(self):
        papers = parse_atom(ATOM_SAMPLE)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Momentum and Mean Reversion in Crypto Markets")
        self.assertEqual(papers[0].id, "2401.00001v1")
        self.assertEqual(papers[0].authors, ["A. Author", "B. Author"])

    def test_learn_papers_links_to_curriculum(self):
        brain = Brain()
        seed(brain)
        titles = learn_papers(brain, parse_atom(ATOM_SAMPLE))
        cell = brain.find(titles[0])[0]
        self.assertEqual(cell.kind, "paper")
        self.assertIn("crypto", cell.concepts)
        neighbors = [c.title for c, _ in brain.neighbors(cell.id, limit=30)]
        self.assertIn("Momentum", neighbors)
        self.assertIn("Mean reversion", neighbors)


class CortexTests(unittest.TestCase):
    def test_offline_answer_mentions_top_cell(self):
        brain = Brain()
        seed(brain)
        recalls = brain.recall("what is the kelly criterion", k=4)
        text = answer_offline(brain, "what is the kelly criterion", recalls)
        self.assertIn("Kelly", text)
        self.assertIn("Connections", text)

    def test_offline_answer_on_empty_brain(self):
        text = answer_offline(Brain(), "anything", [])
        self.assertIn("has not learned", text)


if __name__ == "__main__":
    unittest.main()


class PaperFetchTests(unittest.TestCase):
    def test_fetch_retries_on_406_then_succeeds(self):
        import io, urllib.error
        from unittest import mock
        from bigbrain.ingest import papers

        calls = []

        def fake_urlopen(req, timeout):
            calls.append(req.get_header("Accept"))
            if len(calls) == 1:
                raise urllib.error.HTTPError(req.full_url, 406, "Not Acceptable", {}, io.BytesIO(b""))
            resp = mock.MagicMock()
            resp.__enter__.return_value.read.return_value = ATOM_SAMPLE.encode()
            return resp

        with mock.patch.object(papers.urllib.request, "urlopen", fake_urlopen), mock.patch.object(papers.time, "sleep"):
            result = papers.fetch("momentum", max_results=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(a and "atom" in a for a in calls), "every request must send an Accept header")

    def test_fetch_gives_up_on_404(self):
        import io, urllib.error
        from unittest import mock
        from bigbrain.ingest import papers

        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

        with mock.patch.object(papers.urllib.request, "urlopen", fake_urlopen), mock.patch.object(papers.time, "sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                papers.fetch("momentum")

    def test_fetch_alternates_hosts_and_simplifies_query(self):
        import io, urllib.error
        from unittest import mock
        from bigbrain.ingest import papers

        urls = []

        def fake_urlopen(req, timeout):
            urls.append(req.full_url)
            raise urllib.error.HTTPError(req.full_url, 406, "Not Acceptable", {}, io.BytesIO(b""))

        with mock.patch.object(papers.urllib.request, "urlopen", fake_urlopen), mock.patch.object(papers.time, "sleep"):
            with self.assertRaises(RuntimeError):
                papers.fetch("momentum", retries=3)
        self.assertEqual(len(urls), 4)
        self.assertTrue(urls[0].startswith("https://export.arxiv.org/"))
        self.assertTrue(urls[1].startswith("https://arxiv.org/"))
        self.assertIn("cat%3A", urls[0])
        self.assertNotIn("cat%3A", urls[2], "later attempts drop the category filter")
        self.assertIn("all%3Amomentum", urls[2])
