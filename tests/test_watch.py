import unittest

from bigbrain import Brain
from bigbrain.ingest.market import Bar, synthetic
from bigbrain.ingest.textbook import seed
from bigbrain.watch import SIGNALS, Watcher, detect, live_context, readings


def make_bars(closes, start=0):
    return [Bar(f"T{start + i:05d}", c, c * 1.001, c * 0.999, c, 1000.0) for i, c in enumerate(closes)]


class DetectTests(unittest.TestCase):
    def test_rsi_oversold_fires_once_on_the_cross(self):
        closes = [100 + i for i in range(40)] + [140 - 3 * i for i in range(1, 15)]
        rs = readings(make_bars(closes))
        fired = [(i, sig) for i in range(1, len(rs)) for sig in detect(rs[i - 1], rs[i]) if sig == "rsi_oversold"]
        self.assertEqual(len(fired), 1, fired)

    def test_band_break_and_macd(self):
        closes = [100.0] * 60 + [90.0]
        rs = readings(make_bars(closes))
        self.assertIn("below_lower_band", detect(rs[-2], rs[-1]))
        self.assertTrue(set(SIGNALS) >= set(detect(rs[-2], rs[-1])))


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.brain = Brain()
        seed(self.brain, packs=False)
        self.bars = synthetic("BTCUSDT", n=400, seed=5)

    def test_tick_records_calls_learns_observations_and_grades(self):
        w = Watcher(self.brain, "btcusdt", "15m", horizon=4, fetch=lambda s, i, n: self.bars[:300])
        # replay history one closed candle at a time; feeding a growing window simulates real time
        total_fired = 0
        for end in range(200, 301):
            r = w.tick(self.bars[:end])
            total_fired += len(r["fired"])
        self.assertGreater(total_fired, 0)
        n_calls = self.brain.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        self.assertEqual(n_calls, total_fired)
        graded = self.brain.db.execute("SELECT COUNT(*) FROM calls WHERE graded_at IS NOT NULL").fetchone()[0]
        self.assertGreater(graded, 0)
        self.assertLess(graded, n_calls + 1)
        obs = self.brain.cells(kind="observation")
        self.assertEqual(len(obs), total_fired)
        self.assertIn("btcusdt", obs[0].concepts)
        self.assertTrue(any(c["hit_rate"] is not None for c in w.scorecard()))
        # the same closed candle never fires twice
        again = w.tick(self.bars[:300])
        self.assertFalse(again["new_bar"])
        self.assertEqual(again["fired"], [])

    def test_scorecard_becomes_a_lesson_after_five_graded_calls(self):
        w = Watcher(self.brain, "BTCUSDT", "15m", horizon=2, fetch=lambda s, i, n: self.bars)
        for end in range(100, 399):
            w.tick(self.bars[:end])
        lessons = [c for c in self.brain.cells(kind="lesson") if "scorecard" in c.title]
        cards = [c for c in w.scorecard() if c["graded"] >= 5]
        self.assertEqual(len(lessons), len(cards))
        if lessons:
            self.assertIn("hit rate", lessons[0].content)
            self.assertIn("walk-forward", lessons[0].concepts)

    def test_live_context_and_next_close(self):
        w = Watcher(self.brain, "BTCUSDT", "15m", fetch=lambda s, i, n: self.bars)
        w.tick(self.bars[:250])
        lines = live_context(self.brain)
        self.assertEqual(len(lines), 1)
        self.assertIn("BTCUSDT 15m", lines[0])
        self.assertAlmostEqual(w.seconds_until_next_close(now=900 * 10 + 100), 800 + 5)

    def test_run_once_logs_and_handles_errors(self):
        logs = []
        w = Watcher(self.brain, "BTCUSDT", "15m", fetch=lambda s, i, n: self.bars)
        w.run(log=logs.append, once=True)
        self.assertIn("BTCUSDT 15m close", logs[0])
        bad = Watcher(self.brain, "BTCUSDT", "15m", fetch=lambda s, i, n: (_ for _ in ()).throw(RuntimeError("down")))
        bad.run(log=logs.append, once=True)
        self.assertIn("watch error: down", logs[1])


if __name__ == "__main__":
    unittest.main()
