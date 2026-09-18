import unittest

from bigbrain import Brain
from bigbrain.ingest.market import synthetic
from bigbrain.ingest.strategies import STRATEGIES, backtest
from bigbrain.ingest.textbook import seed
from bigbrain.paper import STARTING_EQUITY, PaperTrader
from bigbrain.watch import Watcher


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.brain = Brain()
        self.bars = synthetic("BTCUSDT", n=400, seed=9)

    def test_replay_matches_backtester(self):
        """Stepping candle by candle must reproduce the vectorized backtest (same fills, same costs)."""
        pt = PaperTrader(self.brain, "BTCUSDT", "15m")
        for end in range(1, len(self.bars) + 1):
            pt.step(self.bars[:end])
        report = {r["strategy"]: r for r in pt.report(self.bars[-1].close)}
        for name, (rule, _) in STRATEGIES.items():
            bt = backtest(name, "BTCUSDT", self.bars, rule)
            self.assertAlmostEqual(report[name]["return"], bt.total_return, delta=0.02, msg=name)
            closed = report[name]["trades"] + (1 if report[name]["in_position"] else 0)
            self.assertEqual(closed, bt.trades, msg=name)
        self.assertEqual(report["buy and hold"]["trades"], 0)  # still open, never closed
        self.assertTrue(report["buy and hold"]["in_position"])

    def test_step_is_idempotent_per_candle(self):
        pt = PaperTrader(self.brain, "BTCUSDT", "15m")
        pt.step(self.bars[:100])
        first = pt.report(self.bars[99].close)
        pt.step(self.bars[:100])
        self.assertEqual(pt.report(self.bars[99].close), first)
        n = self.brain.db.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        self.assertEqual(n, sum(1 for r in first if r["in_position"]))

    def test_lessons_after_ten_trades(self):
        seed(self.brain, packs=False)
        bars = synthetic("BTCUSDT", n=1500, seed=2, vol=0.03)
        pt = PaperTrader(self.brain, "BTCUSDT", "15m")
        for end in range(60, len(bars) + 1):
            pt.step(bars[:end])
        closed = self.brain.db.execute("SELECT strategy, COUNT(*) FROM paper_trades WHERE exit_time IS NOT NULL GROUP BY strategy").fetchall()
        lessons = [c.title for c in self.brain.cells(kind="lesson") if "paper trading" in c.title]
        expected = [s for s, n in closed if n >= 10]
        self.assertTrue(expected, "synthetic series should produce ten closed trades for at least one strategy")
        for s in expected:
            self.assertIn(f"BTCUSDT 15m: paper trading {s}", lessons)
        self.assertEqual(len(lessons), len(expected))  # rewritten in place, never duplicated

    def test_watcher_drives_paper_trading(self):
        w = Watcher(self.brain, "BTCUSDT", "15m", fetch=lambda s, i, n: self.bars)
        events = []
        for end in range(200, 300):
            events += w.tick(self.bars[:end])["trades"]
        self.assertTrue(events)
        self.assertTrue(all(e["action"] in ("buy", "sell") for e in events))
        self.assertEqual(len(PaperTrader(self.brain, "BTCUSDT", "15m").report()), len(STRATEGIES))


if __name__ == "__main__":
    unittest.main()
