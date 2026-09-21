import math
import unittest

from bigbrain import Brain, lab
from bigbrain.ingest.market import Bar, synthetic


def bar(i, o, h, l, c):
    return Bar(f"D{i:04d}", o, h, l, c, 1000.0, 1000.0 * c)


class SimulatorTests(unittest.TestCase):
    def test_market_orders_fill_at_the_next_open_with_costs(self):
        bars = [bar(0, 100, 101, 99, 100), bar(1, 102, 103, 101, 102.5), bar(2, 104, 105, 103, 104), bar(3, 106, 107, 105, 106)]
        costs = lab.Costs(fee=0.001, spread_bps=10.0)
        # decide long after bar 0, flat after bar 1: buy at bar 1's open, sell at bar 2's open
        decisions = [{"target": 1}, {"target": 0}, {"target": 0}, {"target": 0}]
        r = lab.simulate(bars, decisions, costs, "15m")
        self.assertEqual(r.trades, 1)
        t = r.log[0]
        self.assertAlmostEqual(t.entry, 102 * (1 + 0.0005))  # lifted the ask (half of 10 bp)
        self.assertAlmostEqual(t.exit, 104 * (1 - 0.0005))  # hit the bid
        self.assertAlmostEqual(t.ret, t.exit / t.entry - 1 - 0.001 - 0.001 * t.exit / t.entry)
        self.assertEqual((t.entry_i, t.exit_i, t.reason), (1, 2, "signal"))

    def test_stops_fill_at_the_level_or_the_open_on_a_gap(self):
        costs = lab.Costs(fee=0.0, spread_bps=0.0)
        bars = [bar(0, 100, 101, 99, 100), bar(1, 100, 101, 99, 100), bar(2, 100, 100.5, 96, 97)]
        r = lab.simulate(bars, [{"target": 1}, {"target": 1, "stop": 98.0}, {"target": 1}], costs, "15m")
        self.assertEqual(r.log[0].reason, "stop")
        self.assertAlmostEqual(r.log[0].exit, 98.0)
        bars = [bar(0, 100, 101, 99, 100), bar(1, 100, 101, 99, 100), bar(2, 95, 96, 94, 95)]  # opens below the stop
        r = lab.simulate(bars, [{"target": 1}, {"target": 1, "stop": 98.0}, {"target": 1}], costs, "15m")
        self.assertAlmostEqual(r.log[0].exit, 95.0)
        self.assertAlmostEqual(r.log[0].ret, -0.05)

    def test_stop_and_reverse_flips_at_the_level(self):
        costs = lab.Costs(fee=0.0, spread_bps=0.0)
        bars = [bar(0, 100, 101, 99, 100), bar(1, 100, 101, 99, 100), bar(2, 100, 100.5, 96, 97), bar(3, 97, 98, 95, 96)]
        decisions = [{"target": 1}, {"target": 1, "stop": 98.0, "reverse": True}, {"target": -1, "stop": 99.5, "reverse": True}, {"target": -1}]
        r = lab.simulate(bars, decisions, costs, "15m")
        self.assertEqual([t.side for t in r.log], [1, -1])
        self.assertAlmostEqual(r.log[0].exit, 98.0)
        self.assertAlmostEqual(r.log[1].entry, 98.0)  # the short opened where the long stopped
        self.assertEqual(r.log[1].reason, "end")
        self.assertAlmostEqual(r.log[1].ret, -(96.0 / 98.0 - 1))

    def test_metrics(self):
        trades = [lab.Trade(1, 0, 1, 1, 1, 0.02, "signal"), lab.Trade(1, 2, 1, 3, 1, -0.01, "stop"), lab.Trade(1, 4, 1, 5, 1, 0.03, "signal"), lab.Trade(1, 6, 1, 7, 1, -0.02, "stop")]
        r = lab.summarize(trades, [1.0, 1.02, 1.0098, 1.04, 1.02], "1h")
        self.assertEqual((r.trades, r.wins), (4, 2))
        self.assertAlmostEqual(r.profit_factor, 0.05 / 0.03)
        self.assertAlmostEqual(r.expectancy, 0.005)
        self.assertLess(r.max_drawdown, 0)
        self.assertGreater(r.tstat, 0)
        self.assertEqual(lab.summarize([], [], "1h").trades, 0)


class RuleTests(unittest.TestCase):
    def test_parabolic_sar_tracks_and_flips(self):
        rising = [bar(i, 100 + i, 101 + i, 99.5 + i, 100.8 + i) for i in range(30)]
        series = lab.psar_series(rising)
        for i in range(3, 30):
            sar, up, nxt = series[i]
            self.assertTrue(up)
            self.assertLessEqual(sar, rising[i - 1].low)  # never above the previous two lows
            self.assertGreater(nxt, sar)  # accelerating toward price
        # a sharp reversal flips the trend and puts the SAR above the highs
        falling = rising + [bar(30 + i, 130 - 3 * i, 131 - 3 * i, 128 - 3 * i, 128.5 - 3 * i) for i in range(10)]
        series = lab.psar_series(falling)
        self.assertFalse(series[-1][1])
        self.assertGreaterEqual(series[-1][0], falling[-2].high)
        decisions = lab.psar_rule(falling, lab.RULES["psar"]["defaults"])
        self.assertEqual(decisions[-1]["target"], -1)
        self.assertTrue(decisions[-1]["reverse"])
        self.assertEqual(decisions[0]["target"], 0)

    def test_other_rules_produce_one_decision_per_candle(self):
        bars = synthetic("X", n=300, seed=3)
        for name, spec in lab.RULES.items():
            d = spec["fn"](bars, spec["defaults"])
            self.assertEqual(len(d), len(bars), name)
            self.assertTrue(all(x["target"] in (-1, 0, 1) for x in d), name)


class GridTests(unittest.TestCase):
    def test_parse_and_plateau(self):
        grid = lab.parse_grid("start=0.03,0.01,0.02;maximum=0.1,0.2", {"start": 0.02, "increment": 0.02, "maximum": 0.2})
        self.assertEqual(grid, {"start": [0.01, 0.02, 0.03], "increment": [0.02], "maximum": [0.1, 0.2]})
        self.assertEqual(len(lab.grid_points(grid)), 6)
        with self.assertRaises(ValueError):
            lab.parse_grid("nope=1", {"start": 0.02})
        # a spike: one setting with a great profit factor whose neighbours all lose
        rows = []
        for p in lab.grid_points(grid):
            r = lab.Result(trades=50, profit_factor=3.0 if p == {"start": 0.02, "increment": 0.02, "maximum": 0.1} else 0.8)
            rows.append((p, r))
        table = lab.plateau(rows, grid)
        spike = next(e for e in table if e["params"]["start"] == 0.02 and e["params"]["maximum"] == 0.1)
        self.assertEqual(spike["n_neighbours"], 3)
        self.assertAlmostEqual(spike["neighbours"], 0.8)
        self.assertAlmostEqual(spike["plateau"], 0.8)  # the spike does not survive the plateau test

    def test_grid_search_runs_every_point(self):
        bars = synthetic("X", n=400, seed=5)
        grid = lab.parse_grid("fast=5,10;slow=30,60", lab.RULES["sma_cross"]["defaults"])
        rows = lab.grid_search("sma_cross", bars, grid, lab.Costs(), "15m", workers=2)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(isinstance(r, lab.Result) for _, r in rows))


class WalkForwardTests(unittest.TestCase):
    def test_parameters_are_chosen_on_the_past_only(self):
        # a synthetic rule with a knob: knob=2 is right in the early windows, knob=1 only in the final one.
        # An honest walk-forward must pick knob=2 for the final window (it cannot see that window) and lose there.
        n = 600

        def knob_rule(bars, params):
            k = int(params["knob"])
            out = []
            for i in range(len(bars)):
                late = i >= 500
                right = (k == 1) == late
                out.append({"target": (1 if right else -1) if i % 2 == 0 else 0, "stop": None, "reverse": False})  # a fresh trade every other candle
            return out

        lab.RULES["knob_test"] = {"fn": knob_rule, "defaults": {"knob": 1}, "doc": "test"}
        try:
            bars = [bar(i, 100 + i * 0.1, 100.2 + i * 0.1, 99.9 + i * 0.1, 100.1 + i * 0.1) for i in range(n)]  # steadily rising
            grid = {"knob": [1, 2]}
            wf = lab.walk_forward("knob_test", bars, grid, lab.Costs(fee=0.0, spread_bps=0.0), "15m", folds=6, workers=1)
            self.assertEqual(len(wf["steps"]), 5)
            for s in wf["steps"][:-1]:
                self.assertEqual(s["params"], {"knob": 2})
            last = wf["steps"][-1]
            self.assertEqual(last["params"], {"knob": 2})  # chosen on the past ...
            self.assertLess(last["out_of_sample"]["expectancy"], 0)  # ... and wrong out of sample, as it should be
            self.assertTrue(all(t.entry_i >= 100 for t in wf["oos_trades"]))  # never a trade from a training window
        finally:
            del lab.RULES["knob_test"]

    def test_verdicts(self):
        self.assertEqual(lab.verdict({"trades": 10, "profit_factor": 5.0, "tstat": 3.0})[0], "insufficient")
        self.assertEqual(lab.verdict({"trades": 300, "profit_factor": 0.9, "tstat": -1.0})[0], "no edge")
        self.assertEqual(lab.verdict({"trades": 300, "profit_factor": 1.1, "tstat": 1.0})[0], "noise")
        self.assertEqual(lab.verdict({"trades": 300, "profit_factor": 1.6, "tstat": 3.0}, {"profit_factor": 2.0})[0], "worth a look")


class EndToEndTests(unittest.TestCase):
    def test_random_walk_gives_no_edge_and_the_brain_remembers(self):
        bars = synthetic("RW", n=3000, seed=11, drift=0.0, vol=0.004)
        grid = lab.parse_grid("start=0.01,0.02;maximum=0.1,0.2", lab.RULES["psar"]["defaults"])
        report = lab.run("psar", bars, grid, lab.Costs(fee=0.0005, spread_bps=1.0), "1m", folds=4, workers=2)
        self.assertEqual(len(report["table"]), 4)
        oos = report["walk_forward"]["out_of_sample"]
        self.assertGreater(oos["trades"], 30)
        self.assertLess(oos["profit_factor"], 1.3)  # a random walk has nothing to catch, and the SAR pays the spread on every flip
        self.assertIn(report["verdict"], ("no edge", "noise"))
        text = lab.format_report(report, "RWUSDT", "1m")
        self.assertIn("walk-forward", text)
        self.assertIn("verdict", text)
        brain = Brain()
        title = lab.learn_result(brain, report, "RWUSDT", "1m", 2)
        cells = [c for c in brain.cells(kind="lesson") if c.title == title]
        self.assertEqual(len(cells), 1)
        self.assertIn("out-of-sample profit factor", cells[0].content)
        lab.learn_result(brain, report, "RWUSDT", "1m", 2)  # rewritten, not duplicated
        self.assertEqual(len([c for c in brain.cells(kind="lesson") if c.title == title]), 1)


if __name__ == "__main__":
    unittest.main()
