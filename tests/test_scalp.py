import io
import random
import unittest
from contextlib import redirect_stdout

from bigbrain import Brain, lab, scalp
from bigbrain.cli import main
from bigbrain.ingest.market import Bar


def bar(i, o, h, l, c, v=1000.0):
    return Bar(f"D{i:04d}", o, h, l, c, v, v * c)


def flat_bars(n, price=100.0, v=1000.0):
    return [bar(i, price, price * 1.001, price * 0.999, price, v) for i in range(n)]


def entries(decisions):
    """Each distinct bracket the rule asked for: a new side or new levels."""
    count, last = 0, None
    for d in decisions:
        key = (d["target"], d.get("stop"), d.get("take")) if d["target"] else None
        if key and key != last:
            count += 1
        last = key
    return count


class HelperTests(unittest.TestCase):
    def test_rolling_extreme_matches_the_naive_window(self):
        vals = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 3]
        got = scalp._rolling_extreme(vals, 3, min)
        self.assertEqual(got, [None] * 3 + [min(vals[i - 3:i]) for i in range(3, len(vals))])
        got = scalp._rolling_extreme(vals, 3, max)
        self.assertEqual(got, [None] * 3 + [max(vals[i - 3:i]) for i in range(3, len(vals))])

    def test_rolling_rank(self):
        vals = [None, 1.0, 2.0, 3.0, 0.5, 2.5, 4.0]
        got = scalp._rolling_rank(vals, 3)
        self.assertEqual(got[:4], [None] * 4)  # index 3 has only two known values before it
        self.assertEqual(got[4], 0.0)  # 0.5 below 1, 2, 3
        self.assertAlmostEqual(got[5], 2 / 3)  # 2.5 among 2, 3, 0.5
        self.assertEqual(got[6], 1.0)

    def test_vwap_z_matches_the_naive_computation(self):
        bars = scalp.random_walk(n=200, seed=3)
        w = 20
        vwap, z = scalp.rolling_vwap_z(bars, w)
        i = 150
        win = bars[i - w + 1:i + 1]
        expect = sum((b.high + b.low + b.close) / 3 * b.volume for b in win) / sum(b.volume for b in win)
        self.assertAlmostEqual(vwap[i], expect)
        devs = []
        for j in range(i - w + 1, i + 1):
            devs.append(bars[j].close - vwap[j])
        mean = sum(devs) / w
        sd = (sum((d - mean) ** 2 for d in devs) / w) ** 0.5
        self.assertAlmostEqual(z[i], devs[-1] / sd, places=6)


class SetupTests(unittest.TestCase):
    def test_sweep_reversal_enters_long_with_stop_past_the_wick(self):
        bars = flat_bars(60)
        bars.append(bar(60, 100.0, 100.05, 99.0, 99.95))  # runs the lows at 99.9, closes back near the top
        bars += [bar(61 + k, 99.95 + 0.2 * k, 100.2 + 0.2 * k, 99.9 + 0.2 * k, 100.15 + 0.2 * k) for k in range(10)]
        d = lab.RULES["sweep"]["fn"](bars, {**lab.RULES["sweep"]["defaults"], "lookback": 20}, {})
        self.assertEqual(d[59]["target"], 0)
        self.assertEqual(d[60]["target"], 1)
        self.assertLess(d[60]["stop"], 99.0)
        risk = 99.95 - d[60]["stop"]
        self.assertAlmostEqual(d[60]["take"], 99.95 + 1.5 * risk)
        r = lab.simulate(bars, d, lab.Costs(fee=0, spread_bps=0), "1m")
        self.assertEqual(r.log[0].reason, "target")
        self.assertGreater(r.log[0].ret, 0)

    def test_side_and_trend_filters(self):
        bars = flat_bars(60)
        bars.append(bar(60, 100.0, 100.05, 99.0, 99.95))
        bars += flat_bars(5)
        d = lab.RULES["sweep"]["fn"](bars, {**lab.RULES["sweep"]["defaults"], "lookback": 20, "side": -1}, {})
        self.assertTrue(all(x["target"] != 1 for x in d))

    def test_vwap_fade_buys_a_stretch_that_turns_and_targets_the_vwap(self):
        bars = []
        for i in range(80):  # wiggle around 100 so the deviations have a spread
            c = 100 + (0.05 if i % 2 else -0.05)
            bars.append(bar(i, c, c + 0.05, c - 0.05, c))
        bars.append(bar(80, 100.0, 100.0, 99.0, 99.1, 3000))  # a hard drop
        bars.append(bar(81, 99.1, 99.4, 99.05, 99.3))  # it turns
        bars += [bar(82 + k, 99.3, 100.2, 99.2, 100.0) for k in range(3)]
        d = lab.RULES["vwap_fade"]["fn"](bars, {**lab.RULES["vwap_fade"]["defaults"], "window": 30, "entry_z": 2.0}, {})
        self.assertEqual(d[80]["target"], 0)  # still falling: no knife catching
        self.assertEqual(d[81]["target"], 1)
        self.assertGreater(d[81]["take"], 99.3)
        self.assertLess(d[81]["stop"], 99.3)

    def test_squeeze_breaks_out_with_volume(self):
        bars = []
        for i in range(160):
            amp = 2.0 if i < 100 else 0.05  # wide, then a tight squeeze
            c = 100 + (amp if i % 2 else -amp)
            bars.append(bar(i, c, c + amp / 2, c - amp / 2, c))
        bars.append(bar(160, 100.0, 101.6, 99.9, 101.5, 5000))
        bars += flat_bars(3, 101.5)
        d = lab.RULES["squeeze"]["fn"](bars, lab.RULES["squeeze"]["defaults"], {})
        self.assertEqual(d[160]["target"], 1)
        quiet = bars[:160] + [bar(160, 100.0, 101.6, 99.9, 101.5, 1000)] + flat_bars(3, 101.5)
        self.assertEqual(lab.RULES["squeeze"]["fn"](quiet, lab.RULES["squeeze"]["defaults"], {})[160]["target"], 0)

    def test_the_rule_and_the_simulator_agree_on_every_position(self):
        bars = scalp.random_walk(n=3000, seed=5)
        free = lab.Costs(fee=0, spread_bps=0)
        for name in scalp.SCALP_RULES:
            for max_bars in (0, 5):  # with and without time exits
                d = lab.RULES[name]["fn"](bars, {**lab.RULES[name]["defaults"], "max_bars": max_bars}, {})
                r = lab.simulate(bars, d, free, "1m")
                self.assertGreater(r.trades, 5, name)
                self.assertEqual(r.trades, entries(d), name)  # every bracket asked for is one trade; none overlap or vanish
                self.assertTrue({t.reason for t in r.log} <= {"stop", "target", "signal", "end"}, name)
                if max_bars:
                    self.assertTrue(all(t.exit_i - t.entry_i <= max_bars for t in r.log), name)

    def test_random_entries_earn_nothing(self):
        """The guard against look-ahead: on a fair random walk, brackets placed at random and every setup average zero."""
        free = lab.Costs(fee=0, spread_bps=0)
        trades = {"random": [], **{n: [] for n in scalp.SCALP_RULES}}
        for seed in range(3):
            bars = scalp.random_walk(n=6000, seed=seed)
            atr, rng = scalp._atr(bars), random.Random(seed)

            def entry(i):
                if atr[i] and rng.random() < 0.03:
                    side = rng.choice((1, -1))
                    return side, bars[i].close - side * 2 * atr[i], bars[i].close + side * 6 * atr[i]

            trades["random"] += lab.simulate(bars, scalp.bracketed(bars, entry, 30), free, "1m").log
            for name in scalp.SCALP_RULES:
                trades[name] += lab.simulate(bars, lab.RULES[name]["fn"](bars, lab.RULES[name]["defaults"], {}), free, "1m").log
        for name, ts in trades.items():
            r = lab.summarize(ts, [1.0], "1m")
            self.assertGreater(r.trades, 100, name)
            self.assertLess(abs(r.tstat), 3.0, f"{name}: {r.expectancy:+.5f} per trade, t={r.tstat:.1f} on a walk with no edge")



class CostTests(unittest.TestCase):
    def test_cost_edge_splits_gross_and_cost(self):
        bars = scalp.random_walk(n=2000, seed=9)
        d = lab.RULES["sweep"]["fn"](bars, lab.RULES["sweep"]["defaults"], {})
        free = lab.simulate(bars, d, lab.Costs(fee=0, spread_bps=0), "1m")
        costs = lab.Costs(fee=0.0005, spread_bps=2.0, slippage_bps=0.5)
        paid = lab.simulate(bars, d, costs, "1m")
        e0, e1 = scalp.cost_edge(free.log, lab.Costs(fee=0, spread_bps=0)), scalp.cost_edge(paid.log, costs)
        self.assertAlmostEqual(e0["cost_bps"], 0.0, places=9)
        self.assertAlmostEqual(e0["gross_bps"], e1["gross_bps"], delta=0.5)  # same moves, measured between quotes
        self.assertAlmostEqual(e1["cost_bps"], 5 + 5 + 2 + 1, delta=0.5)  # two fees, the spread, slippage twice
        self.assertAlmostEqual(e1["net_bps"], e1["gross_bps"] - e1["cost_bps"])
        self.assertEqual(scalp.cost_edge([], costs)["gross_bps"], 0.0)


class ScanTests(unittest.TestCase):
    def test_scan_on_random_walks_finds_nothing(self):
        markets = {f"S{j}": scalp.random_walk(f"S{j}", n=1500, seed=20 + j) for j in range(2)}
        rows = scalp.scan({"1m": markets}, lab.Costs(fee=0.0005, spread_bps=1.0), workers=2)
        self.assertEqual({r["rule"] for r in rows}, set(scalp.SCALP_RULES))
        self.assertTrue(all(r["verdict"] != "worth a look" for r in rows))
        text = scalp.format_scan(rows, lab.Costs(fee=0.0005, spread_bps=1.0), "test")
        self.assertIn("round trip", text)
        brain = Brain(":memory:")
        titles = scalp.learn_scan(brain, rows, "test", 5)
        scalp.learn_scan(brain, rows, "test", 5)  # learning twice replaces, it does not pile up
        self.assertEqual(len(titles), 3)
        hits = brain.db.execute("SELECT COUNT(*) FROM cells WHERE title LIKE 'Scalp cost check:%'").fetchone()[0]
        self.assertEqual(hits, 3)

    def test_a_lucky_gross_edge_is_not_announced(self):
        costs = lab.Costs(fee=0.0005, spread_bps=1.0)

        def row(name, gross_t):
            return {"rule": name, "interval": "1m", "verdict": "no edge", "params": {}, "per_day": 0.0, "oos": {"trades": 200, "tstat": -1.0},
                    "edge": {"gross_bps": 4.0, "cost_bps": 12.0, "net_bps": -8.0, "breakeven_bps": 4.0, "gross_win": 0.5, "gross_t": gross_t}}

        rows = [row(f"r{k}", 0.5) for k in range(5)] + [row("lucky", 2.3)]  # the best of six clears 2.0 by chance often
        self.assertIn("noise", scalp.format_scan(rows, costs, "x"))
        rows[-1]["edge"]["gross_t"] = 4.0
        self.assertIn("strongest edge before costs was lucky", scalp.format_scan(rows, costs, "x"))

    def test_spot_scans_long_only(self):
        markets = {"S": scalp.random_walk("S", n=1200, seed=4)}
        rows = scalp.scan({"1m": markets}, lab.Costs(), rules=("sweep",), long_only=True, workers=1)
        self.assertTrue(all(t.side == 1 for t in rows[0]["report"]["walk_forward"]["oos_trades"]))

    def test_cli_synthetic(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--db", ":memory:", "scalp", "--synthetic", "--intervals", "1m", "--candles", "1200", "--workers", "1"])
        self.assertEqual(code, 0)
        self.assertIn("scalp scan", out.getvalue())


if __name__ == "__main__":
    unittest.main()
