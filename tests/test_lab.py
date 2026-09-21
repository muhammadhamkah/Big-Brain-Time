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
            if spec["portfolio"]:
                continue
            d = spec["fn"](bars, spec["defaults"], {"interval": "15m"})
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

        def knob_rule(bars, params, extras=None):
            k = int(params["knob"])
            out = []
            for i in range(len(bars)):
                late = i >= 500
                right = (k == 1) == late
                out.append({"target": (1 if right else -1) if i % 2 == 0 else 0, "stop": None, "reverse": False})  # a fresh trade every other candle
            return out

        lab.RULES["knob_test"] = {"fn": knob_rule, "defaults": {"knob": 1}, "doc": "test", "portfolio": False, "needs": (), "hint": ""}
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
        # skewed returns: a weak per-trade t but a strong account path (Sharpe 1 over 5 years) also counts
        self.assertEqual(lab.verdict({"trades": 600, "profit_factor": 1.4, "tstat": 1.2, "tstat_ts": 2.2})[0], "worth a look")
        self.assertEqual(lab.verdict({"trades": 600, "profit_factor": 1.4, "tstat": 1.2, "tstat_ts": 1.5})[0], "noise")


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


class HypothesisTests(unittest.TestCase):
    """The rules built for the brain's own research programme."""

    def test_playbook_rule_manages_a_signal_like_the_trader(self):
        from tests.test_trader import stamped
        bars = stamped(synthetic("X", n=900, seed=21, vol=0.02))
        for b in bars:
            b.quote_volume = b.volume * b.close
        decisions = lab.playbook_rule(bars, {**lab.RULES["playbook"]["defaults"], "signal": "rsi_oversold"})
        self.assertEqual(len(decisions), len(bars))
        held = [d for d in decisions if d["target"]]
        self.assertTrue(held, "the synthetic series should trigger RSI oversold at least once")
        self.assertTrue(all(d["target"] == 1 and d["stop"] is not None for d in held))  # long only, always with a stop
        shorts = lab.playbook_rule(bars, {**lab.RULES["playbook"]["defaults"], "signal": "macd_bearish", "max_bars": 48})
        self.assertTrue(any(d["target"] == -1 for d in shorts))
        # a position never outlives max_bars
        run = 0
        for d in decisions:
            run = run + 1 if d["target"] else 0
            self.assertLessEqual(run, 16 + 1)
        r = lab.evaluate("playbook", bars, {"signal": "rsi_oversold"}, lab.Costs(), "15m")
        self.assertGreater(r.trades, 0)
        self.assertEqual(lab.parse_grid("signal=*", lab.RULES["playbook"]["defaults"])["signal"], list(lab.PLAYBOOK_SIGNALS))

    def test_trend_with_volatility_targeting_sizes_by_volatility(self):
        calm = [bar(i, 100 + i * 0.05, 100.3 + i * 0.05, 99.9 + i * 0.05, 100.1 + i * 0.05) for i in range(200)]
        d = lab.trend_vt_rule(calm, lab.RULES["trend_vt"]["defaults"], {"interval": "1d"})
        self.assertEqual(d[-1]["target"], 1)
        self.assertLessEqual(d[-1]["weight"], 1.0)
        wild = synthetic("W", n=200, seed=9, vol=0.08)
        dw = lab.trend_vt_rule(wild, lab.RULES["trend_vt"]["defaults"], {"interval": "1d"})
        weights = [x["weight"] for x in dw if x["target"]]
        self.assertTrue(weights and max(weights) < 0.5)  # 8% daily volatility is far above a 40% annual target: the position shrinks
        # weights scale the account, not the trade's own return
        bars = [bar(0, 100, 101, 99, 100), bar(1, 100, 101, 99, 100), bar(2, 110, 111, 109, 110), bar(3, 110, 111, 109, 110)]
        r = lab.simulate(bars, [{"target": 1, "weight": 0.25}, {"target": 0}, {"target": 0}, {"target": 0}], lab.Costs(fee=0, spread_bps=0), "1d")
        self.assertAlmostEqual(r.log[0].ret, 0.10)
        self.assertAlmostEqual(r.net_return, 0.025)

    def test_funding_is_paid_and_the_carry_rule_collects_it(self):
        costs = lab.Costs(fee=0.0, spread_bps=0.0)
        bars = [bar(i, 100, 100.5, 99.5, 100) for i in range(6)]
        funding = {"D0002": 0.001, "D0004": 0.001}  # 10bp paid at two candles' opens
        r = lab.simulate(bars, [{"target": 1}] * 6, costs, "4h", funding=funding)  # a long held throughout pays both
        self.assertAlmostEqual(r.log[0].ret, -0.002)
        self.assertAlmostEqual(r.log[0].funding, 0.002)
        r = lab.simulate(bars, [{"target": -1}] * 6, costs, "4h", funding=funding)  # a short receives them
        self.assertAlmostEqual(r.log[0].ret, +0.002)
        # the carry rule shorts when annualized funding runs hot, and stands down when it normalizes
        hot = {f"D{i:04d}": 0.0005 for i in range(0, 12)}  # 5bp per 8h = 55% a year
        cold = {f"D{i:04d}": 0.00002 for i in range(12, 24)}
        bars = [bar(i, 100, 100.5, 99.5, 100) for i in range(24)]
        d = lab.funding_carry_rule(bars, lab.RULES["funding_carry"]["defaults"], {"funding": {**hot, **cold}})
        self.assertEqual(d[5]["target"], -1)
        self.assertEqual(d[-1]["target"], 0)
        self.assertIsNotNone(d[5]["stop"])
        self.assertGreater(d[5]["stop"], 100)  # a short's stop sits above

    def test_cross_sectional_momentum_is_market_neutral_and_pooled(self):
        markets = {}
        for j in range(10):  # ten symbols with different drifts: the ranking must find the strongest and weakest
            drift = (j - 4.5) * 0.004
            markets[f"S{j}USDT"] = [bar(i, 100 * (1 + drift) ** i, 100 * (1 + drift) ** i * 1.001, 100 * (1 + drift) ** i * 0.999, 100 * (1 + drift) ** (i + 1)) for i in range(120)]
        for bars in markets.values():
            for i, b in enumerate(bars):
                b.date = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d} 00:00"
        params = {"lookback": 30, "skip": 1, "hold": 7, "top": 0.2}
        by_date = lab.xs_momentum_rule(markets, params, {})
        last = markets["S9USDT"][-1].date
        longs = [s for s in markets if by_date[s][last]["target"] == 1]
        shorts = [s for s in markets if by_date[s][last]["target"] == -1]
        self.assertEqual((sorted(longs), sorted(shorts)), (["S8USDT", "S9USDT"], ["S0USDT", "S1USDT"]))
        self.assertAlmostEqual(sum(by_date[s][last].get("weight", 0) for s in longs + shorts), 1.0)
        r = lab.evaluate("xs_momentum", markets, params, lab.Costs(fee=0.0005, spread_bps=2.0), "1d")
        self.assertEqual(r.symbols, 10)
        self.assertGreaterEqual(r.trades, 4)  # constant drifts never change the ranking, so the four legs are held to the end
        self.assertGreater(r.net_return, 0)  # persistent drifts are what momentum is built for
        self.assertTrue(all(t.symbol for t in r.log))

    def test_a_universe_is_walked_forward_by_date(self):
        markets = {}
        for j in range(4):
            bars = synthetic(f"S{j}", n=400, seed=30 + j, vol=0.02)
            for i, b in enumerate(bars):
                b.date = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d} {j:02d}:00"  # symbols close at different minutes; windows are still cut by date
            markets[f"S{j}USDT"] = bars
        grid = lab.parse_grid("fast=5,10;slow=30,60", lab.RULES["sma_cross"]["defaults"])
        wf = lab.walk_forward("sma_cross", markets, grid, lab.Costs(), "1d", folds=4, workers=2)
        self.assertEqual(len(wf["steps"]), 3)
        for step in wf["steps"]:
            for t in step["out_of_sample"] and wf["oos_trades"]:
                self.assertGreaterEqual(t.date, wf["steps"][0]["train_to"])  # never a trade from before the first boundary
        boundaries = [s["train_to"] for s in wf["steps"]]
        self.assertEqual(boundaries, sorted(boundaries))
        pooled = lab.summarize_pool(wf["oos_trades"], 4)
        self.assertEqual(pooled.trades, wf["out_of_sample"]["trades"])
        report = lab.run("sma_cross", markets, grid, lab.Costs(), "1d", folds=4, workers=2)
        self.assertEqual(len(report["symbols"]), 4)
        self.assertIn("4 pairs", lab.format_report(report, "4 pairs", "1d"))


class BenchmarkAndDataTests(unittest.TestCase):
    def test_buy_and_hold_benchmark_and_ranges(self):
        markets = {"AUSDT": [bar(i, 100 + i, 101 + i, 99 + i, 100 + i) for i in range(10)], "BUSDT": [bar(i, 100 - i, 101 - i, 99 - i, 100 - i) for i in range(10)]}
        bh = lab.buy_and_hold(markets, "D0005")
        self.assertEqual(bh["symbols"], 2)
        self.assertAlmostEqual(bh["net_return"], ((109 / 105) + (91 / 95)) / 2 - 1)
        self.assertLessEqual(bh["max_drawdown"], 0.0)
        from unittest import mock
        ranked = [{"symbol": f"S{i}USDT", "quote_volume": 100 - i, "last": 1.0} for i in range(80)]
        with mock.patch("bigbrain.ingest.market.top_usdt_perps", return_value=ranked):
            self.assertEqual(lab.resolve_symbols("top:3", "perps"), ["S0USDT", "S1USDT", "S2USDT"])
            self.assertEqual(lab.resolve_symbols("top:31-33", "perps"), ["S30USDT", "S31USDT", "S32USDT"])
        self.assertEqual(lab.resolve_symbols("btcusdt, ethusdt", "spot"), ["BTCUSDT", "ETHUSDT"])

    def test_rate_limits_are_retried(self):
        from unittest import mock
        from bigbrain.net import HTTPStatusError
        calls = {"n": 0}

        def flaky(url, headers=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise HTTPStatusError(429, url, {"retry-after": "0"}, b"")
            return b"[1, 2, 3]"

        with mock.patch("bigbrain.net.http_get", side_effect=flaky):
            self.assertEqual(lab._get_json("https://x/klines"), [1, 2, 3])
        self.assertEqual(calls["n"], 3)
        with mock.patch("bigbrain.net.http_get", side_effect=HTTPStatusError(404, "u", {}, b"")):
            with self.assertRaises(HTTPStatusError):
                lab._get_json("https://x/klines")


class HonestyTests(unittest.TestCase):
    def test_a_loss_beyond_the_account_is_a_liquidation_not_a_debt(self):
        costs = lab.Costs(fee=0.0, spread_bps=0.0)
        bars = [bar(0, 100, 101, 99, 100), bar(1, 100, 101, 99, 100), bar(2, 400, 401, 399, 400), bar(3, 400, 401, 399, 400)]
        r = lab.simulate(bars, [{"target": -1}, {"target": -1}, {"target": 0}, {"target": 0}], costs, "1d")  # a short that quadrupled against us
        self.assertAlmostEqual(r.log[0].ret, -3.0)
        self.assertAlmostEqual(r.net_return, -1.0)  # wiped out, not -300%
        self.assertGreaterEqual(min([1.0] + [1 + x for x in [r.net_return]]), 0.0)
        pooled = lab.summarize_pool([lab.Trade(-1, 0, 100, 2, 400, -3.0, "signal", 0.5, "D0002")], 1)
        self.assertAlmostEqual(pooled.net_return, -1.0)

    def test_momentum_legs_carry_stops(self):
        markets = {f"S{j}USDT": [bar(i, 100 + j + i * 0.1 * j, 101 + j + i * 0.1 * j, 99 + j + i * 0.1 * j, 100 + j + i * 0.1 * j) for i in range(60)] for j in range(6)}
        for bars in markets.values():
            for i, b in enumerate(bars):
                b.date = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d} 00:00"
        by_date = lab.xs_momentum_rule(markets, {**lab.RULES["xs_momentum"]["defaults"], "lookback": 10, "hold": 5}, {})
        last = markets["S5USDT"][-1].date
        legs = [by_date[s][last] for s in markets if by_date[s][last]["target"]]
        self.assertTrue(legs)
        for d in legs:
            self.assertIsNotNone(d["stop"])
            self.assertEqual(d["stop"] > 0, True)

    def test_universe_is_ranked_by_volume_at_the_start(self):
        def series(vol_early, vol_late, n=100, first=0):
            out = []
            for i in range(first, n):
                b = bar(i, 100, 101, 99, 100)
                b.date = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d} 00:00"
                b.quote_volume = vol_early if i < 30 else vol_late
                out.append(b)
            return out

        markets = {"OLDBIG": series(1e9, 1e6), "NOWBIG": series(1e6, 1e9), "MID": series(5e8, 5e8), "LATE": series(1e9, 1e9, first=40)}
        kept = lab.rank_at_start(markets, 2)
        self.assertEqual(set(kept), {"OLDBIG", "MID"})  # today's winner and the late listing are out: neither was knowable at the start


if __name__ == "__main__":
    unittest.main()
