import unittest

from bigbrain import Brain, exits


def path_up_then_down(entry=100.0):
    # rises to +3%, then falls back to -3%
    return [(101, 99.5, 100.5), (102, 100.5, 101.5), (103, 101.5, 102.5), (102.5, 100.5, 101), (101, 98.5, 99), (99, 96.5, 97)]


class SimulateTests(unittest.TestCase):
    def test_variants_on_a_giveback_path(self):
        r = exits.simulate(path_up_then_down(), 100.0, 1, 0.01, actual_exit=97.0, costs=0.002)
        self.assertAlmostEqual(r["rule"], -0.03 - 0.002)
        self.assertAlmostEqual(r["tp_1R"], 0.01 - 0.002)  # hit 101 on bar 1
        self.assertAlmostEqual(r["tp_2R"], 0.02 - 0.002)  # hit 102 on bar 2
        self.assertAlmostEqual(r["breakeven_1R"], 0.0 - 0.002)  # armed at 101, stopped at entry on the way down
        self.assertGreater(r["trail_1R"], 0.015)  # trailed one R behind the 103 high -> out near 101.97

    def test_short_side_mirrors(self):
        path = [(100.5, 99, 99.5), (99.5, 98, 98.5), (98.5, 97, 97.5), (100.5, 98, 100)]
        r = exits.simulate(path, 100.0, -1, 0.01, actual_exit=100.0, costs=0.0)
        self.assertAlmostEqual(r["tp_1R"], 0.01)
        self.assertAlmostEqual(r["tp_2R"], 0.02)
        self.assertGreater(r["trail_1R"], 0.0)
        self.assertAlmostEqual(r["rule"], 0.0)

    def test_stop_before_target_in_same_bar(self):
        path = [(102, 98, 100)]
        r = exits.simulate(path, 100.0, 1, 0.01, actual_exit=100.0, costs=0.0)
        self.assertAlmostEqual(r["tp_1R"], -0.01)


class PolicyTests(unittest.TestCase):
    def test_policy_adopts_then_reverts(self):
        brain = Brain()
        for _ in range(exits.MIN_TRADES_FOR_POLICY):
            exits.record(brain, "b", "rsi_oversold", {"rule": -0.005, "tp_1R": 0.004, "tp_2R": -0.002, "breakeven_1R": -0.001, "trail_1R": 0.002})
        self.assertEqual(exits.update_policy(brain, "b", "rsi_oversold"), ("rule", "tp_1R"))
        self.assertEqual(exits.current_policy(brain, "b")["rsi_oversold"], "tp_1R")
        self.assertTrue(any("Exit policy" in c.title for c in brain.cells(kind="lesson")))
        for _ in range(60):
            exits.record(brain, "b", "rsi_oversold", {"rule": 0.01, "tp_1R": -0.01, "tp_2R": 0.0, "breakeven_1R": 0.0, "trail_1R": 0.0})
        self.assertEqual(exits.update_policy(brain, "b", "rsi_oversold"), ("tp_1R", "rule"))
        self.assertEqual(len([c for c in brain.cells(kind="lesson") if "Exit policy" in c.title]), 1)  # rewritten, not duplicated

    def test_no_policy_without_enough_trades(self):
        brain = Brain()
        exits.record(brain, "b", "x", {"rule": -1, "tp_1R": 1, "tp_2R": 1, "breakeven_1R": 1, "trail_1R": 1})
        self.assertIsNone(exits.update_policy(brain, "b", "x"))


class LiveTests(unittest.TestCase):
    def test_trader_applies_learned_policy_and_records_variants(self):
        import json
        from bigbrain.ingest.market import synthetic
        from bigbrain.trader import Trader
        from tests.test_trader import market_at, stamped, universe

        brain = Brain()
        symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        series = {s: stamped(synthetic(s, n=1200, seed=40 + i, vol=0.02)) for i, s in enumerate(symbols)}
        for bars in series.values():
            for b in bars:
                b.quote_volume = b.volume * b.close
        t = Trader(brain, book="ex", interval="15m", wallet=1000.0, top=3, market="perps", fetch_funding=lambda: {})
        for end in range(250, 1200):
            t.tick(market=market_at(series, end), universe=universe(symbols))
        rows = brain.db.execute("SELECT symbol, signal, entry_time, variants, exit_reason FROM trades WHERE book = 'ex'").fetchall()
        self.assertTrue(rows)
        scored = 0
        for r in rows:
            v = json.loads(r["variants"])
            if v:
                self.assertEqual(set(v), set(exits.VARIANTS))
                scored += 1
            else:  # closed by a learned exit: its shadow is still running the original rule, and will score it later
                self.assertIn(f"{r['symbol']}:{r['signal']}:{r['entry_time']}", t.wallet.shadows)
        self.assertGreater(scored, 0)
        self.assertTrue(brain.db.execute("SELECT COUNT(*) FROM exit_stats WHERE book = 'ex'").fetchone()[0] > 0)
        # force a policy and check new positions run under it
        brain.set_state(exits.policy_key("ex"), {"rsi_oversold": "trail_1R", "macd_bullish": "tp_1R", "macd_bearish": "tp_1R", "below_lower_band": "tp_1R", "above_upper_band": "tp_1R", "golden_cross": "tp_1R", "death_cross": "tp_1R", "rsi_overbought": "tp_1R"})
        t2 = Trader(brain, book="ex2", interval="15m", wallet=1000.0, top=3, market="perps", fetch_funding=lambda: {})
        brain.set_state(exits.policy_key("ex2"), brain.get_state(exits.policy_key("ex")))
        reasons = set()
        for end in range(250, 1200):
            for e in t2.tick(market=market_at(series, end), universe=universe(symbols))["events"]:
                if e["action"] in ("sell", "cover"):
                    reasons.add(e["reason"])
        self.assertTrue(reasons & {"target", "trail"}, reasons)


if __name__ == "__main__":
    unittest.main()
