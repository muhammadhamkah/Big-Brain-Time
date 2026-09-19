"""Execution and evidence: the cases an external review of the trader raised, each as a reproduction."""

import json
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from bigbrain import Brain, exits
from bigbrain import postmortem as pm
from bigbrain.ingest.market import Bar, synthetic
from bigbrain.trader import RISK_PER_TRADE, Trader
from tests.test_trader import market_at, stamped, universe

CTX = {"regime": "uptrend", "vol_bucket": "mid", "vol20": 0.5, "atr_pct": 0.01, "rsi": 25.0, "band_pos": 0.1, "macd_hist": 0.0, "hour_utc": 0}
VOL, QV = 50_000_000.0, 1_000_000.0


def bar(date, o, h, l, c):
    return Bar(date, o, h, l, c, 10_000.0, QV)


def trade(book, symbol, entry_time, net_ret, exit_time="2026-09-02 00:00"):
    return pm.Trade(book, symbol, "rsi_oversold", entry_time, 100.0, exit_time, 100.0 * (1 + net_ret), "signal", 1.0, 100.0, net_ret, net_ret, 100 * net_ret,
                    0.1, 0.05, 3, max(net_ret, 0.0), min(net_ret, 0.0), {"regime": "uptrend", "vol_bucket": "mid", "risk_pct": 0.02})


class RecordTests(unittest.TestCase):
    def test_a_repeated_record_never_touches_its_neighbour(self):
        brain = Brain()
        a, b = trade("k", "AAAUSDT", "2026-09-01 00:00", 0.01), trade("k", "BBBUSDT", "2026-09-01 00:00", -0.02)
        self.assertTrue(pm.record(brain, a, [], {"rule": 0.01}))
        self.assertTrue(pm.record(brain, b, [("whipsaw", "x")], {"rule": -0.02}))
        again = trade("k", "AAAUSDT", "2026-09-01 00:00", 0.5)  # same identity, different numbers
        self.assertFalse(pm.record(brain, again, [("clean_win", "x")], {"rule": 0.5}))
        rows = {r["symbol"]: r for r in brain.db.execute("SELECT * FROM trades WHERE book = 'k'")}
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows["BBBUSDT"]["net_ret"], -0.02)
        self.assertEqual(json.loads(rows["BBBUSDT"]["variants"]), {"rule": -0.02})
        self.assertEqual(json.loads(rows["BBBUSDT"]["findings"]), ["whipsaw"])
        self.assertAlmostEqual(rows["AAAUSDT"]["net_ret"], 0.01)
        self.assertEqual(rows["AAAUSDT"]["model_version"], brain.RESULTS_VERSION)


class ReplayTests(unittest.TestCase):
    def test_a_gap_through_the_stop_fills_every_variant_at_the_open(self):
        # long from 100 with the stop at 95; the next candle opens at 80: nothing fills at 95
        r = exits.simulate([(80.0, 81.0, 79.0, 80.0)], 100.0, 1, 0.05, actual_exit=80.0, cost_ratio=0.0)
        for v in exits.VARIANTS:
            self.assertAlmostEqual(r[v], -0.20, msg=v)
        # a short mirrors it: stop at 105, gap open at 120
        r = exits.simulate([(120.0, 121.0, 119.0, 120.0)], 100.0, -1, 0.05, actual_exit=120.0, cost_ratio=0.0)
        for v in exits.VARIANTS:
            self.assertAlmostEqual(r[v], -0.20, msg=v)

    def test_funding_is_charged_to_each_variant_up_to_its_own_exit(self):
        path = [(100.0, 101.0, 99.5, 100.5), (100.5, 102.0, 100.0, 101.5), (101.5, 103.0, 101.0, 102.5), (102.5, 102.6, 97.0, 98.0)]
        fund = [0.0, 0.001, 0.002, 0.003]  # cumulative funding paid, as a fraction of notional, after each bar
        r = exits.simulate(path, 100.0, 1, 0.01, actual_exit=98.0, cost_ratio=0.0, funding_path=fund)
        self.assertAlmostEqual(r["tp_1R"], 0.01)  # out on the first bar, before any funding
        self.assertAlmostEqual(r["tp_2R"], 0.02 - 0.001)  # out on the second bar
        self.assertAlmostEqual(r["rule"], -0.02 - 0.003)  # held to the end, paid all three


class ExecutionTests(unittest.TestCase):
    """A frozen book is used so sizing is deterministic; the execution path is the same for both books."""

    def setUp(self):
        self.brain = Brain()
        self.t = Trader(self.brain, book="x", market="perps", frozen=True, fetch_funding=lambda: {})
        self.signal_bar = bar("2026-09-01 00:00", 100.0, 101.0, 99.0, 100.5)

    def test_live_entries_fill_at_the_book_or_not_at_all(self):
        t = self.t
        now = time.time()
        t.live, t.quotes = True, {
            "AAAUSDT": {"bid": 100.6, "ask": 100.8, "at": now},
            "BBBUSDT": {"bid": 100.6, "ask": 100.8, "at": now},
            "CCCUSDT": {"bid": 100.6, "ask": 100.8, "at": now - 120},  # stale
        }
        ev = t._consider_entry("AAAUSDT", "rsi_oversold", self.signal_bar, CTX, VOL, QV)
        self.assertEqual(ev["action"], "buy")
        self.assertGreaterEqual(ev["price"], 100.8)  # lifts the ask ...
        self.assertLess(ev["price"], 100.8 * 1.0005)  # ... plus impact only; the spread is already in the quote
        pos = t.wallet.positions["AAAUSDT:rsi_oversold"]
        self.assertEqual(pos["fill_basis"], "quote")
        self.assertEqual(pos["entry_time"], self.signal_bar.date)
        self.assertEqual(pos["last_funding_hour"], "2026-09-01 00")  # funding at 00:00 happened before this fill
        ev = t._consider_entry("BBBUSDT", "rsi_overbought", self.signal_bar, CTX, VOL, QV)
        self.assertEqual(ev["action"], "short")
        self.assertLessEqual(ev["price"], 100.6)  # hits the bid
        for symbol in ("CCCUSDT", "DDDUSDT"):  # stale, missing
            ev = t._consider_entry(symbol, "rsi_oversold", self.signal_bar, CTX, VOL, QV)
            self.assertEqual(ev["action"], "skip")
            self.assertIn("quote", ev["why"])
        self.assertEqual(t.wallet.pending, {})  # live never queues an entry
        self.assertEqual(set(t.wallet.positions), {"AAAUSDT:rsi_oversold", "BBBUSDT:rsi_overbought"})

    def test_replay_entries_queue_and_fill_at_the_next_open(self):
        t = self.t
        ev = t._consider_entry("AAAUSDT", "rsi_oversold", self.signal_bar, CTX, VOL, QV)
        self.assertEqual(ev["action"], "queued")
        self.assertEqual(t.wallet.positions, {})
        self.assertIn("AAAUSDT:rsi_oversold", t.wallet.pending)
        # the same signal on the same candle does not queue twice, and the opposite side is refused while the order rests
        self.assertIsNone(t._consider_entry("AAAUSDT", "rsi_oversold", self.signal_bar, CTX, VOL, QV))
        self.assertIn("conflict", t._consider_entry("AAAUSDT", "rsi_overbought", self.signal_bar, CTX, VOL, QV)["why"])
        nxt = bar("2026-09-01 00:15", 103.0, 104.0, 102.5, 103.5)  # opens well above the signal close
        events = t._fill_pending("AAAUSDT", nxt, VOL, QV, catch_up=False)
        self.assertEqual(events[0]["action"], "buy")
        self.assertGreater(events[0]["price"], 103.0)  # the open plus slippage, never the signal candle's close
        pos = t.wallet.positions["AAAUSDT:rsi_oversold"]
        self.assertEqual((pos["entry_time"], pos["fill_basis"], pos["entered_at_open"]), (nxt.date, "next_open", True))
        self.assertEqual(pos["context"]["decided"], self.signal_bar.date)
        self.assertEqual(t.wallet.pending, {})
        # an open-filled position is managed on its own candle: a stop touched there fires there
        crash = bar("2026-09-01 00:30", 103.5, 103.6, 90.0, 91.0)
        t.wallet.positions["AAAUSDT:rsi_oversold"] = pos
        events = t._manage_exits("AAAUSDT", crash, SimpleNamespace(rsi=None, sma20=None, close=91.0), [], VOL, QV)
        self.assertEqual(events[0]["reason"], "stop")
        self.assertAlmostEqual(events[0]["price"], pos["stop"], delta=pos["stop"] * 0.001)  # the stop level less slippage, not the close

    def test_rule_exits_queue_in_replay_and_hit_the_bid_live(self):
        t = self.t
        t._consider_entry("AAAUSDT", "rsi_oversold", self.signal_bar, CTX, VOL, QV)
        fill_bar = bar("2026-09-01 00:15", 100.0, 100.5, 99.8, 100.2)
        t._fill_pending("AAAUSDT", fill_bar, VOL, QV, catch_up=False)
        recovered = SimpleNamespace(rsi=60.0, sma20=None, close=100.4)
        signal_exit = bar("2026-09-01 00:30", 100.2, 100.6, 100.0, 100.4)
        self.assertEqual(t._manage_exits("AAAUSDT", signal_exit, recovered, [], VOL, QV), [])  # decided, not filled
        self.assertEqual(t.wallet.positions["AAAUSDT:rsi_oversold"]["pending_exit"], "signal")
        nxt = bar("2026-09-01 00:45", 99.0, 99.5, 98.5, 99.2)
        events = t._fill_pending("AAAUSDT", nxt, VOL, QV, catch_up=False)
        self.assertEqual((events[0]["action"], events[0]["reason"], events[0]["basis"]), ("sell", "signal", "next_open"))
        self.assertLess(events[0]["price"], 99.0)  # the next open, less slippage
        row = self.brain.db.execute("SELECT exit_time, context FROM trades WHERE book = 'x'").fetchone()
        self.assertEqual(row["exit_time"], nxt.date)
        self.assertEqual(json.loads(row["context"])["exit_basis"], "next_open")
        # live: the same decision fills at once, at the bid
        t.live, t.quotes = True, {"BBBUSDT": {"bid": 100.3, "ask": 100.5, "at": time.time()}}
        t._consider_entry("BBBUSDT", "rsi_oversold", self.signal_bar, CTX, VOL, QV)
        events = t._manage_exits("BBBUSDT", signal_exit, recovered, [], VOL, QV)
        self.assertEqual((events[0]["action"], events[0]["basis"]), ("sell", "quote"))
        self.assertLessEqual(events[0]["price"], 100.3)
        self.assertNotIn("BBBUSDT:rsi_oversold", t.wallet.positions)

    def test_a_cancelled_order_never_spends_money_it_no_longer_has(self):
        t = self.t
        t._consider_entry("AAAUSDT", "rsi_oversold", self.signal_bar, CTX, VOL, QV)
        t.wallet.cash = 1.0  # the money went elsewhere before the candle opened
        events = t._fill_pending("AAAUSDT", bar("2026-09-01 00:15", 100.0, 100.5, 99.8, 100.2), VOL, QV, catch_up=False)
        self.assertEqual(events[0]["action"], "skip")
        self.assertEqual(t.wallet.positions, {})
        self.assertAlmostEqual(t.wallet.cash, 1.0)


class ShadowTests(unittest.TestCase):
    def test_a_variant_exit_still_scores_the_rule_on_the_full_path_with_funding(self):
        brain = Brain()
        t = Trader(brain, book="sh", market="perps", fetch_funding=lambda: {})
        key = "AAAUSDT:rsi_oversold"
        t.wallet.pending[key] = {"symbol": "AAAUSDT", "signal": "rsi_oversold", "side": 1, "notional": 100.0, "stop_pct": 0.02, "explore": False, "variant": "tp_1R",
                                 "max_bars": 16, "exit_rule": "rsi_recovered", "placed": "2026-09-01 07:00", "signal_close": 100.0, "p_win": 0.5, "samples": 0, "ctx": CTX}
        quiet = SimpleNamespace(rsi=40.0, sma20=None, close=100.0)
        t._process_bar("AAAUSDT", bar("2026-09-01 07:15", 100.0, 100.5, 99.8, 100.2), quiet, [], VOL, QV)
        pos = t.wallet.positions[key]
        entry = pos["entry_price"]
        events = t._process_bar("AAAUSDT", bar("2026-09-01 07:30", 100.2, 103.0, 100.0, 102.8), quiet, [], VOL, QV)  # +1R target hit
        self.assertEqual(events[0]["reason"], "target")
        self.assertNotIn(key, t.wallet.positions)
        shadow_key = f"AAAUSDT:rsi_oversold:{pos['entry_time']}"
        self.assertIn(shadow_key, t.wallet.shadows)
        self.assertEqual(len(t.wallet.shadows[shadow_key]["path"]), 2)  # the two candles the position saw, once each
        row = brain.db.execute("SELECT variants FROM trades WHERE book = 'sh'").fetchone()
        self.assertEqual(json.loads(row["variants"]), {})  # not scored yet: the rule has not exited
        self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM exit_stats WHERE book = 'sh'").fetchone()[0], 0)
        t.funding = {"AAAUSDT": {"rate": 0.001, "next": 0}}  # longs pay 10bp at 08:00
        t._process_bar("AAAUSDT", bar("2026-09-01 07:45", 102.8, 104.0, 102.0, 103.5), quiet, [], VOL, QV)
        t._process_bar("AAAUSDT", bar("2026-09-01 08:00", 103.5, 104.5, 103.0, 104.0), SimpleNamespace(rsi=60.0, sma20=None, close=104.0), [], VOL, QV)  # rule fires
        self.assertTrue(t.wallet.shadows[shadow_key]["pending"])  # replay: the rule's order waits for the next open
        t._process_bar("AAAUSDT", bar("2026-09-01 08:15", 105.0, 105.5, 104.5, 105.2), quiet, [], VOL, QV)
        self.assertNotIn(shadow_key, t.wallet.shadows)
        v = json.loads(brain.db.execute("SELECT variants FROM trades WHERE book = 'sh'").fetchone()["variants"])
        self.assertEqual(set(v), set(exits.VARIANTS))
        rule_gross = 105.0 / entry - 1
        self.assertLess(v["rule"], rule_gross)  # costs and the 08:00 funding came off
        self.assertLess(v["rule"], rule_gross - 0.001 * 104.0 / entry + 1e-9)  # funding of 10bp of the 08:00 mark
        self.assertAlmostEqual(v["tp_1R"], 0.02 - (rule_gross - v["rule"] - 0.001 * 104.0 / entry), places=6)  # same costs, no funding: out before 08:00
        stats = exits.stats(brain, "sh", "rsi_oversold")
        self.assertEqual({k: n for k, (n, _) in stats.items()}, {v_: 1 for v_ in exits.VARIANTS})


class EvidenceTests(unittest.TestCase):
    def test_full_size_needs_independent_days(self):
        brain = Brain()
        for i in range(25):  # 25 winners, all closed on one day: one observation, however many trades
            tr = trade("ev", f"S{i}USDT", f"2026-09-01 {i:02d}:00", 0.01, exit_time="2026-09-01 23:00")
            pm.record(brain, tr, []); pm.update_belief(brain, tr)
        b = pm.belief(brain, "ev", "rsi_oversold", "uptrend", "mid")
        self.assertEqual(b["blocks"], 1)
        self.assertEqual(pm.verdict(b), "half")
        for i in range(25):  # the same winners spread over five days
            tr = trade("ev2", f"S{i}USDT", f"2026-09-01 {i:02d}:00", 0.01, exit_time=f"2026-09-{1 + i % 5:02d} 23:00")
            pm.record(brain, tr, []); pm.update_belief(brain, tr)
        b = pm.belief(brain, "ev2", "rsi_oversold", "uptrend", "mid")
        self.assertEqual(b["blocks"], 5)
        self.assertEqual(pm.verdict(b), "full")
        for i in range(12):  # losers on one day are not yet an avoid
            tr = trade("ev3", f"S{i}USDT", f"2026-09-01 {i:02d}:00", -0.02, exit_time="2026-09-01 23:00")
            pm.record(brain, tr, []); pm.update_belief(brain, tr)
        self.assertEqual(pm.verdict(pm.belief(brain, "ev3", "rsi_oversold", "uptrend", "mid")), "explore")
        for i in range(12):  # over five days they are
            tr = trade("ev4", f"S{i}USDT", f"2026-09-01 {i:02d}:00", -0.02, exit_time=f"2026-09-{1 + i % 5:02d} 23:00")
            pm.record(brain, tr, []); pm.update_belief(brain, tr)
        self.assertEqual(pm.verdict(pm.belief(brain, "ev4", "rsi_oversold", "uptrend", "mid")), "avoid")

    def test_old_results_versions_do_not_count_as_evidence(self):
        brain = Brain()
        for i in range(25):
            tr = trade("old", f"S{i}USDT", f"2026-09-01 {i:02d}:00", 0.01, exit_time=f"2026-09-{1 + i % 5:02d} 23:00")
            pm.record(brain, tr, [], {"rule": 0.01, "tp_1R": 0.02}); pm.update_belief(brain, tr)
        brain.db.execute("UPDATE trades SET model_version = 1 WHERE book = 'old'")  # recorded under the previous execution model
        b = pm.belief(brain, "old", "rsi_oversold", "uptrend", "mid")
        self.assertEqual(b["blocks"], 0)
        self.assertNotEqual(pm.verdict(b), "full")
        t = Trader(brain, book="old", market="perps", fetch_funding=lambda: {})
        t.rebuild_stats()
        self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM exit_stats WHERE book = 'old'").fetchone()[0], 0)


class FrozenTests(unittest.TestCase):
    def test_frozen_book_ignores_beliefs_and_policies(self):
        brain = Brain()
        for i in range(12):
            tr = trade("fz", f"S{i}USDT", f"2026-09-01 {i:02d}:00", -0.02, exit_time=f"2026-09-{1 + i % 6:02d} 23:00")
            pm.record(brain, tr, []); pm.update_belief(brain, tr)
        brain.set_state(exits.policy_key("fz"), {"rsi_oversold": "tp_1R"})
        adaptive = Trader(brain, book="fz", market="perps", fetch_funding=lambda: {})
        frozen = Trader(brain, book="fz-frozen", market="perps", frozen=True, fetch_funding=lambda: {})
        b = bar("2026-09-10 00:00", 100.0, 101.0, 99.0, 100.5)
        skips = [adaptive._consider_entry("AAAUSDT", "rsi_oversold", b, CTX, VOL, QV) for _ in range(1)]
        self.assertTrue(skips[0] is None or skips[0]["action"] in ("skip", "queued"))  # the adaptive book acts on the evidence (skip, or a rare re-test)
        ev = frozen._consider_entry("AAAUSDT", "rsi_oversold", b, CTX, VOL, QV)
        self.assertEqual(ev["action"], "queued")
        order = frozen.wallet.pending["AAAUSDT:rsi_oversold"]
        self.assertEqual(order["variant"], "rule")
        self.assertFalse(order["explore"])
        self.assertAlmostEqual(order["notional"], min(1000.0 * RISK_PER_TRADE / order["stop_pct"], 250.0))  # full fixed size
        frozen._save()
        self.assertTrue(Trader(brain, book="fz-frozen").frozen)  # the mode is part of the book

    def test_companion_books_tick_on_identical_inputs(self):
        brain = Brain()
        symbols = ["AAAUSDT", "BBBUSDT"]
        series = {s: stamped(synthetic(s, n=400, seed=60 + i, vol=0.02)) for i, s in enumerate(symbols)}
        for bars in series.values():
            for b in bars:
                b.quote_volume = b.volume * b.close
        seen = {"universe": 0, "bars": 0, "quotes": 0}

        def fetch_universe(n):
            seen["universe"] += 1
            return universe(symbols)

        def fetch_bars(symbol, interval, lookback):
            seen["bars"] += 1
            return series[symbol][:400]

        def fetch_quotes():
            seen["quotes"] += 1
            return {s: {"bid": series[s][-1].close * 0.999, "ask": series[s][-1].close * 1.001, "at": time.time()} for s in symbols}

        main = Trader(brain, book="pair", market="perps", top=2, fetch_universe=fetch_universe, fetch_bars=fetch_bars, fetch_quotes=fetch_quotes, fetch_funding=lambda: {})
        frozen = Trader(brain, book="pair-frozen", market="perps", top=2, frozen=True, fetch_funding=lambda: {})
        lines = []
        main.run(log=lines.append, once=True, companions=(frozen,))
        self.assertEqual((seen["universe"], seen["bars"], seen["quotes"]), (1, 2, 1))  # fetched once, shared
        for t in (main, frozen):
            st = brain.get_state(f"trader:{t.book}")
            self.assertTrue(st["last_tick"])
            self.assertEqual(st["last_bar"], {s: series[s][398].date for s in symbols})
            self.assertFalse(t.lock_path().exists())
        self.assertTrue(any("[pair-frozen]" in line for line in lines))


class AtomicTests(unittest.TestCase):
    def test_a_failed_close_rolls_back_money_and_evidence_together(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            brain = Brain(os.path.join(tmp, "b.db"))
            symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
            series = {s: stamped(synthetic(s, n=700, seed=100 + i, vol=0.025)) for i, s in enumerate(symbols)}
            for bars in series.values():
                for b in bars:
                    b.quote_volume = b.volume * b.close
            t = Trader(brain, book="atom", market="perps", frozen=True, fetch_funding=lambda: {})
            end = 250
            while end < 700 and not t.wallet.positions:
                t.tick(market=market_at(series, end), universe=universe(symbols))
                end += 1
            self.assertTrue(t.wallet.positions)
            boom = mock.patch.object(pm, "learn_postmortem", side_effect=RuntimeError("disk on fire"))
            failed = False
            with boom:
                while end < 700 and not failed:
                    trades_before = brain.db.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
                    try:
                        t.tick(market=market_at(series, end), universe=universe(symbols))
                    except RuntimeError:
                        failed = True
                    end += 1
            self.assertTrue(failed, "some position should have tried to close")
            self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM trades").fetchone()[0], trades_before)  # the record was rolled back ...
            stored = brain.get_state("trader:atom")
            self.assertAlmostEqual(t.wallet.cash, stored["cash"])  # ... and so was the money: memory matches the database
            self.assertEqual(set(t.wallet.positions), set(stored["positions"]))
            # and the next tick closes it for real, once
            t.tick(market=market_at(series, end), universe=universe(symbols))
            self.assertGreaterEqual(brain.db.execute("SELECT COUNT(*) FROM trades").fetchone()[0], trades_before)
            brain.close()


if __name__ == "__main__":
    unittest.main()
