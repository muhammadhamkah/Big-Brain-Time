import json
import unittest

from bigbrain import Brain
from bigbrain import postmortem as pm
from bigbrain.ingest.market import Bar, parse_top_usdt_pairs, synthetic
from bigbrain.ingest.textbook import seed
from bigbrain.trader import MIN_NOTIONAL, Trader, slippage_bps


def universe(symbols):
    return [{"symbol": s, "quote_volume": 50_000_000.0 / (i + 1), "last": 1.0} for i, s in enumerate(symbols)]


def market_at(series, end):
    return {s: bars[:end] for s, bars in series.items()}


class UniverseTests(unittest.TestCase):
    def test_top_pairs_filters_and_sorts(self):
        tickers = [
            {"symbol": "BTCUSDT", "quoteVolume": "5000000000", "lastPrice": "100000"},
            {"symbol": "ETHUSDT", "quoteVolume": "3000000000", "lastPrice": "4000"},
            {"symbol": "USDCUSDT", "quoteVolume": "9000000000", "lastPrice": "1"},
            {"symbol": "BTCUPUSDT", "quoteVolume": "100000000", "lastPrice": "1"},
            {"symbol": "ETHBTC", "quoteVolume": "100000000", "lastPrice": "0.04"},
            {"symbol": "SOLUSDT", "quoteVolume": "800000000", "lastPrice": "200"},
            {"symbol": "EURUSDT", "quoteVolume": "800000000", "lastPrice": "1.1"},
        ]
        top = parse_top_usdt_pairs(tickers, n=2)
        self.assertEqual([t["symbol"] for t in top], ["BTCUSDT", "ETHUSDT"])
        self.assertEqual([t["symbol"] for t in parse_top_usdt_pairs(tickers, n=10)], ["BTCUSDT", "ETHUSDT", "SOLUSDT"])


class CostTests(unittest.TestCase):
    def test_slippage_widens_for_thin_pairs_and_big_orders(self):
        btc = slippage_bps(250, 10_000e6, 2_000_000)
        thin = slippage_bps(250, 1e6, 2_000)
        self.assertLess(btc, 1.0)
        self.assertGreater(thin, 10.0)
        self.assertGreater(slippage_bps(50_000, 1e6, 2_000), thin)


class BeliefTests(unittest.TestCase):
    def test_prior_then_evidence(self):
        brain = Brain()
        b = pm.belief(brain, "main", "rsi_oversold", "uptrend", "mid")
        self.assertAlmostEqual(b["p_win"], 0.5)
        self.assertEqual(b["samples"], 0)
        for i in range(10):
            t = pm.Trade("main", "X", "rsi_oversold", f"t{i}", 1, f"t{i+1}", 1, "time", 1, 100, -0.01, -0.012, -1.2, 0.2, 0.0, 3, 0.002, -0.015, {"regime": "downtrend", "vol_bucket": "high"})
            pm.update_belief(brain, t)
        b = pm.belief(brain, "main", "rsi_oversold", "downtrend", "high")
        self.assertGreaterEqual(b["samples"], pm.MIN_SAMPLES)
        self.assertLess(b["expectancy"], 0)
        self.assertLess(b["p_win"], 0.3)
        other = pm.belief(brain, "main", "rsi_oversold", "uptrend", "low")  # falls back to the broad record
        self.assertLess(other["p_win"], 0.5)
        self.assertLess(other["samples"], pm.MIN_SAMPLES)


class LensTests(unittest.TestCase):
    def _trade(self, **kw):
        base = dict(book="main", symbol="SOLUSDT", signal="rsi_oversold", entry_time="a", entry_price=100.0, exit_time="b", exit_price=97.0, exit_reason="stop",
                    qty=1, notional=100, gross_ret=-0.03, net_ret=-0.032, pnl=-3.2, fees=0.2, slippage=0.05, bars_held=5, mfe=0.004, mae=-0.031,
                    context={"regime": "downtrend", "vol_bucket": "high", "risk_pct": 0.03, "atr_pct": 0.02, "rsi": 29})
        base.update(kw)
        return pm.Trade(**base)

    def test_loss_findings(self):
        tags = [t for t, _ in pm.lenses(self._trade())]
        self.assertIn("fought_the_trend", tags)
        self.assertIn("stop_inside_noise", tags)
        self.assertIn("shallow_extreme", tags)

    def test_costs_and_giveback(self):
        t = self._trade(exit_reason="time", gross_ret=0.0005, net_ret=-0.0015, mfe=0.04, mae=-0.002, context={"regime": "uptrend", "vol_bucket": "mid", "risk_pct": 0.03})
        tags = [t for t, _ in pm.lenses(t)]
        self.assertIn("gave_back_open_gain", tags)
        self.assertIn("costs_ate_the_edge", tags)

    def test_win_findings(self):
        t = self._trade(exit_reason="signal", exit_price=104, gross_ret=0.04, net_ret=0.038, pnl=3.8, bars_held=2, mfe=0.045, mae=-0.003, context={"regime": "uptrend", "vol_bucket": "mid", "risk_pct": 0.02})
        tags = [t for t, _ in pm.lenses(t)]
        self.assertIn("dip_in_uptrend", tags)
        self.assertIn("fast_resolution", tags)


class TraderTests(unittest.TestCase):
    def setUp(self):
        self.brain = Brain()
        seed(self.brain, packs=False)
        self.symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]
        self.series = {s: synthetic(s, n=900, seed=i + 1, vol=0.02) for i, s in enumerate(self.symbols)}
        for bars in self.series.values():
            for b in bars:
                b.quote_volume = b.volume * b.close

    def test_replay_trades_costs_and_learns(self):
        t = Trader(self.brain, book="test", interval="15m", wallet=1000.0, top=5, market="spot")
        buys = sells = 0
        for end in range(250, 900, 1):
            r = t.tick(market=market_at(self.series, end), universe=universe(self.symbols))
            for e in r["events"]:
                buys += e["action"] == "buy"
                sells += e["action"] == "sell"
            self.assertGreaterEqual(t.wallet.cash, -1e-6)
        self.assertGreater(buys, 5)
        self.assertGreater(sells, 5)
        rep = t.report()
        # equity reconciles with cash plus the marked value of open positions, and drawdown is tracked
        self.assertAlmostEqual(rep["equity"], t.wallet.equity(), places=6)
        self.assertLessEqual(rep["max_drawdown"], 0.0)
        # every closed trade paid fees on both sides and some slippage
        rows = self.brain.db.execute("SELECT fees, slippage, notional, net_ret, gross_ret FROM trades WHERE book = 'test'").fetchall()
        self.assertEqual(len(rows), sells)
        for r in rows:
            self.assertGreater(r["fees"], 0.0019 * r["notional"])
            self.assertGreater(r["slippage"], 0)
            self.assertLess(r["net_ret"], r["gross_ret"])
        # beliefs exist and post-mortem cells were written for instructive trades
        self.assertTrue(t.beliefs())
        pms = self.brain.cells(kind="postmortem")
        self.assertTrue(pms)
        self.assertIn("Finding", pms[0].content)
        self.assertTrue(any("paper trading" in c.concepts for c in pms))
        # the brain can recall why it lost
        losses = [c for c in pms if c.title.split(":")[1].strip().startswith("loss")]
        if losses:
            hits = [r.cell.title for r in self.brain.recall(f"why did the brain lose on {losses[0].title.split(':')[0]}", k=5, reinforce=False)]
            self.assertTrue(any("loss on" in h or "win on" in h for h in hits), hits)

    def test_state_persists_and_ticks_are_idempotent(self):
        t = Trader(self.brain, book="persist", interval="15m", wallet=500.0, top=5, market="spot")
        for end in range(250, 400):
            t.tick(market=market_at(self.series, end), universe=universe(self.symbols))
        before = t.report()
        again = t.tick(market=market_at(self.series, 399), universe=universe(self.symbols))
        self.assertEqual(again["events"], [])
        t2 = Trader(self.brain, book="persist", interval="15m", wallet=999.0, top=5, market="spot")  # wallet arg ignored for an existing book
        self.assertAlmostEqual(t2.wallet.start, 500.0)
        self.assertAlmostEqual(t2.report()["equity"], before["equity"], places=4)

    def test_belief_blocks_losing_context(self):
        t = Trader(self.brain, book="blocked", interval="15m", wallet=1000.0, top=5, market="spot")
        for i in range(12):  # twelve losses spread over six days: enough independent evidence to call the context a loser
            tr = pm.Trade("blocked", "X", "rsi_oversold", f"2026-08-{1 + i:02d} 00:00", 1, f"2026-08-{1 + i // 2:02d} 04:00", 1, "stop", 1, 100, -0.02, -0.022, -2.2, 0.2, 0, 3, 0.001, -0.02, {"regime": "downtrend", "vol_bucket": "mid"})
            pm.record(self.brain, tr, [])
            pm.update_belief(self.brain, tr)
        self.assertEqual(pm.verdict(pm.belief(self.brain, "blocked", "rsi_oversold", "downtrend", "mid")), "avoid")
        skips = []
        for end in range(250, 900):
            for e in t.tick(market=market_at(self.series, end), universe=universe(self.symbols))["events"]:
                if e["action"] == "skip" and e["signal"] == "rsi_oversold":
                    skips.append(e)
        self.assertTrue(skips, "the brain should decline rsi oversold entries in a context it believes loses")
        self.assertIn("expectancy", skips[0]["why"])

    def test_spot_never_shorts(self):
        t = Trader(self.brain, book="spotonly", interval="15m", wallet=1000.0, top=5, market="spot")
        for end in range(250, 700):
            for e in t.tick(market=market_at(self.series, end), universe=universe(self.symbols))["events"]:
                self.assertNotEqual(e["action"], "short")
        self.assertEqual(self.brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'spotonly' AND side = -1").fetchone()[0], 0)

    def test_min_notional_respected(self):
        self.assertEqual(MIN_NOTIONAL, 10.0)
        t = Trader(self.brain, book="tiny", interval="15m", wallet=12.0, top=5, market="spot")
        for end in range(250, 500):
            t.tick(market=market_at(self.series, end), universe=universe(self.symbols))
        rows = self.brain.db.execute("SELECT MIN(notional) FROM trades WHERE book = 'tiny'").fetchone()[0]
        if rows is not None:
            self.assertGreaterEqual(rows, MIN_NOTIONAL - 1e-9)


def stamped(bars, start="2026-09-01 00:00"):
    """Give synthetic bars real 15-minute UTC timestamps so funding times occur."""
    from datetime import datetime, timedelta
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M")
    for i, b in enumerate(bars):
        b.date = (t0 + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M")
    return bars


class PerpsTests(unittest.TestCase):
    def setUp(self):
        self.brain = Brain()
        self.symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        self.series = {s: stamped(synthetic(s, n=900, seed=10 + i, vol=0.02)) for i, s in enumerate(self.symbols)}
        for bars in self.series.values():
            for b in bars:
                b.quote_volume = b.volume * b.close
        self.rates = {s: {"rate": 0.0005, "next": 0} for s in self.symbols}  # longs pay 5bp every 8h

    def test_perps_trade_both_sides_and_pay_funding(self):
        t = Trader(self.brain, book="perps", interval="15m", wallet=1000.0, top=3, market="perps", fetch_funding=lambda: self.rates)
        actions = []
        for end in range(250, 900):
            r = t.tick(market=market_at(self.series, end), universe=universe(self.symbols))
            actions += [e["action"] for e in r["events"]]
            self.assertAlmostEqual(r["equity"], t.wallet.equity(), places=6)
        self.assertIn("short", actions)
        self.assertIn("cover", actions)
        self.assertIn("buy", actions)
        rows = self.brain.db.execute("SELECT side, funding, fees, notional, net_ret, pnl, bars_held FROM trades WHERE book = 'perps'").fetchall()
        self.assertTrue(any(r["side"] == -1 for r in rows))
        # futures taker fee is 0.05% per side
        for r in rows:
            self.assertAlmostEqual(r["fees"] / r["notional"], 0.001, delta=0.0003)
        # with a positive rate, longs held through a funding time paid and shorts received
        long_funded = [r for r in rows if r["side"] == 1 and r["funding"] != 0]
        short_funded = [r for r in rows if r["side"] == -1 and r["funding"] != 0]
        self.assertTrue(long_funded or short_funded, "some trade should have been held through 00:00/08:00/16:00 UTC")
        self.assertTrue(all(r["funding"] > 0 for r in long_funded))
        self.assertTrue(all(r["funding"] < 0 for r in short_funded))
        # post-mortems name the side and the book keeps its market when reopened
        self.assertTrue(any(" short " in c.title for c in self.brain.cells(kind="postmortem")))
        self.assertEqual(Trader(self.brain, book="perps", market="spot").market, "perps")

    def test_short_accounting(self):
        from bigbrain.trader import Position, Wallet
        from dataclasses import asdict
        w = Wallet(cash=900.0, start=1000.0, peak=1000.0)
        pos = Position(symbol="X", signal="rsi_overbought", entry_time="a", entry_price=100.0, qty=1.0, notional=100.0, stop=104.0, max_bars=10,
                       exit_rule="rsi_cooled", context={"risk_pct": 0.04}, explore=False, mark=95.0, side=-1, margin=100.0 / 3)
        w.cash = 1000.0 - pos.margin
        w.positions[pos.key] = asdict(pos)
        self.assertAlmostEqual(w.equity(), 1000.0 + 5.0)  # short is up 5 USDT
        w.positions[pos.key]["mark"] = 103.0
        self.assertAlmostEqual(w.equity(), 1000.0 - 3.0)
        self.assertAlmostEqual(w.open_risk(), 4.0)
        self.assertAlmostEqual(w.gross_notional(), 103.0)

    def test_perps_use_margin_and_cap_open_risk(self):
        from bigbrain.trader import MAX_OPEN_RISK
        t = Trader(self.brain, book="lev", interval="15m", wallet=1000.0, top=3, market="perps", fetch_funding=lambda: {})
        max_open, max_gross = 0, 0.0
        for end in range(250, 900):
            r = t.tick(market=market_at(self.series, end), universe=universe(self.symbols))
            max_open = max(max_open, r["open"])
            max_gross = max(max_gross, t.wallet.gross_notional() / max(r["equity"], 1))
            self.assertLessEqual(t.wallet.open_risk(), MAX_OPEN_RISK * r["equity"] * 1.05)
            self.assertGreaterEqual(t.wallet.cash, -1e-6)
        self.assertLessEqual(max_gross, 3.05)
        rows = self.brain.db.execute("SELECT notional FROM trades WHERE book = 'lev'").fetchall()
        self.assertTrue(rows)
        stored = self.brain.get_state("trader:lev")
        for p in stored["positions"].values():
            self.assertAlmostEqual(p["margin"], p["notional"] / 3.0, places=6)


class CatchUpTests(unittest.TestCase):
    def test_missed_candles_are_replayed_for_open_positions(self):
        brain_a, brain_b = Brain(), Brain()
        symbols = ["AAAUSDT", "BBBUSDT"]
        series = {s: stamped(synthetic(s, n=700, seed=50 + i, vol=0.02)) for i, s in enumerate(symbols)}
        for bars in series.values():
            for b in bars:
                b.quote_volume = b.volume * b.close
        # same book name in two brains: identical decisions. Run A sees every candle;
        # run B has a 40-candle gap after bar 400 (laptop asleep) and must catch up.
        a = Trader(brain_a, book="x", interval="15m", wallet=1000.0, top=2, market="perps", fetch_funding=lambda: {})
        b = Trader(brain_b, book="x", interval="15m", wallet=1000.0, top=2, market="perps", fetch_funding=lambda: {})
        for end in range(250, 401):
            a.tick(market=market_at(series, end), universe=universe(symbols))
            b.tick(market=market_at(series, end), universe=universe(symbols))
        open_before = len(b.wallet.positions)
        for end in range(401, 441):
            a.tick(market=market_at(series, end), universe=universe(symbols))
        events = b.tick(market=market_at(series, 441), universe=universe(symbols))["events"]  # the gap
        a.tick(market=market_at(series, 441), universe=universe(symbols))
        cutoff = series["AAAUSDT"][399].date
        q = "SELECT symbol, signal, entry_time, exit_time, exit_reason FROM trades WHERE entry_time <= ? ORDER BY entry_time, symbol, signal"
        closed_a = brain_a.db.execute(q, (cutoff,)).fetchall()
        closed_b = brain_b.db.execute(q, (cutoff,)).fetchall()
        self.assertTrue(closed_a)
        # positions that were open before the gap must close at the same candle and for the same reason in both runs
        self.assertEqual([tuple(r) for r in closed_a], [tuple(r) for r in closed_b])
        if open_before:
            self.assertTrue(any(e.get("catch_up") for e in events if e["action"] in ("sell", "cover")) or not closed_b)


class IntegrityTests(unittest.TestCase):
    def test_duplicate_trades_are_impossible_and_stats_rebuild(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            brain = Brain(os.path.join(tmp, "b.db"))
            symbols = ["AAAUSDT", "BBBUSDT"]
            series = {s: stamped(synthetic(s, n=600, seed=70 + i, vol=0.02)) for i, s in enumerate(symbols)}
            for bars in series.values():
                for b in bars:
                    b.quote_volume = b.volume * b.close
            t = Trader(brain, book="i", interval="15m", wallet=1000.0, top=2, market="perps", fetch_funding=lambda: {})
            for end in range(250, 600):
                t.tick(market=market_at(series, end), universe=universe(symbols))
            n = brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'i'").fetchone()[0]
            self.assertGreater(n, 0)
            beliefs_before = sorted(tuple(r) for r in brain.db.execute("SELECT signal, regime, vol_bucket, wins, losses FROM beliefs WHERE book = 'i'"))
            # a second recording of an existing trade is ignored
            row = brain.db.execute("SELECT * FROM trades WHERE book = 'i' LIMIT 1").fetchone()
            tr = pm.Trade(book="i", symbol=row["symbol"], signal=row["signal"], entry_time=row["entry_time"], entry_price=1, exit_time=row["exit_time"], exit_price=1,
                          exit_reason="stop", qty=1, notional=1, gross_ret=0, net_ret=0, pnl=0, fees=0, slippage=0, bars_held=1, mfe=0, mae=0)
            pm.record(brain, tr, [])
            self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'i'").fetchone()[0], n)
            # corrupt the beliefs, then rebuild from the log
            brain.db.execute("UPDATE beliefs SET wins = wins * 2, losses = losses * 2 WHERE book = 'i'")
            t.rebuild_stats()
            beliefs_after = sorted(tuple(r) for r in brain.db.execute("SELECT signal, regime, vol_bucket, wins, losses FROM beliefs WHERE book = 'i'"))
            self.assertEqual(beliefs_before, beliefs_after)
            # a pre-existing duplicate row is removed when the brain opens
            brain.db.execute("DROP INDEX trades_unique")
            brain.db.execute("INSERT INTO trades SELECT NULL, book, symbol, signal, entry_time, entry_price, exit_time, exit_price, exit_reason, qty, notional, gross_ret, net_ret, pnl, fees, slippage, bars_held, mfe, mae, context, findings, explore, side, funding, variants, model_version FROM trades WHERE id = ?", (row["id"],))
            brain.db.commit()
            self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'i'").fetchone()[0], n + 1)
            brain.close()
            reopened = Brain(os.path.join(tmp, "b.db"))
            self.assertEqual(reopened.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'i'").fetchone()[0], n)
            reopened.close()

    def test_other_trader_processes_are_detected(self):
        from unittest import mock
        from bigbrain import trader as tr
        import os
        fake = (f"{os.getpid()} python -m bigbrain.cli trade\n{os.getppid()} /usr/bin/caffeinate -s /x/.venv/bin/python -m bigbrain.cli trade\n"
                f"4242 /Users/x/.venv/bin/python -m bigbrain.cli trade --book main\n4243 /Users/x/.venv/bin/bigbrain dashboard\n4244 grep bigbrain trade\n"
                f"4245 /usr/bin/caffeinate -s /x/.venv/bin/python -m bigbrain.cli trade\n4246 /bin/sh -c bigbrain trade\n")
        brain = Brain()
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=fake)):
            others = Trader(brain, book="main", market="spot").other_traders()
            self.assertEqual([pid for pid, _ in others], [4242])
            # a trader on another book is not a conflict: that is how a frozen baseline runs beside the adaptive book
            self.assertEqual(Trader(brain, book="main-frozen", market="spot").other_traders(), [])
        t = Trader(brain, book="p", market="spot")
        with mock.patch.object(Trader, "other_traders", return_value=[(4242, "python -m bigbrain.cli trade")]):
            with self.assertRaises(RuntimeError):
                t.acquire_lock()

    def test_lock_refuses_a_second_trader(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            brain = Brain(os.path.join(tmp, "b.db"))
            a = Trader(brain, book="lock", market="spot")
            a.lock_path().parent.mkdir(parents=True, exist_ok=True)
            a.lock_path().write_text(str(os.getppid()))  # another live process holds the book
            with self.assertRaises(RuntimeError):
                a.acquire_lock()
            a.lock_path().unlink()
            a.acquire_lock()  # free again, and re-entrant for the same process
            a.acquire_lock()
            a.release_lock()
            self.assertFalse(a.lock_path().exists())
            # a stale lock from a dead process is ignored
            a.lock_path().write_text("999999")
            a.acquire_lock()
            a.release_lock()


class StopFloorTests(unittest.TestCase):
    def test_stops_never_sit_inside_the_noise(self):
        from bigbrain.trader import MIN_STOP_PCT
        brain = Brain()
        symbols = ["QUIETUSDT", "BBBUSDT"]
        series = {s: stamped(synthetic(s, n=500, seed=90 + i, vol=0.0005 if s == "QUIETUSDT" else 0.02)) for i, s in enumerate(symbols)}
        for bars in series.values():
            for b in bars:
                b.quote_volume = b.volume * b.close
        t = Trader(brain, book="floor", interval="15m", wallet=1000.0, top=2, market="perps", fetch_funding=lambda: {})
        for end in range(250, 500):
            t.tick(market=market_at(series, end), universe=universe(symbols))
        for p in t.wallet.positions.values():
            self.assertGreaterEqual(p["context"]["risk_pct"], MIN_STOP_PCT - 1e-9, p["symbol"])
        for r in brain.db.execute("SELECT context FROM trades WHERE book = 'floor'"):
            self.assertGreaterEqual(json.loads(r["context"])["risk_pct"], MIN_STOP_PCT - 1e-9)


class ConflictTests(unittest.TestCase):
    def test_never_long_and_short_the_same_pair(self):
        brain = Brain()
        symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        series = {s: stamped(synthetic(s, n=900, seed=80 + i, vol=0.025)) for i, s in enumerate(symbols)}
        for bars in series.values():
            for b in bars:
                b.quote_volume = b.volume * b.close
        t = Trader(brain, book="c", interval="15m", wallet=1000.0, top=3, market="perps", fetch_funding=lambda: {})
        conflicts = 0
        for end in range(250, 900):
            r = t.tick(market=market_at(series, end), universe=universe(symbols))
            conflicts += sum(1 for e in r["events"] if e["action"] == "skip" and "conflict" in e["why"])
            sides = {}
            for p in t.wallet.positions.values():
                sides.setdefault(p["symbol"], set()).add(p["side"])
            for sym, ss in sides.items():
                self.assertEqual(len(ss), 1, f"{sym} is both long and short")
        self.assertGreater(conflicts, 0, "the synthetic run should have produced at least one refused hedge")


class ResetTests(unittest.TestCase):
    def test_reset_keeps_or_forgets_learning(self):
        brain = Brain()
        symbols = ["AAAUSDT", "BBBUSDT"]
        series = {s: stamped(synthetic(s, n=600, seed=30 + i, vol=0.02)) for i, s in enumerate(symbols)}
        t = Trader(brain, book="r", interval="15m", wallet=1000.0, top=2, market="perps", fetch_funding=lambda: {})
        for end in range(250, 600):
            t.tick(market=market_at(series, end), universe=universe(symbols))
        trades_before = brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'r'").fetchone()[0]
        self.assertGreater(trades_before, 0)
        r = t.reset(wallet=500.0)
        self.assertEqual(t.wallet.positions, {})
        self.assertAlmostEqual(t.wallet.cash, 500.0)
        self.assertAlmostEqual(t.wallet.equity(), 500.0)
        self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'r'").fetchone()[0], trades_before)
        reopened = Trader(brain, book="r")
        self.assertEqual(reopened.wallet.positions, {})
        self.assertAlmostEqual(reopened.wallet.start, 500.0)
        from bigbrain import exits
        self.assertGreater(brain.db.execute("SELECT COUNT(*) FROM exit_stats WHERE book = 'r'").fetchone()[0], 0)
        brain.set_state(exits.policy_key("r"), {"rsi_oversold": "tp_1R"})
        reopened.reset(forget=True)
        self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM trades WHERE book = 'r'").fetchone()[0], 0)
        self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM beliefs WHERE book = 'r'").fetchone()[0], 0)
        self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM exit_stats WHERE book = 'r'").fetchone()[0], 0)
        self.assertEqual(exits.current_policy(brain, "r"), {})
        self.assertEqual([c for c in brain.cells(kind="postmortem") if c.source == "trade:r"], [])


class DashboardTests(unittest.TestCase):
    def test_render_reflects_live_prices_and_state(self):
        from bigbrain import dashboard
        brain = Brain()
        self.assertIn("No trading book", dashboard.render(brain, "none"))
        symbols = ["AAAUSDT", "BBBUSDT"]
        series = {s: stamped(synthetic(s, n=500, seed=20 + i, vol=0.02)) for i, s in enumerate(symbols)}
        t = Trader(brain, book="dash", interval="15m", wallet=1000.0, top=2, market="perps", fetch_funding=lambda: {})
        logs = []
        for end in range(250, 500):
            t.tick(market=market_at(series, end), universe=universe(symbols))
        t._log("[00:00:00] test feed line")
        brain.set_state("trader:dash", {**__import__("dataclasses").asdict(t.wallet), "market": "perps"})
        text = dashboard.render(brain, "dash", prices={}, width=140, height=50)
        self.assertIn("BIG BRAIN TIME", text)
        self.assertIn("status unknown", text)
        st = brain.get_state("trader:dash"); st["last_tick"] = "2020-01-01 00:00:00"; brain.set_state("trader:dash", st)
        self.assertIn("TRADER STOPPED", dashboard.render(brain, "dash", prices={}, width=140, height=50))
        from bigbrain.trader import _now
        st["last_tick"] = _now(); brain.set_state("trader:dash", st)
        self.assertIn("trader running", dashboard.render(brain, "dash", prices={}, width=140, height=50))
        self.assertIn("test feed line", text)
        self.assertIn("WHAT THE BRAIN BELIEVES", text)
        state = brain.get_state("trader:dash")
        if state["positions"]:
            p = next(iter(state["positions"].values()))
            bumped = dashboard.render(brain, "dash", prices={p["symbol"]: p["entry_price"] * 1.10}, width=140, height=50)
            self.assertNotEqual(bumped, text)
        self.assertEqual(len(dashboard.sparkline([1, 2, 3, 4, 5, 6, 7, 8], 8)), 8)
        self.assertEqual(dashboard.sparkline([5, 5, 5], 10), "▄▄▄")


if __name__ == "__main__":
    unittest.main()
