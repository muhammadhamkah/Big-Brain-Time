import json
import unittest

from bigbrain import Brain, lab
from bigbrain.ingest.market import synthetic
from bigbrain.strategy import StrategyTrader
from tests.test_trader import market_at, universe


def daily(symbols, n=500, seed=0, vol=0.03):
    from datetime import datetime, timedelta
    series = {}
    for i, s in enumerate(symbols):
        bars = synthetic(s, n=n, seed=seed + i, vol=vol, drift=0.001)
        t0 = datetime(2025, 1, 1)
        for j, b in enumerate(bars):
            b.date = (t0 + timedelta(days=j)).strftime("%Y-%m-%d %H:%M")
            b.quote_volume = b.volume * b.close
        series[s] = bars
    return series


class StrategyBookTests(unittest.TestCase):
    def setUp(self):
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.series = daily(self.symbols, seed=200)

    def test_the_book_follows_the_rule_and_keeps_a_control(self):
        brain = Brain()
        t = StrategyTrader(brain, book="tr", rule="trend_vt", symbols=self.symbols, interval="1d", wallet=1000.0, market="perps",
                           grid={"sma": [50], "vol_window": [20], "target_vol": [0.15], "short": [1]}, fetch_funding=lambda: {})
        actions = []
        for end in range(260, 500):
            r = t.tick(market=market_at(self.series, end), universe=universe(self.symbols))
            actions += [e["action"] for e in r["events"]]
            self.assertAlmostEqual(r["equity"], t.wallet.equity(), places=6)
            for p in t.wallet.positions.values():
                self.assertEqual(p["signal"], "trend_vt")
                self.assertLessEqual(p["notional"], 1000.0 * 1.6 / 3 + 1e-6)  # never more than the coin's equal share of the account (times the weight, at most 1)
                self.assertEqual(p["context"]["risk_pct"], 0.0)  # the rule carries no stop; the protective level never fills
                self.assertEqual(p["model_version"], brain.RESULTS_VERSION)
        self.assertIn("buy", actions)
        self.assertIn("short", actions)  # short=1: both directions
        self.assertIn("sell", actions)
        # frozen: the rule decides. The trades are tallied under the rule's own name, and no exit statistics are learned
        self.assertEqual([r[0] for r in brain.db.execute("SELECT DISTINCT signal FROM beliefs")], ["trend_vt"])
        self.assertEqual(brain.db.execute("SELECT COUNT(*) FROM exit_stats").fetchone()[0], 0)
        rows = brain.db.execute("SELECT signal, exit_reason, model_version FROM trades WHERE book = 'tr'").fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(r["signal"] == "trend_vt" and r["exit_reason"] in ("signal", "end") for r in rows))
        rep = t.report()
        self.assertEqual(rep["params"], {"sma": 50, "vol_window": 20, "target_vol": 0.15, "short": 1})
        self.assertEqual(rep["benchmark"]["symbols"], 3)
        self.assertEqual(rep["benchmark"]["since"], self.series["BTCUSDT"][258].date)
        cfg = brain.get_state("strategy:tr")
        self.assertEqual(cfg["hold_base"]["BTCUSDT"], self.series["BTCUSDT"][258].close)
        # a reopened book keeps its parameters, control and positions
        t2 = StrategyTrader(brain, book="tr", rule="trend_vt", symbols=self.symbols, fetch_funding=lambda: {})
        self.assertEqual(t2.params, rep["params"])
        self.assertEqual(t2.wallet.positions, json.loads(json.dumps(t.wallet.positions)))

    def test_entries_and_exits_land_on_the_candles_the_lab_traded(self):
        brain = Brain()
        sym = ["BTCUSDT"]
        series = {"BTCUSDT": self.series["BTCUSDT"]}
        params = {"sma": 50, "vol_window": 20, "target_vol": 0.15, "short": 1}
        t = StrategyTrader(brain, book="one", rule="trend_vt", symbols=sym, interval="1d", wallet=1000.0, market="perps",
                           grid={k: [v] for k, v in params.items()}, fetch_funding=lambda: {})
        for end in range(260, 500):
            t.tick(market=market_at(series, end), universe=universe(sym))
        bars = series["BTCUSDT"][:499]  # the last closed candle the book saw
        ref = lab.simulate(bars, lab.trend_vt_rule(bars, params, {"interval": "1d"}), lab.Costs(), "1d", start=259)
        lab_entries = sorted(bars[tr.entry_i].date for tr in ref.log if tr.reason != "end")
        lab_exits = sorted(bars[tr.exit_i].date for tr in ref.log if tr.reason != "end")
        book = brain.db.execute("SELECT entry_time, exit_time FROM trades WHERE book = 'one' ORDER BY entry_time").fetchall()
        self.assertEqual(sorted(r["entry_time"] for r in book), lab_entries)  # fills at the same next-open candles
        self.assertEqual(sorted(r["exit_time"] for r in book), lab_exits)

    def test_refit_chooses_on_history_and_logs_it(self):
        brain = Brain()
        t = StrategyTrader(brain, book="rf", rule="trend_vt", symbols=self.symbols, interval="1d", wallet=1000.0, market="perps",
                           grid={"sma": [30, 100], "vol_window": [20], "target_vol": [0.15], "short": [0, 1]}, refit_days=30, fetch_funding=lambda: {})
        t.tick(market=market_at(self.series, 300), universe=universe(self.symbols))
        first = t.params
        self.assertIn(first["sma"], (30, 100))
        self.assertEqual(t.last_refit, self.series["BTCUSDT"][298].date)
        self.assertTrue(any("REFIT" in line for line in t.wallet.log))
        t.tick(market=market_at(self.series, 310), universe=universe(self.symbols))
        self.assertEqual(t.last_refit, self.series["BTCUSDT"][298].date)  # not due yet
        t.tick(market=market_at(self.series, 335), universe=universe(self.symbols))
        self.assertEqual(t.last_refit, self.series["BTCUSDT"][333].date)  # thirty days on, chosen again

    def test_a_flip_exits_at_the_next_open_and_enters_right_after(self):
        brain = Brain()
        sym = ["BTCUSDT"]
        # a series that trends up, then down: the both-sided rule must go long, then flip to short
        import math
        up = [1.0 * (1.01 ** i) * (1 + 0.004 * math.sin(i * 1.7)) for i in range(150)]  # a little noise: real volatility, not zero
        down = [up[-1] * (0.99 ** i) * (1 + 0.004 * math.sin(i * 1.7)) for i in range(1, 150)]
        from bigbrain.ingest.market import Bar
        from datetime import datetime, timedelta
        bars = []
        t0 = datetime(2025, 1, 1)
        for i, c in enumerate(up + down):
            bars.append(Bar((t0 + timedelta(days=i)).strftime("%Y-%m-%d %H:%M"), c, c * 1.005, c * 0.995, c, 1000.0, 1000.0 * c))
        series = {"BTCUSDT": bars}
        t = StrategyTrader(brain, book="flip", rule="trend_vt", symbols=sym, interval="1d", wallet=1000.0, market="perps",
                           grid={"sma": [30], "vol_window": [20], "target_vol": [0.15], "short": [1]}, fetch_funding=lambda: {})
        sides = []
        for end in range(100, len(bars)):
            t.tick(market=market_at(series, end), universe=universe(sym))
            for p in t.wallet.positions.values():
                if not sides or sides[-1] != p["side"]:
                    sides.append(p["side"])
        self.assertEqual(sides, [1, -1])
        rows = brain.db.execute("SELECT side, exit_reason FROM trades WHERE book = 'flip'").fetchall()
        self.assertEqual([(r["side"], r["exit_reason"]) for r in rows], [(1, "signal")])
        self.assertEqual(t.wallet.pending, {})


class MarkTests(unittest.TestCase):
    def test_live_marks_and_the_control_use_the_quote_not_yesterdays_close(self):
        import time
        brain = Brain()
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        series = daily(symbols, seed=300)
        t = StrategyTrader(brain, book="mk", rule="trend_vt", symbols=symbols, interval="1d", wallet=1000.0, market="perps",
                           grid={"sma": [30], "vol_window": [20], "target_vol": [0.15], "short": [1]}, fetch_funding=lambda: {})
        market = market_at(series, 300)
        quotes = {s: {"bid": market[s][-2].close * 1.049, "ask": market[s][-2].close * 1.051, "at": time.time()} for s in symbols}  # a 5% rally since the close
        t.tick(market=market, universe=universe(symbols), quotes=quotes, live=True)
        self.assertTrue(t.wallet.positions)
        for p in t.wallet.positions.values():
            mid = (quotes[p["symbol"]]["bid"] + quotes[p["symbol"]]["ask"]) / 2
            self.assertAlmostEqual(p["mark"], mid)  # not yesterday's close
            self.assertLess(abs(p["side"] * (p["mark"] / p["entry_price"] - 1)), 0.002)  # entered at the quote, marked at the quote: no phantom loss
        self.assertAlmostEqual(t.report()["benchmark"]["return"], 0.0)  # the control starts where the book could trade
        self.assertGreater(t.wallet.equity(), 1000.0 - 1.0)  # only fees and impact are gone


class DashboardTests(unittest.TestCase):
    def test_dashboard_shows_the_book_against_its_control(self):
        from bigbrain import dashboard
        brain = Brain()
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        series = daily(symbols, seed=400)
        t = StrategyTrader(brain, book="db", rule="trend_vt", symbols=symbols, interval="1d", wallet=1000.0, market="perps",
                           grid={"sma": [30], "vol_window": [20], "target_vol": [0.15], "short": [1]}, fetch_funding=lambda: {})
        for end in range(260, 300):
            t.tick(market=market_at(series, end), universe=universe(symbols))
        base = brain.get_state("strategy:db")["hold_base"]
        text = dashboard.render(brain, "db", prices={s: base[s] * 1.10 for s in symbols}, width=160, height=50)
        self.assertIn("STRATEGY", text)
        self.assertIn("trend_vt with sma=30", text)
        self.assertIn("+10.00%", text)  # the control at live prices
        self.assertIn("gap", text)


if __name__ == "__main__":
    unittest.main()
