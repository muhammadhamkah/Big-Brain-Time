import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from bigbrain import polymarket as pm
from bigbrain.cli import main
from bigbrain.net import HTTPStatusError


def raw_market(cid, price, per_day=10.0, closed=False, accepting=True):
    return {"condition_id": cid, "question": f"Will {cid} happen?", "market_slug": cid, "active": True, "closed": closed,
            "accepting_orders": accepting, "minimum_tick_size": 0.01, "minimum_order_size": 5,
            "tokens": [{"token_id": f"{cid}-yes", "outcome": "Yes", "price": price}, {"token_id": f"{cid}-no", "outcome": "No", "price": 1 - price}],
            "rewards": {"rates": [{"asset_address": "usdc", "rewards_daily_rate": per_day}], "min_size": 20, "max_spread": 3.5}}


def book(t, bids, asks, asset="A"):
    return {"type": "book", "t": t, "asset": asset, "market": "M", "bids": bids, "asks": asks, "tick": 0.01, "min": 5}


def trade(ts, price, size, asset="A"):
    return {"type": "trade", "t": ts, "ts": ts, "asset": asset, "market": "M", "side": "SELL", "price": price, "size": size}


class MarketTests(unittest.TestCase):
    def test_parse_and_choose_reward_markets(self):
        m = pm.parse_market(raw_market("c1", 0.4, per_day=25))
        self.assertEqual((m.reward_per_day, m.reward_max_spread, m.reward_min_size, m.min_size), (25.0, 3.5, 20.0, 5.0))
        self.assertIsNone(pm.parse_market(raw_market("c2", 0.4, closed=True)))
        self.assertIsNone(pm.parse_market(raw_market("c3", 0.4, accepting=False)))
        pages = {"MA==": {"data": [raw_market("a", 0.5, 5), raw_market("b", 0.97, 50)], "next_cursor": "Mg=="},
                 "Mg==": {"data": [raw_market("c", 0.3, 40), raw_market("d", 0.6, 1, closed=True)], "next_cursor": pm.END_CURSOR}}
        asked = []

        def fetch(path):
            asked.append(path)
            return pages[path.split("next_cursor=")[1]]

        chosen = pm.reward_markets(5, fetch=fetch)
        self.assertEqual([m.condition_id for m in chosen], ["c", "a"])  # b sits at 97c, d is closed; richest pool first
        self.assertEqual(len(asked), 2)

    def test_taker_fee_matches_polymarkets_examples(self):
        self.assertAlmostEqual(pm.taker_fee(100, 0.50, 0.07), 1.75)
        self.assertAlmostEqual(pm.taker_fee(100, 0.99, 0.07), 0.0693)
        self.assertEqual(pm.taker_fee(0.0001, 0.5, 0.07), 0.0)  # rounds to zero below 0.00001


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.market = pm.parse_market(raw_market("m1", 0.5))
        self.now = [1_800_000_000.0]
        self.trades = []

        self.urls = []

        def fetch(url):
            self.urls.append(url)
            assert url == "https://data-api.polymarket.com/v2/trades?condition=m1&limit=100", url
            return {"data": list(self.trades), "pagination": {"next_cursor": None}}

        def post(path, body):
            assert path == "/books"
            return [{"asset_id": b["token_id"], "market": "m1", "bids": [{"price": "0.48", "size": "10"}, {"price": "0.49", "size": "5"}],
                     "asks": [{"price": "0.52", "size": "7"}, {"price": "0.51", "size": "3"}], "tick_size": "0.01", "min_order_size": "5"} for b in body]

        def sleep(s):
            self.now[0] += s

        self.rec = pm.Recorder([self.market], self.dir.name, fetch=fetch, post=post, clock=lambda: self.now[0], sleep=sleep)

    def tearDown(self):
        self.dir.cleanup()

    def rows(self):
        return [json.loads(line) for f in Path(self.dir.name).glob("*.jsonl") for line in f.read_text().splitlines()]

    def ev(self, tx, price, size, ts):
        return {"conditionId": "m1", "asset": "m1-yes", "side": "SELL", "price": price, "size": size, "timestamp": ts,
                "transactionHash": tx, "outcome": "Yes", "outcomeIndex": 0}

    def test_books_are_sorted_best_first(self):
        self.rec.poll_books()
        rows = self.rows()
        self.assertEqual(len(rows), 2)  # both outcomes
        self.assertEqual(rows[0]["bids"][0], [0.49, 5.0])
        self.assertEqual(rows[0]["asks"][0], [0.51, 3.0])
        self.assertEqual(json.loads((Path(self.dir.name) / "markets.json").read_text())[0]["condition_id"], "m1")

    def test_trades_are_deduplicated_and_gaps_admitted(self):
        self.trades = [self.ev("t1", 0.5, 10, 1_800_000_000), self.ev("t2", 0.49, 4, 1_800_000_001_000)]  # seconds and milliseconds
        self.rec.poll_trades(self.market)
        self.rec.poll_trades(self.market)
        trades = [r for r in self.rows() if r["type"] == "trade"]
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[1]["ts"], 1_800_000_001.0)
        self.trades = [self.ev("t3", 0.5, 1, 1_800_000_005), self.ev("t4", 0.5, 1, 1_800_000_006)]  # nothing overlaps: maybe missed some
        self.rec.poll_trades(self.market)
        self.assertEqual(self.rec.stats["gaps"], 1)
        self.assertEqual(sum(r["type"] == "gap" for r in self.rows()), 1)

    def test_a_retired_source_falls_through_to_the_next(self):
        asked = []

        def fetch(url):
            asked.append(url)
            if "/v2/" in url:
                raise HTTPStatusError(404, url, {}, b"")
            return [self.ev("t1", 0.5, 10, 1_800_000_000)]  # v1 answers a bare list

        self.rec.fetch = fetch
        self.rec.poll_trades(self.market)
        self.rec.poll_trades(self.market)
        self.assertEqual(sum("/v2/" in u for u in asked), 1)  # the retired source is not asked again
        self.assertEqual(self.rec.stats["trades"], 1)
        sample = json.loads((Path(self.dir.name) / "sample-trade.json").read_text())
        self.assertIn("/trades?market=m1", sample["source"])

        def gone(url):
            raise HTTPStatusError(404, url, {}, b"")

        self.rec.fetch, self.rec.source = gone, 0
        with self.assertRaisesRegex(RuntimeError, "no public trade source"):
            self.rec.poll_trades(self.market)

    def test_trade_fields_are_read_under_any_of_their_names(self):
        old_style = {"market": {"asset_id": "m1-no"}, "side": "BUY", "price": "0.41", "size": "3", "timestamp": "1800000000",
                     "transaction_hash": "0xab"}
        clob_style = {"asset_id": "m1-yes", "side": "SELL", "price": "0.6", "size": "2", "match_time": "1800000000000", "id": "x"}
        by_index = {"outcomeIndex": 1, "price": 0.4, "size": 1, "timestamp": 1800000000}
        self.assertEqual(pm.normalize_trade(old_style, self.market)["asset"], "m1-no")
        t = pm.normalize_trade(clob_style, self.market)
        self.assertEqual((t["asset"], t["ts"], t["id"]), ("m1-yes", 1_800_000_000.0, "x"))
        self.assertEqual(pm.normalize_trade(by_index, self.market)["asset"], "m1-no")
        self.assertIsNone(pm.normalize_trade({"side": "BUY"}, self.market))

    def test_a_failing_market_does_not_block_the_others(self):
        two = [self.market, pm.parse_market(raw_market("m2", 0.4))]
        asked = []

        def fetch(url):
            asked.append(url)
            if "m1" in url:
                raise ConnectionError("timeout")
            return {"data": []}

        rec = pm.Recorder(two, self.dir.name, fetch=fetch, post=lambda p, b: [], clock=lambda: self.now[0],
                          sleep=lambda s: self.now.__setitem__(0, self.now[0] + s))
        rec.run(hours=10 / 3600)
        self.assertTrue(any("m2" in u for u in asked))
        self.assertGreater(rec.stats["errors"], 0)

    def test_run_stops_on_time_and_survives_errors(self):
        calls = {"n": 0}
        good = self.rec.post

        def flaky(path, body):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ConnectionError("network blip")
            return good(path, body)

        self.rec.post = flaky
        stats = self.rec.run(hours=60 / 3600)
        self.assertGreaterEqual(stats["books"], 20)
        self.assertEqual(stats["errors"], 1)


class SimulateTests(unittest.TestCase):
    def test_queue_fills_then_a_trade_through_sells(self):
        rows = [book(0, [[0.50, 100]], [[0.52, 100]]),
                trade(2, 0.50, 60),  # 100 shares queued ahead: 40 still ahead
                trade(3, 0.50, 50),  # the queue is gone and 10 more traded: our 10 fill
                book(4, [[0.50, 50]], [[0.52, 100]]),
                trade(6, 0.53, 20),  # a buyer paid above our ask: it filled
                book(8, [[0.50, 50]], [[0.52, 100]])]
        r = pm.simulate(rows, "A", size=10, latency=1, queue_model="queue")
        self.assertEqual([(f.side, f.price, f.size) for f in r.fills], [("BUY", 0.50, 10), ("SELL", 0.52, 10)])
        self.assertAlmostEqual(r.pnl_mid, 0.20)  # two cents on ten shares
        self.assertEqual(r.inventory, 0)
        strict = pm.simulate(rows, "A", size=10, latency=1, queue_model="through")
        self.assertEqual(strict.fills, [])  # trades at our price never fill in the strict model

    def test_latency_and_inventory_limits(self):
        rows = [book(0, [[0.50, 0]], [[0.52, 10]]), trade(0.5, 0.49, 50)]  # before the quote is live
        self.assertEqual(pm.simulate(rows, "A", size=10, latency=1).fills, [])
        rows = [book(0, [[0.50, 0]], [[0.52, 10]]), trade(2, 0.49, 50), book(3, [[0.50, 0]], [[0.52, 10]]), trade(5, 0.49, 50)]
        r = pm.simulate(rows, "A", size=10, max_inventory=10, latency=1)
        self.assertEqual(sum(f.size for f in r.fills if f.side == "BUY"), 10)  # stopped buying at the limit

    def test_large_quotes_are_placed_by_default(self):
        rows = [book(0, [[0.50, 0]], [[0.52, 10]]), trade(2, 0.49, 500)]
        r = pm.simulate(rows, "A", size=200, latency=1)
        self.assertEqual(sum(f.size for f in r.fills), 200)  # the inventory limit scales with the quote size

    def test_reward_eligibility_needs_the_minimum_size_near_the_mid(self):
        meta = {"reward_max_spread": 3.5, "reward_min_size": 200}
        rows = [book(0, [[0.50, 0]], [[0.52, 0]]), book(60, [[0.50, 0]], [[0.52, 0]])]
        self.assertEqual(pm.simulate(rows, "A", size=10, meta=meta).reward_s, 0)
        self.assertEqual(pm.simulate(rows, "A", size=200, meta=meta).reward_s, 60)

    def test_never_sells_what_it_does_not_hold(self):
        rows = [book(0, [[0.50, 1000]], [[0.52, 0]]), trade(2, 0.60, 100)]
        r = pm.simulate(rows, "A", size=10, latency=1)
        self.assertEqual(r.fills, [])

    def test_markout_and_leftovers(self):
        rows = [book(0, [[0.50, 0]], [[0.52, 10]]), trade(2, 0.49, 50),  # bought 10 at 0.50
                book(12, [[0.45, 10]], [[0.47, 10]]), book(70, [[0.44, 10]], [[0.46, 10]])]  # then the price fell away
        r = pm.simulate(rows, "A", size=10, latency=1, taker_rate=0.07)
        self.assertEqual(r.inventory, 10)
        self.assertAlmostEqual(pm.markout(r.fills, "mid10"), -4.0)  # cents per share against us
        self.assertAlmostEqual(pm.markout(r.fills, "mid60"), -5.0)
        self.assertAlmostEqual(r.pnl_mid, -0.5)
        self.assertAlmostEqual(r.pnl_dump, -5.0 + 4.4 - pm.taker_fee(10, 0.44, 0.07))
        self.assertTrue(r.quoted_s > 0)

    def test_quotes_stop_near_the_extremes(self):
        rows = [book(0, [[0.97, 0]], [[0.98, 10]]), trade(2, 0.96, 50)]
        self.assertEqual(pm.simulate(rows, "A", size=10, latency=1).fills, [])


class EndToEndTests(unittest.TestCase):
    def test_record_then_replay_from_the_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "polymarket" / "20261006-0100"
            m = pm.parse_market(raw_market("m1", 0.5))
            pm.Recorder([m], session, fetch=lambda p: [], post=lambda p, b: [])  # writes markets.json
            rows = [book(0, [[0.50, 0]], [[0.52, 10]], asset="m1-yes"), trade(2, 0.49, 50, asset="m1-yes"),
                    book(3, [[0.50, 0]], [[0.52, 0]], asset="m1-yes"), trade(5, 0.53, 50, asset="m1-yes"),
                    book(6, [[0.50, 0]], [[0.52, 0]], asset="m1-no")]
            (session / "2026-10-06.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            results = pm.run(session, size=10)
            self.assertEqual([r.asset for r in results], ["m1-yes"])  # one outcome per market
            self.assertAlmostEqual(results[0].pnl_mid, 0.20)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--db", str(Path(tmp) / "brain.db"), "poly", "simulate"])
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("Polymarket maker replay", text)
            self.assertIn("fill model 'through'", text)
            self.assertIn("Will m1 happen?", text)

    def test_markets_command_handles_an_unreachable_api(self):
        err = io.StringIO()
        with mock.patch.object(pm, "reward_markets", side_effect=OSError("blocked")), mock.patch("sys.stderr", err):
            self.assertEqual(main(["--db", ":memory:", "poly", "markets"]), 1)
        self.assertIn("could not reach Polymarket", err.getvalue())


if __name__ == "__main__":
    unittest.main()
