import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

from bigbrain import Brain, decider, lab, scalp
from bigbrain.cli import main
from bigbrain.ingest.market import Bar

COSTS = lab.Costs(fee=0.0005, spread_bps=1.0, slippage_bps=0.5)


def dated(bars, start_hour=0):
    """The same candles with real 15-minute timestamps."""
    out = []
    for i, b in enumerate(bars):
        minutes = start_hour * 60 + 15 * i
        day, rem = divmod(minutes, 1440)
        out.append(Bar(f"2026-{1 + day // 28:02d}-{1 + day % 28:02d} {rem // 60:02d}:{rem % 60:02d}", b.open, b.high, b.low, b.close, b.volume))
    return out


class StubJev:
    """Stands in for TypeSafe: follows the memory it is shown, skips when there is none."""

    def __init__(self):
        self.bodies = []

    def ask(self, body):
        self.bodies.append(body)
        crit = body["questions"]["decision"]["criteria"]
        with_signal = next(k for k, v in crit.items() if "with the signal" in v)
        mem = body["state"].get("memory")
        if mem is None:
            choice = with_signal
        else:
            verdict = mem["this_setup_in_this_context"]["verdict"]
            choice = "SKIP" if "losing" in verdict or verdict == "too few trades to judge" else with_signal
        probs = {k: (0.8 if k == choice else 0.1) for k in decider.CHOICES}
        return {"choice": choice, "probabilities": probs, "confidence": 0.7}


class MemoryTests(unittest.TestCase):
    def proposal(self, date, exit_date, ret, block):
        return decider.Proposal("X", date, "rsi_oversold", 1, {}, ("rsi_oversold", "uptrend", "mid"), ret, -ret, exit_date, block)

    def test_an_outcome_is_known_only_after_its_exit(self):
        a = self.proposal("2026-01-01 00:00", "2026-01-01 05:00", 0.01, "d1")
        b = self.proposal("2026-01-01 01:00", "2026-01-01 02:00", -0.01, "d1")
        m = decider.Memory()
        m.load([a, b])
        m.advance("2026-01-01 01:59")
        self.assertEqual(m.take.get(a.key, []), [])
        m.advance("2026-01-01 02:00")
        self.assertEqual(m.take[a.key], [-0.01])  # b exited first, though it was proposed later
        m.advance("2026-01-01 05:00")
        self.assertEqual(m.take[a.key], [-0.01, 0.01])
        self.assertEqual(m.fade[a.key], [0.01, -0.01])

    def test_lessons_are_written_and_recalled(self):
        props = [self.proposal(f"2026-01-{d:02d} 00:00", f"2026-01-{d:02d} 01:00", -0.002, f"d{d}") for d in range(1, 21)]
        m = decider.Memory(lesson_every=10)
        m.load(props)
        m.advance("2026-02-01 00:00")
        self.assertEqual(m.lessons, 2)
        ev = m.evidence(props[0])
        self.assertEqual(ev["this_setup_in_this_context"]["trades"], 20)
        self.assertIn("losing", ev["this_setup_in_this_context"]["verdict"])
        self.assertTrue(any("rsi_oversold" in lesson["title"] for lesson in ev["lessons"]))

    def test_errors_are_counted_across_days_not_trades(self):
        vals = [0.01, 0.012, 0.011, -0.01, -0.012, -0.011] * 5
        same_day = ["a", "a", "a", "b", "b", "b"] * 5
        _, se_days, k = decider.clustered(vals, [f"{d}{i // 6}" for i, d in enumerate(same_day)])
        _, se_trades, _ = decider.clustered(vals, [str(i) for i in range(len(vals))])
        self.assertEqual(k, 10)
        self.assertGreater(se_days, 1.5 * se_trades)  # trades that move together are not independent evidence
        self.assertEqual(decider.record([0.01] * 20, ["one day"] * 20)["verdict"], "too few trades to judge")


class RequestTests(unittest.TestCase):
    def setUp(self):
        bars = dated(scalp.random_walk("SECRETCOIN", n=1500, seed=2, vol=0.004))
        self.props = decider.proposals({"SECRETCOINUSDT": bars}, COSTS, "15m")

    def test_proposals_carry_both_sides_and_words_only(self):
        self.assertGreater(len(self.props), 50)
        p = self.props[10]
        self.assertTrue({"trend", "volatility"} <= set(p.context))
        self.assertTrue(all(isinstance(v, str) for v in p.context.values()))
        self.assertGreater(p.exit_date, p.date)
        self.assertEqual(p.block, p.date[:10])

    def test_jev_never_sees_the_symbol_or_the_date(self):
        m = decider.Memory()
        m.load(self.props)
        p = self.props[-1]
        m.advance(p.date)
        for evidence in (None, m.evidence(p)):
            text = json.dumps(decider.request_body(p, COSTS, evidence))
            self.assertNotIn("SECRETCOIN", text)
            self.assertNotIn("2026-", text)
        self.assertNotIn("memory", decider.request_body(p, COSTS, None)["state"])
        self.assertIn("memory", decider.request_body(p, COSTS, m.evidence(p))["state"])

    def test_criteria_name_the_signal_side(self):
        p = self.props[0]
        crit = decider.request_body(p, COSTS, None)["questions"]["decision"]["criteria"]
        self.assertEqual(set(crit), set(decider.CHOICES))
        with_signal = "LONG" if p.side == 1 else "SHORT"
        self.assertIn("with the signal", crit[with_signal])

    def test_invalid_answers_are_rejected(self):
        good = {"choice": "SKIP", "probabilities": {"LONG": 0.1, "SHORT": 0.1, "SKIP": 0.8}, "confidence": 0.7}
        self.assertEqual(decider.validate(good), good)
        for bad in ({**good, "choice": "BUY"}, {**good, "probabilities": {"LONG": 0.5, "SHORT": 0.5}},
                    {**good, "probabilities": {"LONG": 0.5, "SHORT": 0.5, "SKIP": 0.5}}, {**good, "confidence": 2}, {}):
            with self.assertRaises(ValueError):
                decider.validate(bad)


class RunTests(unittest.TestCase):
    def test_every_decider_sees_the_same_proposals_and_earns_its_choice(self):
        markets = {f"W{j}": dated(scalp.random_walk(f"W{j}", n=3000, seed=40 + j, vol=0.004)) for j in range(2)}
        props = decider.proposals(markets, COSTS, "15m")
        jev = StubJev()
        r = decider.run(props, COSTS, decider.DECIDERS, decisions=300, jev=jev)
        decided = r["proposals"]
        self.assertEqual(len(decided), 300)
        self.assertEqual(len(jev.bodies), 600)  # jev and jev_blind, once per proposal
        for p in decided:
            for d in decider.DECIDERS:
                c = p.choices[d]
                expect = 0.0 if c == "SKIP" else (p.take if (c == "LONG") == (p.side == 1) else p.fade)
                self.assertEqual(p.values[d], expect)
            self.assertEqual(p.values["take_all"], p.take)
            self.assertEqual(p.values["jev_blind"], p.take)  # the stub with no memory follows the signal
        s = r["summary"]
        self.assertEqual(s["take_all"]["taken"], 300)
        self.assertLess(s["jev"]["taken"], 300)
        text = decider.format_report(r, "test")
        self.assertIn("verdict:", text)
        self.assertIn("No decider made money distinguishable from luck", text)  # random walks: nothing makes money

    def test_memory_cannot_make_money_on_a_fair_walk(self):
        """The look-ahead guard: if outcomes leaked into memory early, following memory would profit here."""
        markets = {f"W{j}": dated(scalp.random_walk(f"W{j}", n=5000, seed=70 + j, vol=0.004)) for j in range(3)}
        props = decider.proposals(markets, lab.Costs(fee=0, spread_bps=0), "15m")
        r = decider.run(props, lab.Costs(fee=0, spread_bps=0), ("take_all", "beliefs", "jev"), decisions=1500, jev=StubJev())
        for d in ("beliefs", "jev"):
            self.assertLess(r["summary"][d]["t"], 2.4, d)


class JevClientTests(unittest.TestCase):
    def test_posts_once_and_caches(self):
        answer = {"choice": "LONG", "probabilities": {"LONG": 0.7, "SHORT": 0.1, "SKIP": 0.2}, "confidence": 0.6}
        reply = mock.MagicMock()
        reply.__enter__.return_value.read.return_value = json.dumps({"answers": {"decision": answer}}).encode()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", return_value=reply) as urlopen:
            jev = decider.Jev(key="k", cache=f"{tmp}/cache.jsonl")
            body = {"state": {"a": 1}, "questions": {}}
            self.assertEqual(jev.ask(body), answer)
            self.assertEqual(jev.ask(body), answer)
            self.assertEqual(urlopen.call_count, 1)
            request = urlopen.call_args[0][0]
            self.assertEqual(request.full_url, decider.JEV_URL)
            self.assertEqual(request.get_header("Authorization"), "Bearer k")
            again = decider.Jev(key="k", cache=f"{tmp}/cache.jsonl")  # a rerun reads the cache from disk
            self.assertEqual(again.ask(body), answer)
            self.assertEqual(urlopen.call_count, 1)

    def test_needs_a_key(self):
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
            with self.assertRaises(ValueError):
                decider.Jev()


class CliTests(unittest.TestCase):
    def test_runs_offline_without_a_key(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), redirect_stdout(out):
            code = main(["--db", ":memory:", "decide", "--synthetic", "--candles", "2500", "--decisions", "200"])
        self.assertEqual(code, 0)
        self.assertIn("Jev sits this one out", out.getvalue())
        self.assertIn("decider experiment", out.getvalue())

    def test_learns_the_result(self):
        markets = {"W": dated(scalp.random_walk("W", n=2500, seed=5, vol=0.004))}
        props = decider.proposals(markets, COSTS, "15m")
        r = decider.run(props, COSTS, ("take_all", "beliefs"), decisions=100)
        brain = Brain(":memory:")
        title = decider.learn_result(brain, r, "test")
        self.assertTrue(brain.recall("decider experiment beliefs take_all", k=3))
        self.assertIn("Decider experiment", title)


if __name__ == "__main__":
    unittest.main()
