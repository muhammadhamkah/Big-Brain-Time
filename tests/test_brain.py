import unittest

from bigbrain import Brain
from bigbrain.concepts import extract_concepts, stem, tokenize
from bigbrain.ingest.textbook import CURRICULUM, seed


class ConceptTests(unittest.TestCase):
    def test_extracts_canonical_concepts(self):
        found = extract_concepts("The RSI is overbought; consider a trailing stop and check the Sharpe ratio.")
        self.assertIn("rsi", found)
        self.assertIn("stop loss", found)
        self.assertIn("sharpe ratio", found)

    def test_extraction_is_ordered_by_first_appearance(self):
        found = extract_concepts("Drawdown first, then momentum.")
        self.assertEqual(found[:2], ["drawdown", "momentum"])

    def test_tokenizer_stems_and_drops_stopwords(self):
        toks = tokenize("The positions were positioned by position")
        self.assertEqual(set(toks), {"position"})
        self.assertEqual(stem("backtesting"), "backtest")
        self.assertEqual(stem("loss"), "loss")


class InterpretTests(unittest.TestCase):
    def test_rsi_readings_and_bands_become_trading_words(self):
        from bigbrain.concepts import interpret
        self.assertIn("oversold", interpret("BTC RSI at 28 and below its lower Bollinger band"))
        self.assertIn("overbought", interpret("RSI(14) is 74"))
        self.assertIn("breakout", interpret("closed above the upper band"))
        self.assertEqual(interpret("RSI is 55, nothing special"), "RSI is 55, nothing special")
        self.assertIn("mean reversion", extract_concepts(interpret("RSI at 28")))


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.brain = Brain()

    def test_learning_links_related_cells(self):
        self.brain.learn("concept", "RSI", "RSI above 70 is overbought and below 30 oversold; it measures momentum.")
        _, synapses = self.brain.learn("note", "Dip buying", "Buy when RSI is oversold, a mean reversion play with a stop loss.")
        self.assertEqual(len(synapses), 1)
        self.assertIn("rsi", synapses[0].reason)
        self.assertGreater(synapses[0].weight, 0.1)

    def test_unrelated_cells_do_not_link(self):
        self.brain.learn("note", "Cooking", "Boil pasta for nine minutes and add salt to the water.")
        _, synapses = self.brain.learn("note", "Gardening", "Tomatoes need six hours of sun and regular watering.")
        self.assertEqual(synapses, [])

    def test_learning_is_idempotent(self):
        self.brain.learn("concept", "Momentum", "Winners keep winning.")
        self.brain.learn("concept", "Momentum", "Winners keep winning.")
        self.assertEqual(self.brain.count_cells(), 1)

    def test_recall_ranks_direct_match_first_and_spreads(self):
        seed(self.brain, packs=False)
        results = self.brain.recall("how much should I risk per trade when volatility is high", k=5)
        titles = [r.cell.title for r in results]
        self.assertIn("Position sizing", titles[:3])
        self.assertTrue(any(r.via for r in results), "activation should spread across synapses")

    def test_recall_reinforces_links(self):
        seed(self.brain, packs=False)
        first = self.brain.recall("kelly criterion position sizing", k=3)
        a, b = first[0].cell.id, first[1].cell.id
        before = self.brain._get_synapse(a, b)
        self.brain.recall("kelly criterion position sizing", k=3)
        after = self.brain._get_synapse(a, b)
        self.assertIsNotNone(after)
        if before is not None:
            self.assertGreater(after.weight, before.weight)
        self.assertGreaterEqual(self.brain.get(a).activations, 2)

    def test_seed_is_idempotent_and_connected(self):
        from bigbrain.ingest.textbook import all_titles
        self.assertEqual(seed(self.brain, packs=False), len(CURRICULUM))
        self.assertEqual(seed(self.brain, packs=False), 0)
        stats = self.brain.stats()
        self.assertEqual(stats["cells"], len(CURRICULUM))
        self.assertGreaterEqual(len(all_titles()), len(CURRICULUM))
        self.assertEqual(len(all_titles()), len(set(all_titles())), "lesson titles must be unique across packs")
        self.assertGreater(stats["synapses"], len(CURRICULUM))  # every concept links to more than one other

    def test_export_graph(self):
        seed(self.brain, packs=False)
        graph = self.brain.export_graph()
        self.assertEqual(len(graph["nodes"]), len(CURRICULUM))
        self.assertEqual(len(graph["edges"]), self.brain.count_synapses())
        self.assertIn("graph brain {", self.brain.export_dot())

    def test_forget_removes_cells_and_their_links(self):
        from bigbrain.ingest.market import learn_bars, synthetic
        seed(self.brain, packs=False)
        learn_bars(self.brain, "SYNTH", synthetic(n=300), source="synthetic data")
        before = self.brain.count_cells()
        n = self.brain.forget(source="synthetic")
        self.assertGreater(n, 0)
        self.assertEqual(self.brain.count_cells(), before - n)
        self.assertEqual(self.brain.cells(kind="observation"), [])
        dangling = self.brain.db.execute(
            "SELECT COUNT(*) FROM synapses WHERE a NOT IN (SELECT id FROM cells) OR b NOT IN (SELECT id FROM cells)"
        ).fetchone()[0]
        self.assertEqual(dangling, 0)
        with self.assertRaises(ValueError):
            self.brain.forget()

    def test_persistence(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.db")
            b1 = Brain(path)
            b1.learn("note", "Persistent", "The brain remembers between runs.")
            b1.close()
            b2 = Brain(path)
            self.assertEqual(b2.count_cells(), 1)
            self.assertEqual(b2.find("Persist")[0].title, "Persistent")
            b2.close()


if __name__ == "__main__":
    unittest.main()


class KnowledgePackTests(unittest.TestCase):
    def test_full_seed_is_dense_and_recall_reaches_pack_lessons(self):
        from bigbrain.ingest.textbook import all_titles
        brain = Brain()
        n = seed(brain)
        self.assertEqual(n, len(all_titles()))
        self.assertGreater(brain.count_synapses() / n, 10, "each lesson should link to many others")
        titles = [r.cell.title for r in brain.recall("how do I compute the implied move before earnings from option prices", k=5)]
        self.assertTrue(any("implied move" in t.lower() or "earnings" in t.lower() for t in titles), titles)
        titles = [r.cell.title for r in brain.recall("what happened to LTCM and what does it teach about leverage", k=5)]
        self.assertTrue(any("LTCM" in t or "Long-Term Capital" in t for t in titles), titles)
