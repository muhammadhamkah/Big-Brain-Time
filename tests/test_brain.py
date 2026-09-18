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
        seed(self.brain)
        results = self.brain.recall("how much should I risk per trade when volatility is high", k=5)
        titles = [r.cell.title for r in results]
        self.assertIn("Position sizing", titles[:3])
        self.assertTrue(any(r.via for r in results), "activation should spread across synapses")

    def test_recall_reinforces_links(self):
        seed(self.brain)
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
        self.assertEqual(seed(self.brain), len(CURRICULUM))
        self.assertEqual(seed(self.brain), 0)
        stats = self.brain.stats()
        self.assertEqual(stats["cells"], len(CURRICULUM))
        self.assertGreater(stats["synapses"], len(CURRICULUM))  # every concept links to more than one other

    def test_export_graph(self):
        seed(self.brain)
        graph = self.brain.export_graph()
        self.assertEqual(len(graph["nodes"]), len(CURRICULUM))
        self.assertEqual(len(graph["edges"]), self.brain.count_synapses())
        self.assertIn("graph brain {", self.brain.export_dot())

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
