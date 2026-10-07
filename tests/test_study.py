import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from bigbrain import decider, lab, scalp, study
from bigbrain.cli import main
from tests.test_decider import dated

COSTS = lab.Costs(fee=0.0005, spread_bps=1.0, slippage_bps=0.5)


class PaperTests(unittest.TestCase):
    def test_papers_carry_words_only_and_the_memory_of_their_time(self):
        markets = {"SECRETCOINUSDT": dated(scalp.random_walk("SECRETCOIN", n=3000, seed=4, vol=0.004))}
        props = decider.proposals(markets, COSTS, "15m")
        examples = study.build(props)
        self.assertEqual(len(examples), len(props))
        text = json.dumps([e.state for e in examples])
        self.assertNotIn("SECRETCOIN", text)
        self.assertNotIn("2026-", text)
        self.assertEqual(examples[0].state["record here"], "no trades yet")  # nothing had closed yet
        self.assertIn(examples[-1].beliefs, study.OPTIONS)

    def test_the_answer_key_leans_with_the_size_of_the_result(self):
        p = decider.Proposal("X", "d", "rsi_oversold", 1, {}, ("rsi_oversold", "uptrend", "mid"), 0.008, -0.010, "e", "b")
        key = dict(zip(study.OPTIONS, study.soft_target(p, 0.004)))
        self.assertAlmostEqual(sum(key.values()), 1.0)
        self.assertGreater(key["LONG"], key["SKIP"])
        self.assertGreater(key["SKIP"], key["SHORT"])
        small = dict(zip(study.OPTIONS, study.soft_target(decider.Proposal("X", "d", "rsi_oversold", 1, {}, (), 0.0003, -0.0015, "e", "b"), 0.004)))
        self.assertLess(small["LONG"], key["LONG"])  # a tiny win barely leans
        short = decider.Proposal("X", "d", "rsi_overbought", -1, {}, (), 0.008, -0.010, "e", "b")  # a short signal that paid
        self.assertEqual(study.values(short)["SHORT"], 0.008)


class ExamTests(unittest.TestCase):
    def setUp(self):
        self.examples = study.build(study.planted(days=300, per_day=6))

    def test_no_exam_is_studied_before_it_is_sat(self):
        ws = study.windows(self.examples, exams=3, study_first=0.4)
        self.assertEqual(len(ws), 3)
        for w in ws:
            self.assertTrue(w["train"] and w["test"])
            self.assertTrue(all(e.prop.exit_date < w["start"] for e in w["train"]))  # every result was known before the exam
            self.assertTrue(all(e.prop.date >= w["start"] for e in w["test"]))
        tested = [id(e) for w in ws for e in w["test"]]
        self.assertEqual(len(tested), len(set(tested)))  # each question is in one exam only
        self.assertLess(ws[0]["start"], ws[1]["start"])
        self.assertGreater(len(ws[2]["train"]), len(ws[0]["train"]))  # later exams have studied more

    def test_rules_find_a_planted_rule_and_skip_noise(self):
        r = study.sit(study.windows(study.build(study.planted(days=600)), exams=2), [study.RulesLearner()])
        o = r["overall"]
        self.assertGreater(o["rules"]["t"], 4)
        self.assertGreater(o["rules"]["per_trade"], 0.001)
        self.assertLess(o["take_all"]["per_opportunity"], 0)
        self.assertIn("rules made money", study.verdict(r))
        markets = {f"W{j}": dated(scalp.random_walk(f"W{j}", n=12000, seed=80 + j, vol=0.004)) for j in range(3)}
        noise = study.sit(study.windows(study.build(decider.proposals(markets, COSTS, "15m")), exams=2), [study.RulesLearner()])
        self.assertLess(noise["overall"]["rules"]["t"], 2.4)  # a fair walk has no rule to find
        self.assertIn("No learner made money", study.verdict(noise))

    def test_report_and_memory(self):
        r = study.sit(study.windows(self.examples, exams=2), [study.RulesLearner()])
        text = study.format_report(r, "practice")
        self.assertIn("exam 1", text)
        self.assertIn("verdict:", text)
        from bigbrain import Brain
        brain = Brain(":memory:")
        self.assertIn("Study:", study.learn_result(brain, r, "practice"))


class CliTests(unittest.TestCase):
    def test_practice_run_without_laya_installed(self):
        out = io.StringIO()
        with mock.patch.object(study, "_require_laya", side_effect=ImportError("Laya is not installed: pip install laya")), redirect_stdout(out):
            code = main(["--db", ":memory:", "study", "--practice", "--practice-days", "400", "--exams", "2"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Laya sits this one out", text)
        self.assertIn("rules made money", text)


def _laya_available():
    return all(importlib.util.find_spec(m) for m in ("laya", "tokenizers", "torch", "transformers"))


def tiny_laya(base: Path, words: list[str]) -> None:
    """A small stand-in for the Laya checkpoint: the same DecisionModel and file layout, a two-layer encoder and a
    word-level tokenizer, so the training loop runs for real without downloading the 1 GB model."""
    import torch
    from laya.common import DecisionModel
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModel, ModernBertConfig, PreTrainedTokenizerFast

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(special + sorted(set(words)))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(base / "tokenizer")
    ecfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                            max_position_embeddings=1024, pad_token_id=0, cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3,
                            global_attn_every_n_layers=1)
    ecfg.save_pretrained(base / "encoder")
    torch.manual_seed(0)
    model = DecisionModel(AutoModel.from_config(ecfg), head_layers=1, n_act=2)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(base / "model.safetensors"))
    (base / "rl_agent_config.json").write_text(json.dumps({"encoder": "tiny", "head_layers": 1, "act_costs": {"ask": 1.0}, "max_len": 512, "head_max_len": 128}))


@unittest.skipUnless(_laya_available(), "needs pip install laya (PyTorch)")
class LayaTests(unittest.TestCase):
    def test_the_training_loop_runs_and_learns_a_planted_rule(self):
        """The official loss on a tiny stand-in trained from scratch (so with higher learning rates than Laya's fine-tuning):
        it must come to favour the planted rule's trades. The real, pre-trained model is only examined on a Mac."""
        import re
        import tempfile

        examples = study.build(study.planted(days=240, per_day=8, seed=3))
        words = re.findall(r"\w+|[^\w\s]", json.dumps([e.state for e in examples]) + " " + study.INSTRUCTIONS + " " + json.dumps(study.CRITERIA))
        with tempfile.TemporaryDirectory() as tmp:
            tiny_laya(Path(tmp) / "laya_base", words)
            learner = study.LayaLearner(tmp, train_size=2000, epochs=8, micro_batch=32, grad_accum=1, device="cpu", encoder_lr=1e-3, head_lr=1e-3)
            ws = study.windows(examples, exams=1, study_first=0.6)
            ws[0]["test"] = ws[0]["test"][:600]
            r = study.sit(ws, [learner])
        decided = r["exams"][0]["props"]
        self.assertEqual(len(decided), 600)
        rule = [p for p in decided if p.side == 1 and p.context["momentum"] == "positive and rising"]
        other = [p for p in decided if p not in rule]
        long_on_rule = sum(p.choices["laya"] == "LONG" for p in rule) / len(rule)
        long_elsewhere = sum(p.choices["laya"] == "LONG" for p in other) / len(other)
        self.assertGreater(long_on_rule, long_elsewhere + 0.25, (long_on_rule, long_elsewhere))

if __name__ == "__main__":
    unittest.main()
