"""Study: train a decision model on years of past trades, then grade it on years it has never seen.

Like a student working through past exam papers, the model studies thousands of old trade proposals, each
with what actually happened, and is then examined on a later stretch of time it never studied. Only the exam
counts. A model that scores well on the papers it studied and badly on a new one memorised; it did not learn.

* **Past papers.** Every signal the trader's playbook fired over the years, described in words (the setup,
  the market, and the brain's record of that setup up to that moment), with what taking it, fading it and
  skipping it would have earned after costs.
* **The answer key.** Not one right answer: a trade that made 80 bp says "take it" firmly, one that made
  3 bp barely leans. The key gives each of LONG, SHORT and SKIP a probability that grows with what it would
  have earned (a softmax of the results over a typical move), the soft targets Laya is trained on.
* **Exams.** The timeline is cut into windows. The first part is study only; each later window is an exam,
  answered by a model trained only on trades that had *closed* before the exam began.
* **Classmates.** Every exam is also sat by ``take_all`` (every signal), ``beliefs`` (the brain's rule on its
  memory) and ``rules`` (simple rules learned from the same papers: "when the signal is A and the market shows B,
  take this side"). If the big model cannot beat the simple rules, it has added nothing.
* **A test of the test.** ``planted()`` builds practice papers with a hidden rule in them. A learner that
  cannot find a rule known to be there will not find one in real markets.

Laya (Convai Innovations, Apache 2.0) is trained exactly as its own Apple-silicon script trains it: soft
targets, a policy-gradient term rewarded by proper scoring rules plus a soft cross-entropy term, AdamW with a
cosine schedule. It needs ``pip install laya`` (PyTorch and the model, about 1 GB, downloaded once).
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from bigbrain import decider
from bigbrain.decider import Proposal, clustered

OPTIONS = ("LONG", "SHORT", "SKIP")
INSTRUCTIONS = "Should this signal be traded? Choose the answer that pays best after trading costs."
CRITERIA = {"LONG": "buy now", "SHORT": "sell short now", "SKIP": "stay out"}
LEARNERS = ("rules", "laya")


@dataclass
class Example:
    prop: Proposal
    state: dict  # what the model reads: words only, no symbol, no date
    beliefs: str  # the brain's belief rule at the time, for the classmate that uses it
    choices: dict = field(default_factory=dict)


def values(p: Proposal) -> dict:
    """What each answer would have earned."""
    long, short = (p.take, p.fade) if p.side == 1 else (p.fade, p.take)
    return {"LONG": long, "SHORT": short, "SKIP": 0.0}


def soft_target(p: Proposal, scale: float) -> list[float]:
    """The answer key: each option's probability grows with what it would have earned, in units of a typical move."""
    v = values(p)
    z = [v[o] / scale for o in OPTIONS]
    top = max(z)
    e = [math.exp(x - top) for x in z]
    return [x / sum(e) for x in e]


def compact(record: dict) -> str:
    if not record.get("trades"):
        return "no trades yet"
    return f"{record['verdict']}, {record['average']} over {record['trades']} trades"


def build(props: list[Proposal]) -> list[Example]:
    """The past papers, oldest first. The memory each one carries knows only trades closed before it."""
    memory = decider.Memory(lessons=False)
    memory.load(props)
    out = []
    for p in props:
        memory.advance(p.date)
        ev = memory.evidence(p)
        state = {"signal": f"{p.signal} ({decider.SIGNALS[p.signal][1]}), pointing {'long' if p.side == 1 else 'short'}",
                 "market": "; ".join(f"{k}: {v}" for k, v in p.context.items()),
                 "record here": compact(ev["this_setup_in_this_context"]),
                 "betting against it here": compact(ev["betting_against_it_in_this_context"]),
                 "record anywhere": compact(ev["this_setup_in_any_context"])}
        out.append(Example(p, state, decider.beliefs_choice(p, ev)))
    return out


def windows(examples: list[Example], exams: int = 3, study_first: float = 0.4) -> list[dict]:
    """Exam windows: the first ``study_first`` of the timeline is study only, the rest is cut into ``exams`` windows.

    Each exam trains on trades that closed before it began: a trade still open at the start is left out, because
    its result was not yet known."""
    dates = sorted({e.prop.date for e in examples})
    if len(dates) < 10 or exams < 1:
        raise ValueError("not enough history to set an exam")
    first = int(len(dates) * study_first)
    span = (len(dates) - first) // exams
    if span < 1:
        raise ValueError("not enough history for that many exams")
    out = []
    for k in range(exams):
        start = dates[first + k * span]
        end = dates[first + (k + 1) * span] if k < exams - 1 else None
        train = [e for e in examples if e.prop.exit_date < start]
        test = [e for e in examples if e.prop.date >= start and (end is None or e.prop.date < end)]
        out.append({"exam": k + 1, "start": start, "end": end or dates[-1], "train": train, "test": test})
    return out


# ---------------------------------------------------------------- learners
class RulesLearner:
    """The plain classmate: it learns simple rules from the same papers, of the form "when the signal (or its direction)
    is A and one feature of the market is B, take the long (or short) side". A rule is adopted only when it paid with
    strong evidence across days (three standard errors by default: many candidate rules are checked, and at two some
    would pass by luck). On an exam question it follows the strongest rule that applies, and skips when none does."""

    name = "rules"

    def __init__(self, min_trades: int = 50, min_t: float = 3.0) -> None:
        self.min_trades, self.min_t = min_trades, min_t
        self.rules: dict[tuple, tuple[str, float]] = {}

    @staticmethod
    def cells(e: Example) -> list[tuple]:
        p = e.prop
        anchors = [("signal", p.signal), ("direction", "long" if p.side == 1 else "short")]
        feats = list(p.context.items())
        return [(a,) for a in anchors] + [(a, f) for a in anchors for f in feats]

    def fit(self, train: list[Example], log=None) -> None:
        groups: dict[tuple, list[Example]] = {}
        for e in train:
            for c in self.cells(e):
                groups.setdefault(c, []).append(e)
        self.rules = {}
        for c, es in groups.items():
            if len(es) < self.min_trades:
                continue
            blocks = [e.prop.block for e in es]
            for o in ("LONG", "SHORT"):
                m, se, days = clustered([values(e.prop)[o] for e in es], blocks)
                t = m / se if se else 0.0
                if days >= 10 and t >= self.min_t and t > self.rules.get(c, ("", 0.0))[1]:
                    self.rules[c] = (o, t)
        if log:
            shown = sorted(self.rules.items(), key=lambda kv: -kv[1][1])[:3]
            log(f"    {len(self.rules)} rules adopted" + "".join(f"; {o} when {' and '.join(f'{k}={v}' for k, v in c)} (t {t:.1f})" for c, (o, t) in shown))

    def predict(self, test: list[Example], log=None) -> list[dict]:
        out = []
        for e in test:
            best = max((self.rules[c] for c in self.cells(e) if c in self.rules), key=lambda r: r[1], default=("SKIP", 0.0))
            out.append({o: (0.9 if o == best[0] else 0.05) for o in OPTIONS})
        return out


class LayaLearner:
    """Laya fine-tuned on the papers, the way its own Apple-silicon script fine-tunes it; a fresh copy of the base
    model for every exam, so no exam is sat by a model that studied it."""

    name = "laya"

    def __init__(self, workdir: str | Path, model_id: str = "convaiinnovations/laya", train_size: int = 4000, epochs: int = 2,
                 micro_batch: int = 2, grad_accum: int = 16, train_layers: int = 0, device: str = "auto", max_len: int = 512,
                 seed: int = 7) -> None:
        self.workdir, self.model_id = Path(workdir), model_id
        self.train_size, self.epochs, self.micro_batch, self.grad_accum = train_size, epochs, micro_batch, grad_accum
        self.train_layers, self.device_name, self.max_len, self.head_max_len, self.seed = train_layers, device, max_len, 128, seed
        self.model = None
        _require_laya()

    # -- set-up
    def _base(self) -> Path:
        from huggingface_hub import snapshot_download
        from laya.agent import _fix_tokenizer_config

        base = self.workdir / "laya_base"
        if not (base / "model.safetensors").exists():
            snapshot_download(self.model_id, local_dir=str(base))
        _fix_tokenizer_config(str(base))
        return base

    def _device(self):
        import torch

        if self.device_name in ("auto", "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if self.device_name in ("auto", "cuda") and torch.cuda.is_available() else "cpu")

    def _items(self, examples: list[Example], scale: float | None) -> list[dict]:
        from laya.common import QTYPES, build_sequence

        out = []
        for e in examples:
            ids, markers = build_sequence(self.tok, e.state, {"t": "choice", "ins": INSTRUCTIONS, "crit": CRITERIA}, self.max_len, self.head_max_len)
            if len(markers) != len(OPTIONS):
                continue
            target = soft_target(e.prop, scale) if scale else [1 / 3] * 3
            out.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": target})
        return out

    @staticmethod
    def _collate(items: list[dict], pad_id: int):
        import torch

        b, n = len(items), max(len(i["ids"]) for i in items)
        k = max(len(i["markers"]) for i in items)
        ids = torch.full((b, n), pad_id, dtype=torch.long)
        att = torch.zeros((b, n), dtype=torch.long)
        pos = torch.zeros((b, k), dtype=torch.long)
        mask = torch.zeros((b, k), dtype=torch.bool)
        target = torch.zeros((b, k), dtype=torch.float32)
        for r, it in enumerate(items):
            ids[r, :len(it["ids"])] = torch.tensor(it["ids"])
            att[r, :len(it["ids"])] = 1
            pos[r, :len(it["markers"])] = torch.tensor(it["markers"])
            mask[r, :len(it["markers"])] = True
            target[r, :len(it["target"])] = torch.tensor(it["target"])
        return ids, att, pos, mask, target, torch.tensor([it["qtype"] for it in items], dtype=torch.long)

    # -- study
    def fit(self, train: list[Example], log=None) -> None:
        import torch
        from laya.common import build_model, proper_reward
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.release()
        base = self._base()
        cfg = json.loads((base / "rl_agent_config.json").read_text())
        cfg.update({"max_len": self.max_len, "head_max_len": self.head_max_len, "gradient_checkpointing": True})
        self.tok = AutoTokenizer.from_pretrained(base / "tokenizer")
        model = build_model(cfg, encoder_dir=str(base / "encoder"))
        model.load_state_dict(load_file(str(base / "model.safetensors")), strict=True)
        model.float()
        if self.train_layers > 0:  # study only the top layers and the decision head: lighter, and less room to memorise
            total = model.encoder.config.num_hidden_layers
            for name, prm in model.named_parameters():
                m = re.search(r"\.layers\.(\d+)\.", name)
                prm.requires_grad = not name.startswith("encoder.") or bool(m and int(m.group(1)) >= total - self.train_layers)
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
        device = self._device()
        model.to(device).train()

        rng = random.Random(self.seed)
        papers = train if len(train) <= self.train_size else rng.sample(train, self.train_size)
        scale = typical_move(papers)
        items = self._items(papers, scale)
        if log:
            log(f"    studying {len(items)} papers on {device} for {self.epochs} epochs (a typical move is {scale * 1e4:.0f} bp) ...")
        enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
        updates = max(1, math.ceil(len(items) / self.micro_batch / self.grad_accum) * self.epochs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)
        for epoch in range(self.epochs):
            random.Random(self.seed + epoch).shuffle(items)
            opt.zero_grad(set_to_none=True)
            sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, self.epochs - 1)  # exploration noise anneals as in Laya's own recipe
            total, steps = 0.0, 0
            for start in range(0, len(items), self.micro_batch):
                ids, att, pos, mask, target, qtype = (t.to(device) for t in self._collate(items[start:start + self.micro_batch], self.tok.pad_token_id))
                logits, act = model(ids, att, pos, mask, qtype)
                logits = logits.float()
                k = mask.sum(-1, keepdim=True).float()
                eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
                noisy = logits.detach().unsqueeze(0) + eps
                probs = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    reward = proper_reward(probs, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                    adv = reward - reward.mean(0, keepdim=True)
                    adv = adv / (adv.std() + 1e-6)
                logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                loss_rl = -(adv * logp).mean()
                loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
                loss = (loss_rl + loss_ce + 0.0 * act.sum()) / self.grad_accum
                loss.backward()
                steps += 1
                if steps % self.grad_accum == 0 or start + self.micro_batch >= len(items):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                total += loss.item() * self.grad_accum
                if log and steps % 500 == 0:
                    log(f"      epoch {epoch + 1}/{self.epochs}, {start + self.micro_batch}/{len(items)} papers, loss {total / steps:.4f}")
            if log:
                log(f"    epoch {epoch + 1}/{self.epochs} done, average loss {total / max(1, steps):.4f}")
        model.eval()
        self.model, self.device = model, device

    def predict(self, test: list[Example], log=None, batch: int = 16) -> list[dict]:
        import torch

        items = self._items(test, None)
        if len(items) != len(test):
            raise RuntimeError("some exam papers did not fit Laya's input; shorten the state")
        out = []
        with torch.no_grad():
            for start in range(0, len(items), batch):
                ids, att, pos, mask, _, qtype = (t.to(self.device) for t in self._collate(items[start:start + batch], self.tok.pad_token_id))
                logits, _ = self.model(ids, att, pos, mask, qtype)
                p = torch.softmax(logits.float().masked_fill(~mask, -1e4), -1).cpu().tolist()
                out += [dict(zip(OPTIONS, row[:3])) for row in p]
        return out

    def release(self) -> None:
        """Free the previous exam's model before studying for the next one."""
        if self.model is not None:
            self.model = None
            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def _require_laya() -> None:
    import importlib.util

    if not all(importlib.util.find_spec(m) for m in ("laya", "torch")):
        raise ImportError("Laya is not installed: pip install laya  (it brings PyTorch; the model, about 1 GB, downloads on first use)")


def typical_move(examples: list[Example]) -> float:
    """The scale of the answer key: the median size of a trade's result in the papers studied."""
    sizes = sorted(abs(e.prop.take) for e in examples if e.prop.take)
    return sizes[len(sizes) // 2] if sizes else 0.005


# ------------------------------------------------------------------- exams
def sit(examples_by_window: list[dict], learners: list, log=None) -> dict:
    """Every learner studies for every exam from scratch, then every classmate sits it."""
    results = []
    for w in examples_by_window:
        if log:
            log(f"  exam {w['exam']}: {w['start'][:10]} to {w['end'][:10]}, {len(w['test'])} questions, studied from {len(w['train'])} closed trades")
        for e in w["test"]:
            e.choices = {"take_all": "LONG" if e.prop.side == 1 else "SHORT", "beliefs": e.beliefs}
        for learner in learners:
            if log:
                log(f"    {learner.name} studying ...")
            learner.fit(w["train"], log=log)
            answers = learner.predict(w["test"], log=log)
            for e, probs in zip(w["test"], answers):
                e.choices[learner.name] = max(OPTIONS, key=lambda o: probs[o])
        names = ["take_all", "beliefs"] + [lr.name for lr in learners]
        props = _graded(w["test"], names)
        results.append({"exam": w["exam"], "start": w["start"], "end": w["end"], "studied": len(w["train"]),
                        "summary": decider.summarize(props, names), "props": props})
        for learner in learners:
            if hasattr(learner, "release"):
                learner.release()
    every = [p for r in results for p in r["props"]]
    names = ["take_all", "beliefs"] + [lr.name for lr in learners]
    return {"exams": results, "overall": decider.summarize(every, names) if every else {}, "names": names}


def _graded(examples: list[Example], names: list[str]) -> list[Proposal]:
    out = []
    for e in examples:
        p = Proposal(e.prop.symbol, e.prop.date, e.prop.signal, e.prop.side, e.prop.context, e.prop.key, e.prop.take, e.prop.fade,
                     e.prop.exit_date, e.prop.block)
        p.choices = {n: e.choices[n] for n in names}
        p.values = {n: values(p)[e.choices[n]] for n in names}
        out.append(p)
    return out


def format_report(result: dict, label: str) -> str:
    names = result["names"]
    lines = [f"study: {label}", "", "each exam is a stretch of time the learners never studied; a skip earns zero, the same as doing nothing", ""]
    head = f"  {'':12}" + "".join(f"{n:>15}" for n in names)
    for r in result["exams"]:
        lines.append(f"exam {r['exam']}: {r['start'][:10]} to {r['end'][:10]}  ({r['summary'][names[0]]['opportunities']} questions, {r['studied']} closed trades studied)")
        lines.append(head)
        s = r["summary"]
        lines.append(f"  {'per question':12}" + "".join(f"{s[n]['per_opportunity'] * 1e4:+12.1f} bp" for n in names))
        lines.append(f"  {'t':12}" + "".join(f"{s[n]['t']:15.1f}" for n in names))
        lines.append(f"  {'taken':12}" + "".join(f"{s[n]['taken']:15}" for n in names))
        lines.append("")
    o = result["overall"]
    if o:
        lines.append("all exams together, per question (skips count as zero); t across days:")
        for n in names:
            r = o[n]
            lines.append(f"  {n:10} {r['per_opportunity'] * 1e4:+8.1f} bp  t {r['t']:5.1f}  took {r['taken']:6} of {r['opportunities']}, "
                         f"{r['win_rate']:.0%} won, {r['per_trade'] * 1e4:+.1f} bp per trade taken; against take_all {r['vs_take_all'] * 1e4:+.1f} bp (t {r['vs_take_all_t']:.1f})")
        lines += ["", verdict(result)]
    return "\n".join(lines)


def verdict(result: dict) -> str:
    o, names = result["overall"], result["names"]
    bar = 2.4
    parts = []
    passed = [n for n in names if o[n]["t"] >= bar and all(r["summary"][n]["per_opportunity"] > 0 for r in result["exams"])]
    if passed:
        best = max(passed, key=lambda n: o[n]["per_opportunity"])
        parts.append(f"{best} made money on unseen years: {o[best]['per_opportunity'] * 1e4:+.1f} bp per question, {o[best]['t']:.1f} standard errors, "
                     "positive in every exam.")
    else:
        parts.append("No learner made money on the unseen years with a margin distinguishable from luck and in every exam.")
    if "laya" in o and "rules" in o:
        gap = o["laya"]["per_opportunity"] - o["rules"]["per_opportunity"]
        parts.append(f"Laya against the simple rules learned from the same papers: {gap * 1e4:+.1f} bp per question.")
    for n in ("laya", "rules"):
        if n in o:
            r = o[n]
            if r["skipped_would_have"] is not None and r["taken"]:
                wise = r["skipped_would_have"] < r["per_trade"]
                parts.append(f"{n} chose {'well' if wise else 'no better than chance'}: the trades it took made {r['per_trade'] * 1e4:+.1f} bp, "
                             f"the ones it skipped would have made {r['skipped_would_have'] * 1e4:+.1f} bp.")
    return "verdict: " + " ".join(parts)


def learn_result(brain, result: dict, label: str) -> str:
    title = f"Study: {', '.join(result['names'])} on {label}"
    o = result["overall"]
    text = (f"Learners were trained on past trade proposals from the trader's signals on {label} and examined on {len(result['exams'])} later stretches of "
            "time they never studied. " + " ".join(f"{n} earned {r['per_opportunity'] * 1e4:+.1f} bp per question ({r['t']:.1f} standard errors across days), "
                                                    f"taking {r['taken']} of {r['opportunities']}." for n, r in o.items()) + " " + verdict(result))
    brain.forget(title=title)
    brain.learn("lesson", title, text, source="study", extra_concepts=["machine learning", "walk-forward", "overfitting", "laya"])
    brain.commit()
    return title


# ----------------------------------------------------------- practice papers
PRACTICE_RULE = "a long signal while momentum is positive and rising wins; everything else loses its costs"


def planted(days: int = 900, per_day: int = 12, seed: int = 1) -> list[Proposal]:
    """Practice papers with a hidden rule: long signals taken while momentum is positive and rising make money; every
    other proposal is noise that loses its costs. The rule lives in the market description, not in the memory's keys,
    so a learner has to read the description to find it."""
    rng = random.Random(seed)
    signals = list(decider.PLAYBOOK)
    trends = ["uptrend (strong)", "uptrend (weak)", "downtrend (strong)", "downtrend (weak)"]
    vols = ["low for this market", "mid for this market", "high for this market"]
    moms = ["positive and rising", "positive and falling", "negative and rising", "negative and falling"]
    rsis = ["oversold (below 30)", "weak (30-45)", "neutral (45-55)", "strong (55-70)", "overbought (above 70)"]
    out = []
    start = datetime(2020, 1, 1)
    for d in range(days):
        day = (start + timedelta(days=d)).strftime("%Y-%m-%d")
        for k in range(per_day):
            hour = (k * 24) // per_day
            date = f"{day} {hour:02d}:00"
            signal = rng.choice(signals)
            side = decider.PLAYBOOK[signal]["side"]
            ctx = {"trend": rng.choice(trends), "volatility": rng.choice(vols), "rsi": rng.choice(rsis), "momentum": rng.choice(moms),
                   "last_24_candles": rng.choice(["fell", "went sideways", "rose"])}
            edge = 0.0050 if side == 1 and ctx["momentum"] == "positive and rising" else 0.0
            move = rng.gauss(edge, 0.006)  # the market move in the signal's direction
            cost = 0.0012
            regime = "uptrend" if ctx["trend"].startswith("up") else "downtrend"
            out.append(Proposal("PRACTICE", date, signal, side, ctx, (signal, regime, ctx["volatility"].split()[0]), move - cost, -move - cost,
                                f"{day} {min(hour + 1, 23):02d}:30", day))
    out.sort(key=lambda p: (p.date, p.signal))
    return out
