"""The cortex: turns recalled cells into an answer.

Two modes:

* **Offline** (default, no dependencies): a structured briefing built from the
  recalled cells, their connections and their sources. Always available.
* **Claude-powered** (``pip install "bigbrain[cortex]"`` and an API key or an
  ``ant auth login`` profile): the recalled cells become the context for a
  Claude Opus 5 request, so the brain reasons over what it knows and answers
  like a professor citing their own notes. The model only sees what the brain
  recalled, so the answer is grounded in the brain's knowledge, not the
  model's guesses.
"""

from __future__ import annotations

import os
import sys

from bigbrain.brain import Brain
from bigbrain.cells import Recall

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You are the cortex of Big Brain Time, a trading knowledge brain.
You answer questions using ONLY the knowledge cells the brain has recalled, which are provided as context.
Each cell has an id, a kind (concept, paper, observation, lesson, note), a title, content and a source.

Answer like an experienced professor of trading: precise, honest about uncertainty, and explicit about risk.
Cite the cells you rely on by title. When cells disagree, say so. When the recalled knowledge is insufficient,
say what the brain would need to learn next instead of inventing facts. Never present a backtest as a guarantee."""


def is_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or _has_profile())


def _has_profile() -> bool:
    return (os.path.expanduser("~/.config/anthropic") and os.path.isdir(os.path.expanduser("~/.config/anthropic")))


def format_context(brain: Brain, recalls: list[Recall]) -> str:
    parts = []
    for r in recalls:
        links = ", ".join(f"{other.title} ({syn.weight:.2f})" for other, syn in brain.neighbors(r.cell.id, limit=4))
        parts.append(
            f"[cell {r.cell.id}] kind={r.cell.kind} score={r.score}\n"
            f"title: {r.cell.title}\nsource: {r.cell.source or 'n/a'}\ncontent: {r.cell.content}\n"
            f"linked to: {links or 'nothing yet'}"
        )
    return "\n\n".join(parts)


def answer_offline(brain: Brain, question: str, recalls: list[Recall]) -> str:
    if not recalls:
        return "The brain has not learned anything relevant yet. Feed it: `bigbrain seed`, `bigbrain learn papers`, `bigbrain learn market`."
    from bigbrain.watch import live_context

    lines = [f"Question: {question}"]
    live = live_context(brain)
    if live:
        lines += ["", "Live (from the watcher):"] + [f"  {l}" for l in live]
    lines += ["", "What the brain knows (strongest first):"]
    for i, r in enumerate(recalls, 1):
        how = "direct match" if not r.via else f"reached through {len(r.via)} linked cell(s)"
        lines.append(f"{i}. {r.cell.title} [{r.cell.kind}, score {r.score:.2f}, {how}]")
        lines.append(f"   {r.cell.summary(400)}")
        if r.cell.source:
            lines.append(f"   source: {r.cell.source}")
    top = recalls[0].cell
    neighbors = brain.neighbors(top.id, limit=5)
    if neighbors:
        lines += ["", f"Connections from '{top.title}':"]
        for other, syn in neighbors:
            lines.append(f"  - {other.title} (weight {syn.weight:.2f}, {syn.reason})")
    kinds = {r.cell.kind for r in recalls}
    missing = [k for k in ("paper", "observation", "lesson") if k not in kinds]
    if missing:
        lines += ["", "To answer this better the brain still needs: " + ", ".join(f"{k}s" for k in missing) + " on this topic."]
    return "\n".join(lines)


def answer_with_claude(brain: Brain, question: str, recalls: list[Recall], model: str = DEFAULT_MODEL) -> str:
    import anthropic

    client = anthropic.Anthropic()
    from bigbrain.watch import live_context

    context = format_context(brain, recalls) or "(the brain has recalled nothing relevant)"
    live = live_context(brain)
    if live:
        context = "Live market state from the watcher:\n" + "\n".join(live) + "\n\n" + context
    with client.beta.messages.stream(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Recalled knowledge cells:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        return "The cortex declined to answer this question."
    return "".join(block.text for block in response.content if block.type == "text")


def answer(brain: Brain, question: str, k: int = 8, use_claude: bool | None = None, model: str = DEFAULT_MODEL) -> tuple[str, list[Recall]]:
    """Recall relevant cells, then answer. Returns (answer text, recalled cells)."""
    recalls = brain.recall(question, k=k)
    if use_claude is None:
        use_claude = is_available()
    if use_claude:
        try:
            return answer_with_claude(brain, question, recalls, model=model), recalls
        except Exception as exc:  # bad key, no network, model unavailable: never lose the answer
            reason = _explain_failure(exc)
            print(f"(cortex unavailable: {reason}; showing the brain's recall instead)\n", file=sys.stderr)
    return answer_offline(brain, question, recalls), recalls


def _explain_failure(exc: Exception) -> str:
    try:
        import anthropic
    except ImportError:
        return str(exc)
    if isinstance(exc, anthropic.AuthenticationError):
        return "Anthropic rejected the API key; check ANTHROPIC_API_KEY or run `ant auth login`"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "this key is not allowed to use the model; try --model claude-sonnet-5"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate limited; try again in a minute"
    if isinstance(exc, anthropic.APIConnectionError):
        return "could not reach the Anthropic API (network?)"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}"
    return str(exc)
