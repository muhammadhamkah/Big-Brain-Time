"""Ingestion pipelines: each one turns a source of trading knowledge into cells.

Every pipeline separates *fetching* (network, safe to run in parallel) from
*learning* (writes to the brain, done on one thread). The bridge is ``Item``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    """One piece of knowledge ready to be learned."""

    kind: str
    title: str
    content: str
    source: str = ""
    extra_concepts: list[str] = field(default_factory=list)


def learn_items(brain, items: list[Item]) -> list[str]:
    """Teach the brain every item. Returns the titles learned."""
    learned = []
    for item in items:
        cell, _ = brain.learn(item.kind, item.title, item.content, source=item.source, extra_concepts=item.extra_concepts)
        learned.append(cell.title)
    return learned
