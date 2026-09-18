"""Data types that make up the brain."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


def make_cell_id(kind: str, title: str, content: str) -> str:
    """Deterministic id, so learning the same fact twice merges instead of duplicating."""
    digest = hashlib.sha1(f"{kind}\n{title}\n{content}".encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class Cell:
    """One unit of knowledge: a concept, a paper, a market observation, a lesson."""

    id: str
    kind: str  # concept | paper | observation | indicator | strategy | lesson | note
    title: str
    content: str
    source: str = ""
    concepts: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    activations: int = 0  # how many times recall has fired this cell

    @classmethod
    def new(cls, kind: str, title: str, content: str, source: str = "") -> "Cell":
        return cls(id=make_cell_id(kind, title, content), kind=kind, title=title, content=content, source=source)

    def summary(self, width: int = 140) -> str:
        text = " ".join(self.content.split())
        return text if len(text) <= width else text[: width - 1] + "…"


@dataclass
class Synapse:
    """A weighted link between two cells. Weight lives in [0, 1]."""

    a: str
    b: str
    weight: float
    reason: str = ""

    def other(self, cell_id: str) -> str:
        return self.b if cell_id == self.a else self.a


@dataclass
class Recall:
    """A cell returned by recall, with its score and how it was reached."""

    cell: Cell
    score: float
    via: list[str] = field(default_factory=list)  # ids of cells whose activation spread here
