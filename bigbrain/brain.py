"""The Brain: a persistent, self-linking knowledge graph.

Learning
    ``learn()`` turns a piece of information into a cell, extracts the concepts
    it touches, and wires it to every existing cell that shares those concepts
    or the same vocabulary. Link strength combines concept overlap (Jaccard)
    with term-frequency cosine similarity.

Recall
    ``recall()`` scores cells against a question directly, then lets that
    activation spread along synapses so related knowledge surfaces even when
    it does not share words with the question. Cells recalled together are
    reinforced (Hebbian learning: "fire together, wire together").

Storage is a single SQLite file, so the brain survives restarts and grows.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from bigbrain.cells import Cell, Recall, Synapse
from bigbrain.concepts import extract_concepts, interpret, tokenize

SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    concepts TEXT NOT NULL DEFAULT '[]',
    tokens TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    activations INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS synapses (
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    weight REAL NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS synapses_b ON synapses(b);
CREATE TABLE IF NOT EXISTS cell_concepts (
    concept TEXT NOT NULL,
    cell_id TEXT NOT NULL,
    PRIMARY KEY (concept, cell_id)
);
CREATE TABLE IF NOT EXISTS cell_tokens (
    token TEXT NOT NULL,
    cell_id TEXT NOT NULL,
    tf INTEGER NOT NULL,
    PRIMARY KEY (token, cell_id)
);
CREATE INDEX IF NOT EXISTS cell_tokens_cell ON cell_tokens(cell_id);
CREATE TABLE IF NOT EXISTS journal (
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    strategy TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_time TEXT,
    exit_price REAL,
    qty REAL NOT NULL,
    pnl REAL,
    ret REAL,
    UNIQUE (symbol, interval, strategy, entry_time)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_time TEXT NOT NULL,
    exit_price REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    qty REAL NOT NULL,
    notional REAL NOT NULL,
    gross_ret REAL NOT NULL,
    net_ret REAL NOT NULL,
    pnl REAL NOT NULL,
    fees REAL NOT NULL,
    slippage REAL NOT NULL,
    bars_held INTEGER NOT NULL,
    mfe REAL NOT NULL,
    mae REAL NOT NULL,
    context TEXT NOT NULL,
    findings TEXT NOT NULL,
    explore INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS beliefs (
    book TEXT NOT NULL,
    signal TEXT NOT NULL,
    regime TEXT NOT NULL,
    vol_bucket TEXT NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    sum_ret REAL NOT NULL DEFAULT 0,
    sum_win REAL NOT NULL DEFAULT 0,
    sum_loss REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (book, signal, regime, vol_bucket)
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    signal TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    price REAL NOT NULL,
    horizon INTEGER NOT NULL,
    graded_at TEXT,
    outcome_return REAL,
    hit INTEGER,
    UNIQUE (symbol, interval, signal, bar_time)
);
"""


class Brain:
    """A trading knowledge brain backed by SQLite."""

    LINK_THRESHOLD = 0.12  # minimum similarity to grow a synapse
    # How much recall trusts each kind of knowledge. Forum talk and READMEs
    # still surface, but a paper or a measured backtest outranks them.
    KIND_TRUST = {
        "concept": 1.0,
        "paper": 1.0,
        "lesson": 1.0,
        "observation": 1.0,
        "note": 1.0,
        "article": 0.9,
        "postmortem": 0.95,
        "code": 0.8,
        "discussion": 0.7,
    }
    MAX_LINKS_PER_CELL = 30  # strongest links kept when a new cell arrives
    HEBBIAN_STEP = 0.05  # how much co-recall strengthens a synapse
    SPREAD_FACTOR = 0.3  # how much activation leaks across a synapse
    MAX_SPREAD_GAIN = 0.25  # cap on what one cell can collect from spreading

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ------------------------------------------------------------------ learn
    def learn(
        self,
        kind: str,
        title: str,
        content: str,
        source: str = "",
        extra_concepts: Iterable[str] = (),
    ) -> tuple[Cell, list[Synapse]]:
        """Store one piece of knowledge and wire it into the graph.

        Returns the cell and the synapses created for it. Re-learning the exact
        same fact is a no-op that returns the existing cell and no new links.
        """
        cell = Cell.new(kind, title, content, source)
        if self.get(cell.id) is not None:
            return self.get(cell.id), []  # type: ignore[return-value]

        text = f"{title}\n{content}"
        concepts = list(dict.fromkeys([*extract_concepts(text), *extra_concepts]))
        # Title terms count three times: a title says what a cell is about more than any body sentence.
        tokens = tokenize(text) + tokenize(title) + tokenize(title)
        cell.concepts = concepts

        self.db.execute(
            "INSERT INTO cells (id, kind, title, content, source, concepts, tokens, created_at, activations)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (cell.id, kind, title, content, source, json.dumps(concepts), json.dumps(tokens), cell.created_at),
        )
        self.db.executemany(
            "INSERT OR IGNORE INTO cell_concepts (concept, cell_id) VALUES (?, ?)",
            [(c, cell.id) for c in concepts],
        )
        self.db.executemany(
            "INSERT OR IGNORE INTO cell_tokens (token, cell_id, tf) VALUES (?, ?, ?)",
            [(t, cell.id, n) for t, n in tokens.items()],
        )

        synapses = self._wire(cell, tokens)
        self._journal("learn", f"{kind}: {title} (+{len(synapses)} links)")
        self.db.commit()
        return cell, synapses

    def _wire(self, cell: Cell, tokens: dict[str, int]) -> list[Synapse]:
        """Find related cells and create synapses to the strongest matches."""
        candidates: dict[str, float] = {}
        own_concepts = set(cell.concepts)

        # Concept overlap (Jaccard). Candidates come from the inverted index.
        if own_concepts:
            placeholders = ",".join("?" * len(own_concepts))
            rows = self.db.execute(
                f"SELECT cell_id, COUNT(*) AS shared FROM cell_concepts WHERE concept IN ({placeholders})"
                " AND cell_id != ? GROUP BY cell_id",
                (*own_concepts, cell.id),
            ).fetchall()
            counts = self._concept_counts([row["cell_id"] for row in rows])
            for row in rows:
                union = len(own_concepts) + counts.get(row["cell_id"], 0) - row["shared"]
                candidates[row["cell_id"]] = row["shared"] / union if union else 0.0

        # Vocabulary similarity (tf-idf cosine) for cells sharing informative terms.
        if tokens:
            n_cells = self.count_cells()
            df = self._doc_freq(tokens.keys())
            own_vec = {t: tf * self._idf(df.get(t, 0), n_cells) for t, tf in tokens.items()}
            own_norm = math.sqrt(sum(v * v for v in own_vec.values())) or 1.0
            placeholders = ",".join("?" * len(tokens))
            rows = self.db.execute(
                f"SELECT cell_id, token, tf FROM cell_tokens WHERE token IN ({placeholders}) AND cell_id != ?",
                (*tokens.keys(), cell.id),
            ).fetchall()
            dots: dict[str, float] = defaultdict(float)
            for row in rows:
                w = self._idf(df.get(row["token"], 0), n_cells)
                dots[row["cell_id"]] += own_vec[row["token"]] * row["tf"] * w
            if dots:
                strongest = sorted(dots, key=lambda cid: -dots[cid])[: self.MAX_LINKS_PER_CELL * 4]
                norms = self._norms(strongest, n_cells)
                for cid in strongest:
                    cosine = dots[cid] / (own_norm * norms.get(cid, 1.0))
                    candidates[cid] = 0.6 * candidates.get(cid, 0.0) + 0.4 * cosine if cid in candidates else 0.4 * cosine

        ranked = sorted(candidates.items(), key=lambda kv: -kv[1])[: self.MAX_LINKS_PER_CELL]
        synapses: list[Synapse] = []
        for cid, weight in ranked:
            if weight < self.LINK_THRESHOLD:
                continue
            other_concepts = set(json.loads(self.db.execute("SELECT concepts FROM cells WHERE id = ?", (cid,)).fetchone()[0]))
            shared = sorted(own_concepts & other_concepts)
            reason = "shares " + ", ".join(shared[:4]) if shared else "similar vocabulary"
            synapses.append(self._upsert_synapse(cell.id, cid, round(min(weight, 1.0), 4), reason))
        return synapses

    # ----------------------------------------------------------------- recall
    def recall(self, query: str, k: int = 8, spread: bool = True, reinforce: bool = True) -> list[Recall]:
        """Return the cells most relevant to ``query``, with activation spread over synapses."""
        query = interpret(query)
        concepts = set(extract_concepts(query))
        tokens = tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        n_cells = self.count_cells()
        if n_cells == 0:
            return []

        # Words that the brain has learned as concepts (symbols such as "AAPL", tags added by
        # ingestion) count as concepts too, even though they are not in the built-in lexicon.
        words = {w for w in re.findall(r"[a-z0-9][a-z0-9\-/.]*", query.lower()) if len(w) >= 2}
        if words:
            placeholders = ",".join("?" * len(words))
            rows = self.db.execute(
                f"SELECT DISTINCT concept FROM cell_concepts WHERE concept IN ({placeholders})", tuple(words)
            ).fetchall()
            concepts |= {r["concept"] for r in rows}

        # Channel 1: concept overlap, weighted by how rare each concept is. A concept
        # shared by five cells says far more than one shared by a hundred.
        concept_scores: dict[str, float] = defaultdict(float)
        if concepts:
            placeholders = ",".join("?" * len(concepts))
            df_c = {
                r["concept"]: r["n"]
                for r in self.db.execute(
                    f"SELECT concept, COUNT(*) AS n FROM cell_concepts WHERE concept IN ({placeholders}) GROUP BY concept",
                    tuple(concepts),
                )
            }
            weight = {c: self._idf(df_c.get(c, 0), n_cells) for c in concepts}
            total = sum(weight.values()) or 1.0
            for r in self.db.execute(
                f"SELECT cell_id, concept FROM cell_concepts WHERE concept IN ({placeholders})", tuple(concepts)
            ):
                concept_scores[r["cell_id"]] += weight[r["concept"]] / total

        # Channel 2: vocabulary (tf-idf cosine), normalized so the best lexical match scores 1.
        lexical_scores: dict[str, float] = {}
        if tokens:
            df = self._doc_freq(tokens.keys())
            q_vec = {t: tf * self._idf(df.get(t, 0), n_cells) for t, tf in tokens.items()}
            q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
            placeholders = ",".join("?" * len(tokens))
            rows = self.db.execute(
                f"SELECT cell_id, token, tf FROM cell_tokens WHERE token IN ({placeholders})", tuple(tokens.keys())
            ).fetchall()
            dots: dict[str, float] = defaultdict(float)
            for row in rows:
                dots[row["cell_id"]] += q_vec[row["token"]] * row["tf"] * self._idf(df.get(row["token"], 0), n_cells)
            strongest = sorted(dots, key=lambda cid: -dots[cid])[: max(k * 8, 40)]
            norms = self._norms(strongest, n_cells)
            cosines = {cid: dots[cid] / (q_norm * norms.get(cid, 1.0)) for cid in strongest}
            best = max(cosines.values(), default=0.0) or 1.0
            lexical_scores = {cid: c / best for cid, c in cosines.items()}

        for cid in set(concept_scores) | set(lexical_scores):
            scores[cid] = 0.5 * concept_scores.get(cid, 0.0) + 0.5 * lexical_scores.get(cid, 0.0)

        # Spreading activation: each seed wakes its strongest neighbours a little.
        via: dict[str, list[str]] = defaultdict(list)
        if spread and scores:
            seeds = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
            gains: dict[str, float] = defaultdict(float)
            for seed_id, seed_score in seeds:
                for syn in self.synapses_of(seed_id)[:10]:
                    other = syn.other(seed_id)
                    gain = self.SPREAD_FACTOR * seed_score * syn.weight
                    if gain > 0.01:
                        gains[other] += gain
                        via[other].append(seed_id)
            for cid, gain in gains.items():
                scores[cid] += min(gain, self.MAX_SPREAD_GAIN)  # a neighbour can be woken, not outrank a direct hit

        kinds = {r["id"]: r["kind"] for r in self.db.execute("SELECT id, kind FROM cells")} if scores else {}
        weighted = {cid: sc * self.KIND_TRUST.get(kinds.get(cid, ""), 0.8) for cid, sc in scores.items()}
        top = sorted(weighted.items(), key=lambda kv: -kv[1])[:k]
        results = [Recall(cell=self.get(cid), score=round(score, 4), via=via.get(cid, [])) for cid, score in top]  # type: ignore[arg-type]
        results = [r for r in results if r.cell is not None]

        if reinforce and results:
            self._reinforce([r.cell.id for r in results])
            self._journal("recall", query[:120])
            self.db.commit()
        return results

    def _reinforce(self, cell_ids: list[str]) -> None:
        """Hebbian update: cells recalled together strengthen their links."""
        placeholders = ",".join("?" * len(cell_ids))
        self.db.execute(f"UPDATE cells SET activations = activations + 1 WHERE id IN ({placeholders})", cell_ids)
        for i, a in enumerate(cell_ids):
            for b in cell_ids[i + 1 : i + 4]:  # only the nearest neighbours in the ranking wire together
                existing = self._get_synapse(a, b)
                if existing is not None:
                    self._upsert_synapse(a, b, min(1.0, existing.weight + self.HEBBIAN_STEP), existing.reason)
                elif i == 0:
                    self._upsert_synapse(a, b, 0.1, "recalled together")

    # ------------------------------------------------------------------ query
    def get(self, cell_id: str) -> Cell | None:
        row = self.db.execute("SELECT * FROM cells WHERE id = ?", (cell_id,)).fetchone()
        return self._row_to_cell(row) if row else None

    def find(self, title_fragment: str) -> list[Cell]:
        rows = self.db.execute(
            "SELECT * FROM cells WHERE title LIKE ? ORDER BY created_at", (f"%{title_fragment}%",)
        ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def cells(self, kind: str | None = None) -> list[Cell]:
        if kind:
            rows = self.db.execute("SELECT * FROM cells WHERE kind = ? ORDER BY created_at", (kind,)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM cells ORDER BY created_at").fetchall()
        return [self._row_to_cell(r) for r in rows]

    def synapses_of(self, cell_id: str) -> list[Synapse]:
        rows = self.db.execute(
            "SELECT * FROM synapses WHERE a = ? OR b = ? ORDER BY weight DESC", (cell_id, cell_id)
        ).fetchall()
        return [Synapse(r["a"], r["b"], r["weight"], r["reason"]) for r in rows]

    def neighbors(self, cell_id: str, limit: int = 10) -> list[tuple[Cell, Synapse]]:
        out = []
        for syn in self.synapses_of(cell_id)[:limit]:
            other = self.get(syn.other(cell_id))
            if other is not None:
                out.append((other, syn))
        return out

    def count_cells(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM cells").fetchone()[0]

    def count_synapses(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]

    def stats(self) -> dict:
        kinds = {r["kind"]: r["n"] for r in self.db.execute("SELECT kind, COUNT(*) AS n FROM cells GROUP BY kind")}
        concepts = {
            r["concept"]: r["n"]
            for r in self.db.execute(
                "SELECT concept, COUNT(*) AS n FROM cell_concepts GROUP BY concept ORDER BY n DESC LIMIT 15"
            )
        }
        hubs = [
            (self.get(r["id"]), r["degree"])
            for r in self.db.execute(
                "SELECT c.id, (SELECT COUNT(*) FROM synapses s WHERE s.a = c.id OR s.b = c.id) AS degree"
                " FROM cells c ORDER BY degree DESC LIMIT 5"
            )
        ]
        avg_w = self.db.execute("SELECT AVG(weight) FROM synapses").fetchone()[0] or 0.0
        return {
            "cells": self.count_cells(),
            "synapses": self.count_synapses(),
            "avg_synapse_weight": round(avg_w, 3),
            "kinds": kinds,
            "top_concepts": concepts,
            "hubs": [(c.title, d) for c, d in hubs if c is not None],
        }

    def journal(self, limit: int = 20) -> list[tuple[float, str, str]]:
        rows = self.db.execute("SELECT ts, event, detail FROM journal ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [(r["ts"], r["event"], r["detail"]) for r in rows]

    # ----------------------------------------------------------------- export
    def export_graph(self) -> dict:
        """Node/edge JSON, ready for any graph visualiser."""
        nodes = [
            {"id": c.id, "kind": c.kind, "title": c.title, "concepts": c.concepts, "activations": c.activations}
            for c in self.cells()
        ]
        edges = [
            {"source": r["a"], "target": r["b"], "weight": r["weight"], "reason": r["reason"]}
            for r in self.db.execute("SELECT * FROM synapses")
        ]
        return {"nodes": nodes, "edges": edges}

    def export_dot(self) -> str:
        """Graphviz DOT text, for ``dot -Tsvg``."""
        lines = ["graph brain {", "  node [shape=box, style=rounded, fontsize=10];"]
        for c in self.cells():
            label = c.title.replace('"', "'")[:48]
            lines.append(f'  "{c.id}" [label="{label}\\n({c.kind})"];')
        for r in self.db.execute("SELECT * FROM synapses"):
            lines.append(f'  "{r["a"]}" -- "{r["b"]}" [penwidth={0.5 + 3 * r["weight"]:.2f}];')
        lines.append("}")
        return "\n".join(lines)

    def forget(self, source: str | None = None, kind: str | None = None, title: str | None = None) -> int:
        """Remove cells matching a source substring, a kind, and/or a title substring. Returns cells removed."""
        clauses, params = [], []
        if source:
            clauses.append("source LIKE ?"); params.append(f"%{source}%")
        if kind:
            clauses.append("kind = ?"); params.append(kind)
        if title:
            clauses.append("title LIKE ?"); params.append(f"%{title}%")
        if not clauses:
            raise ValueError("forget needs a source, kind or title to match")
        where = " AND ".join(clauses)
        ids = [r["id"] for r in self.db.execute(f"SELECT id FROM cells WHERE {where}", params)]
        for i in range(0, len(ids), self._CHUNK):
            chunk = ids[i : i + self._CHUNK]
            ph = ",".join("?" * len(chunk))
            self.db.execute(f"DELETE FROM synapses WHERE a IN ({ph}) OR b IN ({ph})", chunk + chunk)
            self.db.execute(f"DELETE FROM cell_concepts WHERE cell_id IN ({ph})", chunk)
            self.db.execute(f"DELETE FROM cell_tokens WHERE cell_id IN ({ph})", chunk)
            self.db.execute(f"DELETE FROM cells WHERE id IN ({ph})", chunk)
        self._journal("forget", f"{len(ids)} cells ({where})")
        self.db.commit()
        return len(ids)

    # ------------------------------------------------------------ state
    def get_state(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_state(self, key: str, value) -> None:
        self.db.execute(
            "INSERT INTO state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # --------------------------------------------------------------- internal
    def _get_synapse(self, a: str, b: str) -> Synapse | None:
        a, b = sorted((a, b))
        row = self.db.execute("SELECT * FROM synapses WHERE a = ? AND b = ?", (a, b)).fetchone()
        return Synapse(row["a"], row["b"], row["weight"], row["reason"]) if row else None

    def _upsert_synapse(self, a: str, b: str, weight: float, reason: str) -> Synapse:
        a, b = sorted((a, b))
        self.db.execute(
            "INSERT INTO synapses (a, b, weight, reason) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(a, b) DO UPDATE SET weight = excluded.weight, reason = excluded.reason",
            (a, b, weight, reason),
        )
        return Synapse(a, b, weight, reason)

    _CHUNK = 500  # SQLite bound-parameter safety

    def _doc_freq(self, terms: Iterable[str]) -> dict[str, int]:
        terms = list(dict.fromkeys(terms))
        out: dict[str, int] = {}
        for i in range(0, len(terms), self._CHUNK):
            chunk = terms[i : i + self._CHUNK]
            placeholders = ",".join("?" * len(chunk))
            for r in self.db.execute(
                f"SELECT token, COUNT(*) AS n FROM cell_tokens WHERE token IN ({placeholders}) GROUP BY token", chunk
            ):
                out[r["token"]] = r["n"]
        return out

    def _concept_counts(self, cell_ids: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(0, len(cell_ids), self._CHUNK):
            chunk = cell_ids[i : i + self._CHUNK]
            placeholders = ",".join("?" * len(chunk))
            for r in self.db.execute(
                f"SELECT cell_id, COUNT(*) AS n FROM cell_concepts WHERE cell_id IN ({placeholders}) GROUP BY cell_id", chunk
            ):
                out[r["cell_id"]] = r["n"]
        return out

    def _norms(self, cell_ids: Iterable[str], n_cells: int) -> dict[str, float]:
        """tf-idf vector norms for the given cells, loaded in two batched queries."""
        cell_ids = list(cell_ids)
        token_maps: dict[str, dict[str, int]] = {}
        for i in range(0, len(cell_ids), self._CHUNK):
            chunk = cell_ids[i : i + self._CHUNK]
            placeholders = ",".join("?" * len(chunk))
            for r in self.db.execute(f"SELECT id, tokens FROM cells WHERE id IN ({placeholders})", chunk):
                token_maps[r["id"]] = json.loads(r["tokens"])
        df = self._doc_freq(t for toks in token_maps.values() for t in toks)
        return {
            cid: math.sqrt(sum((tf * self._idf(df.get(t, 0), n_cells)) ** 2 for t, tf in toks.items())) or 1.0
            for cid, toks in token_maps.items()
        }

    @staticmethod
    def _idf(df: int, n: int) -> float:
        return math.log((1 + n) / (1 + df)) + 1.0

    def _journal(self, event: str, detail: str) -> None:
        self.db.execute("INSERT INTO journal (ts, event, detail) VALUES (?, ?, ?)", (time.time(), event, detail))

    @staticmethod
    def _row_to_cell(row: sqlite3.Row) -> Cell:
        return Cell(
            id=row["id"],
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            source=row["source"],
            concepts=json.loads(row["concepts"]),
            created_at=row["created_at"],
            activations=row["activations"],
        )
