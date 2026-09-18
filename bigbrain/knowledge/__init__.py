"""Knowledge packs: written lessons the brain is seeded with, one module per domain.

Each pack is a module in this package that defines:

    DOMAIN = "Options and volatility trading"          # human name
    LESSONS: list[tuple[str, str]] = [(title, text), ...]  # 30-40 dense lessons
    SOURCES = {                                         # curated places to keep learning
        "feeds": ["https://.../feed/", ...],            # RSS/Atom URLs
        "subreddits": ["options", ...],
        "github_queries": ["topic:options-trading", ...],
        "arxiv_topics": ["implied volatility surface", ...],
        "urls": ["https://.../a-reference-page", ...],
    }

``packs()`` discovers every module here, so adding a pack is adding a file.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def packs() -> list[ModuleType]:
    found = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        if hasattr(mod, "LESSONS"):
            found.append(mod)
    return sorted(found, key=lambda m: getattr(m, "DOMAIN", m.__name__))


def all_lessons() -> list[tuple[str, str, str]]:
    """(domain, title, text) for every lesson in every pack."""
    out = []
    for mod in packs():
        domain = getattr(mod, "DOMAIN", mod.__name__)
        out.extend((domain, title, text) for title, text in mod.LESSONS)
    return out


def merged_sources() -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {"feeds": [], "subreddits": [], "github_queries": [], "arxiv_topics": [], "urls": []}
    for mod in packs():
        for key, values in getattr(mod, "SOURCES", {}).items():
            for v in values:
                if key in merged and v not in merged[key]:
                    merged[key].append(v)
    return merged
