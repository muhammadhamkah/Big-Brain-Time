"""Big Brain Time: a self-linking trading knowledge brain.

The brain is made of *cells* (units of knowledge) joined by *synapses*
(weighted links). Every time it learns something new, the new cell is wired
to every existing cell it relates to, so knowledge compounds instead of
piling up. Recall spreads activation across those links, and cells that are
recalled together grow stronger connections, the way biological neurons do.
"""

from bigbrain.brain import Brain
from bigbrain.cells import Cell, Synapse

__all__ = ["Brain", "Cell", "Synapse"]
__version__ = "0.1.0"
