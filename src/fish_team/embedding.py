"""Score surnames by how strongly they *mean* fish, using static word embeddings.

The spelling matcher in ``matching.py`` asks whether a surname looks like a fish name.
This asks a different question: in a GloVe space trained on Wikipedia and newswire, how
close is the surname to the average of common fish words?

Three choices decide whether that works:

1. **The surname alone, lowercased, in a static model.** A contextual embedding of
   "Mike Trout" encodes the ballplayer; the static vector for ``trout`` is mostly the
   fish. Surnames that are only ever names (``dempsey``, ``weaver``) sit with other names
   and score low, which is the right answer.
2. **A fish centroid, not the word "fish".** The neighbours of ``fish`` are food and
   water words, so Jim Rice ranks 10th against it. Against the mean of forty common fish
   words he falls to ~870th, and only real fish names stay at the top.
   Subtracting a food/animal direction as well was tried and made things worse:
   rare surnames have noisy vectors, and enough of them line up with that difference
   by chance to crowd out Salmon and Bass.
3. **No subword fallback.** fastText-style subwords would score an unknown surname by
   its spelling, which is the other matcher's job. Unknown surnames score ``None``.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sources
from .matching import normalize_surname

log = logging.getLogger(__name__)

# Common fish with unambiguous, frequent English names. Each one is in GloVe's
# vocabulary and its nearest neighbours are other fish.
SEED_FISH: tuple[str, ...] = (
    "trout", "salmon", "bass", "pike", "carp", "cod", "haddock", "halibut", "herring",
    "mackerel", "sardine", "anchovy", "tuna", "perch", "walleye", "catfish", "sturgeon",
    "flounder", "sole", "snapper", "grouper", "minnow", "eel", "shad", "smelt", "tilapia",
    "marlin", "swordfish", "barracuda", "bluegill", "crappie", "pollock", "whitefish",
    "char", "bream", "tench", "roach", "chub", "dace", "goby",
)  # fmt: skip

NEAREST_SHOWN = 5
GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz"
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def load_vectors(words: set[str], path: Path | None = None) -> dict[str, np.ndarray]:
    """Read only the vectors for ``words`` from a gzipped GloVe text file.

    The full 400k x 300 matrix is ~500 MB as floats; a few thousand surnames is a few
    megabytes, so it is cheaper to stream the file than to load it.
    """
    path = path or sources.fetch(GLOVE_URL)
    found: dict[str, np.ndarray] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            word, _, rest = line.partition(" ")
            if word in words:
                found[word] = np.array(rest.split(), dtype=np.float32)
                if len(found) == len(words):
                    break
    log.info("embedding vectors found for %d of %d words", len(found), len(words))
    return found


@dataclass(frozen=True)
class Fishiness:
    """One surname's position on the fish axis."""

    surname: str
    score: float
    #: The seed fish whose vectors are closest to this surname, for explaining a pick.
    nearest: tuple[str, ...]


class FishAxis:
    """The direction of the fish centroid, and scoring against it."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors
        self.fish = [w for w in SEED_FISH if w in vectors]
        if len(self.fish) < 2:
            raise ValueError("too few seed fish found in the embedding vocabulary")
        self._fish_units = {w: _unit(vectors[w]) for w in self.fish}
        self._fish_sum = np.sum(list(self._fish_units.values()), axis=0)
        self.axis = _unit(self._fish_sum)

    def score(self, surname: str) -> Fishiness | None:
        """Cosine with the centroid, clipped at zero; ``None`` if the surname is unknown."""
        word = normalize_surname(surname)
        if word not in self.vectors:
            return None
        vector = _unit(self.vectors[word])
        axis = self.axis
        if word in self._fish_units:
            # Leave one out: Trout must earn his score against a centroid built without
            # ``trout``, or every seed surname would be graded on its own answer.
            axis = _unit(self._fish_sum - self._fish_units[word])
        nearest = sorted(
            (w for w in self.fish if w != word),
            key=lambda w: -float(vector @ self._fish_units[w]),
        )[:NEAREST_SHOWN]
        return Fishiness(
            surname=word,
            score=max(0.0, float(vector @ axis)),
            nearest=tuple(nearest),
        )


def build_axis(surnames: list[str], path: Path | None = None) -> FishAxis:
    """Load just the vectors needed to score ``surnames`` and build the axis."""
    wanted = {normalize_surname(s) for s in surnames} | set(SEED_FISH)
    wanted.discard("")
    return FishAxis(load_vectors(wanted, path))
