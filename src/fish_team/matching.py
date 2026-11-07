"""Match surnames against fish head nouns.

Three layers:

1. **Normalisation** -- deaccent, lowercase, drop generational suffixes, so ``Rubén``
   and ``Griffey Jr.`` compare as ``ruben`` and ``griffey``.
2. **N-gram blocking** -- a character-trigram inverted index over the fish vocabulary,
   so each surname is scored against the few dozen plausible fish words rather than all
   1,158 of them.
3. **Scoring** -- a length-guarded blend of Levenshtein and Jaro-Winkler similarity,
   bucketed into named tiers.

On scoring: substring-tolerant scorers must not be used here. ``rapidfuzz.fuzz.WRatio``
and ``partial_ratio`` score a short fish word against any surname containing something
like it, which produced Al Kaline ~ "line" (0.87), Ozzie Smith ~ "saithe" (0.87), Curt
Schilling ~ "schilbid" (0.87) and Pedro Martinez ~ "marine" (0.86) in an early
prototype. Three of those four are held off by the scoring here and are regression
tested in ``tests/test_matching.py``; Martinez ~ "marine" would still clear the floor on
score alone and is excluded instead by keeping "marine" out of the vocabulary, which
``tests/test_fish.py`` covers.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz.distance import JaroWinkler, Levenshtein

from . import sources
from .fish import FishWord, deaccent

log = logging.getLogger(__name__)

try:  # optional: a phonetic tie-break, installed via the "phonetic" extra
    import jellyfish

    _HAVE_JELLYFISH = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAVE_JELLYFISH = False

NGRAM_SIZE = 3
# A fish word more than this many characters longer or shorter than the surname is not
# a plausible near-match, and skipping those is what keeps the index useful.
MAX_LENGTH_DIFF = 2

LEVENSHTEIN_WEIGHT = 0.6
JARO_WINKLER_WEIGHT = 0.4
# Equal metaphone codes nudge a pair up, but never to 1.0: an "exact" match must mean
# the strings are actually equal.
PHONETIC_BONUS = 0.03
PHONETIC_CEILING = 0.99

# Ordered best to worst. An exact match is string equality, not a score.
TIERS: tuple[tuple[str, float], ...] = (
    ("exact", 1.0),
    ("near", 0.90),
    ("close", 0.85),
    ("loose", 0.80),
    ("stretch", 0.75),
)
TIER_NAMES = tuple(name for name, _ in TIERS)
DEFAULT_MIN_TIER = "loose"

_GENERATIONAL_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_NON_NAME = re.compile(r"[^a-z'\-]+")


def normalize_surname(surname: str) -> str:
    """Fold a surname into the same alphabet the fish vocabulary uses."""
    text = _NON_NAME.sub(" ", deaccent(surname).lower().replace(".", " "))
    words = [w.strip("'-") for w in text.split() if w.strip("'-")]
    while len(words) > 1 and words[-1] in _GENERATIONAL_SUFFIXES:
        words.pop()
    if not words:
        return ""
    # Keep multi-word surnames joined: "de la rosa" -> "delarosa", which is how they
    # would have to be spelled to resemble a one-word fish name anyway.
    return "".join(words)


def ngrams(word: str, size: int = NGRAM_SIZE) -> set[str]:
    """Padded character n-grams, so short words still produce several grams."""
    padded = f"{'^' * (size - 1)}{word}{'$' * (size - 1)}"
    return {padded[i : i + size] for i in range(len(padded) - size + 1)}


def score(surname: str, fish_word: str) -> float:
    """Similarity in ``[0, 1]`` between a normalised surname and a fish head noun.

    Returns ``0.0`` when the lengths are too far apart to be a near-match, which also
    stops a short fish word from scoring well against a long surname.
    """
    if not surname or not fish_word:
        return 0.0
    if surname == fish_word:
        return 1.0
    if abs(len(surname) - len(fish_word)) > MAX_LENGTH_DIFF:
        return 0.0
    longest = max(len(surname), len(fish_word))
    levenshtein = 1.0 - Levenshtein.distance(surname, fish_word) / longest
    jaro_winkler = JaroWinkler.similarity(surname, fish_word)
    blended = LEVENSHTEIN_WEIGHT * levenshtein + JARO_WINKLER_WEIGHT * jaro_winkler
    if _HAVE_JELLYFISH and jellyfish.metaphone(surname) == jellyfish.metaphone(fish_word):
        blended = min(blended + PHONETIC_BONUS, PHONETIC_CEILING)
    return blended


def tier_for(value: float) -> str | None:
    """Name the tier a score falls in, or ``None`` if it misses every tier."""
    for name, threshold in TIERS:
        if value >= threshold:
            return name
    return None


def tier_rank(name: str) -> int:
    """Sort key where 0 is the best tier."""
    return TIER_NAMES.index(name)


def tiers_from(min_tier: str) -> tuple[str, ...]:
    """Tier names from ``exact`` down to ``min_tier`` inclusive."""
    if min_tier not in TIER_NAMES:
        raise ValueError(f"unknown tier {min_tier!r}; choose from {TIER_NAMES}")
    return TIER_NAMES[: tier_rank(min_tier) + 1]


@dataclass(frozen=True)
class Match:
    """One surname/fish-word pairing."""

    surname: str
    head: str
    score: float
    tier: str
    example: str
    n_species: int


class FishIndex:
    """Trigram index over the fish vocabulary, with scoring on top."""

    def __init__(self, vocab: dict[str, FishWord], *, ngram_size: int = NGRAM_SIZE):
        self.vocab = vocab
        self.ngram_size = ngram_size
        self.postings: dict[str, set[str]] = defaultdict(set)
        for head in vocab:
            for gram in ngrams(head, ngram_size):
                self.postings[gram].add(head)
        log.info(
            "fish index: %d head nouns, %d distinct %d-grams",
            len(vocab),
            len(self.postings),
            ngram_size,
        )

    def candidates(self, surname: str) -> set[str]:
        """Fish words sharing at least one n-gram with ``surname``, length-filtered."""
        found: set[str] = set()
        for gram in ngrams(surname, self.ngram_size):
            found.update(self.postings.get(gram, ()))
        return {h for h in found if abs(len(h) - len(surname)) <= MAX_LENGTH_DIFF}

    def best_matches(
        self, surname: str, *, min_tier: str = "stretch", limit: int | None = None
    ) -> list[Match]:
        """Every match for ``surname`` at ``min_tier`` or better, best first."""
        normalized = normalize_surname(surname)
        if not normalized:
            return []
        floor = dict(TIERS)[min_tier]
        allowed = set(tiers_from(min_tier))

        matches: list[Match] = []
        for head in self.candidates(normalized):
            value = score(normalized, head)
            if value < floor:
                continue
            tier = tier_for(value)
            if tier is None or tier not in allowed:
                continue
            word = self.vocab[head]
            matches.append(
                Match(
                    surname=normalized,
                    head=head,
                    score=value,
                    tier=tier,
                    example=word.example,
                    n_species=word.n_species,
                )
            )
        # Best score first; break ties toward the better-attested fish word.
        matches.sort(key=lambda m: (-m.score, -m.n_species, m.head))
        return matches[:limit] if limit else matches

    def best_match(self, surname: str, *, min_tier: str = "stretch") -> Match | None:
        matches = self.best_matches(surname, min_tier=min_tier, limit=1)
        return matches[0] if matches else None

    def save(self, path: Path | None = None) -> Path:
        """Persist the postings list so a rebuild need not re-index."""
        path = path or sources.INDEX_DIR / f"fish_{self.ngram_size}grams.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "ngram_size": self.ngram_size,
                    "postings": {g: sorted(h) for g, h in sorted(self.postings.items())},
                },
            )
        )
        log.info("wrote %s", path)
        return path
