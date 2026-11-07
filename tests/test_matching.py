"""Surname matching: normalisation, blocking, scoring and tiers."""

from __future__ import annotations

import pytest

from fish_team import matching
from fish_team.fish import FishWord

# A vocabulary holding both the fish words we want to hit and the ones an earlier
# prototype hit by mistake.
VOCAB_WORDS = {
    "trout": "Trout",
    "tench": "Tench",
    "weever": "Weever",
    "sucker": "Sucker",
    "salmon": "Salmon",
    "fox": "Flying fox",
    "coney": "Coney",
    "escolar": "Escolar",
    "bass": "Rock bass",
    "rice": "Japanese rice fish",
    "betta": "Betta",
    # Near-misses that must stay misses.
    "saithe": "Saithe",
    "cline": "Cline",
    "schilbid": "Glass schilbid",
    "manta": "Manta ray",
    "marlin": "Black marlin",
}


@pytest.fixture(scope="module")
def index():
    vocab = {
        head: FishWord(head=head, n_species=1, example=example, sources={"asfis"})
        for head, example in VOCAB_WORDS.items()
    }
    return matching.FishIndex(vocab)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Trout", "trout"),
        ("Rubén", "ruben"),
        ("Griffey Jr.", "griffey"),
        ("Ken Griffey Sr", "kengriffey"),
        ("de la Rosa", "delarosa"),
        ("O'Neill", "o'neill"),
        ("", ""),
    ],
)
def test_normalize_surname(raw, expected):
    assert matching.normalize_surname(raw) == expected


def test_ngrams_pad_short_words():
    assert matching.ngrams("fox") == {"^^f", "^fo", "fox", "ox$", "x$$"}


def test_identical_strings_score_one():
    assert matching.score("trout", "trout") == 1.0
    assert matching.tier_for(matching.score("trout", "trout")) == "exact"


def test_length_guard_rejects_distant_lengths():
    """A fish word far shorter than the surname is refused outright."""
    assert matching.score("rodriguez", "ray") == 0.0
    assert matching.score("yastrzemski", "ide") == 0.0
    assert matching.score("bass", "barracuda") == 0.0


def test_substring_resemblance_alone_does_not_reach_the_default_floor():
    """``fuzz.partial_ratio`` scored these pairs ~0.87 by finding one inside the other.

    The replacement scorer leaves them in ``stretch``, below the default ``loose``
    floor, because it charges for every character that does not line up.
    """
    for surname, head in (("kaline", "line"), ("schilling", "schilbid"), ("mantle", "manta")):
        value = matching.score(surname, head)
        assert matching.tier_for(value) == "stretch", (surname, head, value)


@pytest.mark.parametrize(
    ("surname", "head", "tier"),
    [
        ("Trout", "trout", "exact"),
        ("Salmon", "salmon", "exact"),
        ("Fox", "fox", "exact"),
        ("Bench", "tench", "loose"),
        ("Weaver", "weever", "close"),
        ("Tucker", "sucker", "close"),
        ("Cone", "coney", "close"),
        ("Escobar", "escolar", "close"),
        ("Betts", "betta", "loose"),
    ],
)
def test_expected_matches(index, surname, head, tier):
    match = index.best_match(surname)
    assert match is not None, f"{surname} found no match"
    assert match.head == head
    assert match.tier == tier


@pytest.mark.parametrize(
    "surname",
    [
        # Each of these was a false positive from rapidfuzz's WRatio/partial_ratio:
        # Smith ~ saithe (0.87), Kaline ~ line (0.87), Martinez ~ marine (0.86),
        # Schilling ~ schilbid (0.87).
        "Smith",
        "Kaline",
        "Martinez",
        "Schilling",
        "Mantle",
        "Rodriguez",
        "Bonds",
    ],
)
def test_prototype_false_positives_stay_below_the_default_floor(index, surname):
    match = index.best_match(surname, min_tier=matching.DEFAULT_MIN_TIER)
    assert match is None, f"{surname} matched {match.head if match else None}"


def test_blocking_returns_a_small_candidate_set(index):
    candidates = index.candidates("trout")
    assert "trout" in candidates
    # Length-filtered, so nothing wildly longer or shorter survives.
    assert all(abs(len(c) - len("trout")) <= matching.MAX_LENGTH_DIFF for c in candidates)
    assert len(candidates) < len(VOCAB_WORDS)


def test_blocking_never_hides_a_real_match(index):
    """Every scoring match must also be reachable through the n-gram index."""
    for surname in ("Bench", "Weaver", "Tucker", "Cone", "Escobar", "Betts"):
        normalized = matching.normalize_surname(surname)
        reachable = index.candidates(normalized)
        brute_force = {
            head
            for head in index.vocab
            if matching.tier_for(matching.score(normalized, head)) is not None
        }
        assert brute_force <= reachable, surname


def test_phonetic_bonus_cannot_manufacture_an_exact_match():
    value = matching.score("bench", "tench")
    assert value < 1.0
    assert matching.tier_for(value) != "exact"


def test_tier_helpers():
    assert matching.tier_for(0.5) is None
    assert matching.tiers_from("close") == ("exact", "near", "close")
    assert matching.tier_rank("exact") < matching.tier_rank("stretch")
    with pytest.raises(ValueError):
        matching.tiers_from("perfect")


def test_matches_are_ordered_best_first(index):
    matches = index.best_matches("Bass", min_tier="stretch")
    assert matches[0].head == "bass"
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)
