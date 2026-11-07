"""Fishiness scoring and blended selection, on hand-built vectors."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from fish_team import embedding, roster
from fish_team.players import Player

# Three dimensions: fish, food, person-name.
_FISH = np.array([1.0, 0.0, 0.0])
_FOOD = np.array([0.0, 1.0, 0.0])
_NAME = np.array([0.0, 0.0, 1.0])


def _vectors() -> dict[str, np.ndarray]:
    vectors = {w: _FISH + 0.1 * _FOOD for w in embedding.SEED_FISH}
    vectors["trout"] = _FISH + 0.05 * _NAME
    vectors["rice"] = _FOOD + 0.2 * _FISH
    vectors["dempsey"] = _NAME
    return vectors


@pytest.fixture
def axis() -> embedding.FishAxis:
    return embedding.FishAxis(_vectors())


def _player(last, war, position="CF"):
    return Player(f"{last.lower()}01", "A", last, 10, 1980, war, position)


def test_fish_beats_food_beats_name(axis):
    trout, rice, dempsey = (axis.score(n) for n in ("Trout", "Rice", "Dempsey"))
    assert trout.score > rice.score > dempsey.score
    assert dempsey.score < 0.05


def test_unknown_surname_is_none(axis):
    assert axis.score("Yastrzemski") is None


def test_surname_is_normalised_before_lookup(axis):
    assert axis.score("Trout Jr.") == axis.score("trout")


def test_seed_surname_is_scored_leave_one_out(axis):
    """Trout's own vector must not be part of the centroid he is measured against."""
    trout = axis.score("Trout")
    assert "trout" not in trout.nearest
    assert trout.score < 1.0


def test_floor_beats_war(axis):
    """A huge-WAR non-fish cannot win a slot, whatever the WAR weight."""
    scored = roster.score_players(
        [_player("Trout", 10.0), _player("Dempsey", 100.0)], axis, war_weight=10.0
    )
    selections = roster.select_blended(scored, min_fishiness=0.3)
    assert selections["CF"].pick.player.last == "Trout"


def test_war_weight_breaks_fishiness(axis):
    """Above the floor, enough WAR weight lets a less fishy player win."""
    players = [_player("Trout", 1.0), _player("Rice", 80.0)]
    fishy_first = roster.select_blended(roster.score_players(players, axis, war_weight=0.0), min_fishiness=0.1)
    war_heavy = roster.select_blended(roster.score_players(players, axis, war_weight=5.0), min_fishiness=0.1)
    assert fishy_first["CF"].pick.player.last == "Trout"
    assert war_heavy["CF"].pick.player.last == "Rice"


def test_empty_slot_is_unfilled(axis):
    selections = roster.select_blended(roster.score_players([_player("Dempsey", 5.0)], axis))
    assert not selections["CF"].filled


def test_load_vectors_reads_only_wanted_words(tmp_path):
    path = tmp_path / "vectors.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("3 2\n")  # gensim-style header line, which must be skipped harmlessly
        fh.write("trout 1.0 0.0\nrice 0.0 1.0\nfox 0.5 0.5\n")
    found = embedding.load_vectors({"trout", "fox", "absent"}, path)
    assert set(found) == {"trout", "fox"}
    np.testing.assert_allclose(found["fox"], [0.5, 0.5])
