"""Exact-first tier selection."""

from __future__ import annotations

from fish_team import roster
from fish_team.matching import Match
from fish_team.players import Player


def _candidate(last, war, position, head, score, tier):
    return roster.Candidate(
        player=Player(f"{last.lower()}01", "A", last, 10, 1980, war, position),
        match=Match(
            surname=last.lower(),
            head=head,
            score=score,
            tier=tier,
            example=head.title(),
            n_species=1,
        ),
    )


def test_better_tier_beats_higher_war():
    """The whole point of exact-first: a perfect match outranks more WAR."""
    pool = {
        "C": [
            _candidate("Dempsey", 25.2, "C", "dempsey", 1.0, "exact"),
            _candidate("Bench", 75.1, "C", "tench", 0.83, "loose"),
        ]
    }
    selections = roster.select_roster(pool)
    chosen = selections["C"]
    assert chosen.pick is not None
    assert chosen.pick.player.last == "Dempsey"
    assert chosen.tier_used == "exact"
    # ...and the cost of that choice is reported rather than hidden.
    assert chosen.war_max is not None
    assert chosen.war_max.player.last == "Bench"
    assert chosen.war_forgone == 49.9


def test_highest_war_wins_within_a_tier():
    pool = {
        "CF": [
            _candidate("Trout", 91.7, "CF", "trout", 1.0, "exact"),
            _candidate("Pollock", 22.1, "CF", "pollock", 1.0, "exact"),
        ]
    }
    chosen = roster.select_roster(pool)["CF"]
    assert chosen.pick is not None
    assert chosen.pick.player.last == "Trout"
    assert chosen.war_forgone == 0.0


def test_min_tier_can_leave_a_slot_unfilled():
    pool = {"DH": [_candidate("Hale", 1.3, "DH", "whale", 0.85, "close")]}

    assert roster.select_roster(pool, min_tier="close")["DH"].filled
    strict = roster.select_roster(pool, min_tier="exact")["DH"]
    assert not strict.filled
    assert strict.pick is None
    assert strict.tier_used is None


def test_every_slot_is_present_even_when_empty():
    selections = roster.select_roster({})
    assert set(selections) == set(roster.SLOTS)
    assert not any(s.filled for s in selections.values())
    assert roster.total_war(selections) == 0.0


def test_total_war_sums_only_the_picks():
    pool = {
        "CF": [_candidate("Trout", 91.7, "CF", "trout", 1.0, "exact")],
        "LF": [_candidate("Rice", 47.7, "LF", "rice", 1.0, "exact")],
    }
    assert roster.total_war(roster.select_roster(pool)) == 139.4


def test_alternates_are_ranked_by_war_and_capped():
    pool = {
        "RF": [
            _candidate(f"Fish{i}", float(i), "RF", "bass", 1.0, "exact")
            for i in range(10)
        ]
    }
    alternates = roster.select_roster(pool)["RF"].alternates
    assert len(alternates) == roster.ALTERNATES_SHOWN
    assert [c.war for c in alternates] == [9.0, 8.0, 7.0, 6.0, 5.0]


def test_collect_candidates_respects_primary_position(monkeypatch):
    class FakeIndex:
        def best_match(self, surname, *, min_tier):
            return Match(surname.lower(), "trout", 1.0, "exact", "Trout", 1)

    squad = [
        Player("a01", "Mike", "Trout", 16, 2011, 91.7, "CF"),
        Player("b01", "Steve", "Trout", 10, 1978, 7.0, "SP"),
        # A position outside SLOTS is ignored rather than crashing.
        Player("c01", "Odd", "Trout", 8, 1990, 3.0, "PH"),
    ]
    grouped = roster.collect_candidates(squad, FakeIndex())
    assert [c.player.bbref_id for c in grouped["CF"]] == ["a01"]
    assert [c.player.bbref_id for c in grouped["SP"]] == ["b01"]
    assert sum(len(v) for v in grouped.values()) == 2


def test_assignment_never_uses_a_player_twice():
    """Trout is eligible at CF and LF; he can only play one, and the other slot must
    fall to the next best player rather than leaving him in both."""
    trout = roster.Candidate(
        player=Player("troutmi01", "Mike", "Trout", 16, 2011, 91.7, "CF", ("CF", "LF")),
        match=Match("trout", "trout", 1.0, "exact", "Trout", 1),
    )
    rice = _candidate("Rice", 47.7, "LF", "rice", 1.0, "exact")
    selections = roster.select_roster({"CF": [trout], "LF": [trout, rice]})
    assert selections["CF"].pick is trout
    assert selections["LF"].pick is rice


def test_assignment_moves_a_player_to_free_a_better_match():
    """Greedy slot-by-slot would give CF to the exact match and leave LF empty; the
    assignment instead fills both."""
    flexible = roster.Candidate(
        player=Player("a01", "A", "Salmon", 10, 1990, 40.0, "CF", ("CF", "LF")),
        match=Match("salmon", "salmon", 1.0, "exact", "Salmon", 1),
    )
    cf_only = _candidate("Bass", 10.0, "CF", "bass", 0.9, "near")
    selections = roster.select_roster({"CF": [flexible, cf_only], "LF": [flexible]})
    assert selections["CF"].pick is cf_only
    assert selections["LF"].pick is flexible


def test_collect_candidates_uses_every_eligible_slot():
    class FakeIndex:
        def best_match(self, surname, *, min_tier):
            return Match(surname.lower(), "trout", 1.0, "exact", "Trout", 1)

    squad = [Player("a01", "Mike", "Trout", 16, 2011, 91.7, "CF", ("CF", "LF", "DH"))]
    grouped = roster.collect_candidates(squad, FakeIndex())
    assert {slot for slot, pool in grouped.items() if pool} == {"CF", "LF", "DH"}
