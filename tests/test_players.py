"""Career aggregation, position assignment and the eligibility filters."""

from __future__ import annotations

import csv

import pytest

from fish_team import players

BAT_HEADER = ["name_common", "player_ID", "year_ID", "WAR"]
PITCH_HEADER = ["name_common", "player_ID", "year_ID", "WAR", "G", "GS"]


def _write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_career_war_sums_batting_and_pitching(tmp_path):
    """A two-way player is credited for both halves of his value."""
    bat = _write(
        tmp_path / "bat.txt",
        BAT_HEADER,
        [
            ["Shohei Ohtani", "ohtansh01", "2021", "4.9"],
            ["Shohei Ohtani", "ohtansh01", "2022", "3.9"],
        ],
    )
    pitch = _write(
        tmp_path / "pitch.txt",
        PITCH_HEADER,
        [
            ["Shohei Ohtani", "ohtansh01", "2021", "4.1", "23", "23"],
            ["Shohei Ohtani", "ohtansh01", "2022", "5.6", "28", "28"],
        ],
    )

    careers: dict[str, players._Career] = {}
    assert players._accumulate_war(bat, careers, pitching=False) == 2
    assert players._accumulate_war(pitch, careers, pitching=True) == 2

    career = careers["ohtansh01"]
    assert career.war == pytest.approx(18.5)
    assert career.years == {2021, 2022}
    assert career.debut == 2021
    assert career.games_pitched == 51
    assert career.games_started == 51


def test_seasons_count_distinct_years_not_rows(tmp_path):
    """A mid-season trade gives a player two rows in one year, not two seasons."""
    bat = _write(
        tmp_path / "bat.txt",
        BAT_HEADER,
        [
            ["Traded Guy", "tradeg01", "1998", "1.0"],
            ["Traded Guy", "tradeg01", "1998", "0.5"],
            ["Traded Guy", "tradeg01", "1999", "2.0"],
        ],
    )
    careers: dict[str, players._Career] = {}
    players._accumulate_war(bat, careers, pitching=False)
    assert len(careers["tradeg01"].years) == 2
    assert careers["tradeg01"].war == pytest.approx(3.5)


def test_malformed_numbers_do_not_abort_the_parse(tmp_path):
    """bbref writes NULL in numeric columns; those rows still count as a season."""
    bat = _write(
        tmp_path / "bat.txt",
        BAT_HEADER,
        [["Old Timer", "oldtio01", "1954", "NULL"]],
    )
    careers: dict[str, players._Career] = {}
    players._accumulate_war(bat, careers, pitching=False)
    assert careers["oldtio01"].war == 0.0
    assert careers["oldtio01"].years == {1954}


@pytest.mark.parametrize(
    ("games", "games_pitched", "games_started", "expected"),
    [
        ({"P": 400}, 400, 350, "SP"),          # clear starter
        ({"P": 700}, 700, 10, "RP"),           # clear reliever
        ({"P": 100}, 100, 50, "SP"),           # exactly at the 50% boundary
        ({"P": 100}, 100, 49, "RP"),           # just under it
        ({"C": 900, "1B": 200}, 0, 0, "C"),    # most games wins
        ({"1B": 200, "C": 199}, 0, 0, "1B"),
        ({"DH": 500, "1B": 400}, 0, 0, "DH"),
        # A pitcher who also batted: still a pitcher, because G_p leads.
        ({"P": 300, "1B": 10}, 300, 290, "SP"),
    ],
)
def test_primary_position(games, games_pitched, games_started, expected):
    career = players._Career(
        games_pitched=games_pitched, games_started=games_started
    )
    assert players._primary_position(games, career) == expected


def test_primary_position_is_none_without_appearances():
    assert players._primary_position({}, players._Career()) is None
    assert players._primary_position({"C": 0}, players._Career()) is None


def test_name_common_fallback_split():
    assert players._split_name_common("Mike Trout") == ("Mike", "Trout")
    assert players._split_name_common("Jose de la Rosa") == ("Jose de la", "Rosa")
    assert players._split_name_common("Ichiro") == ("", "Ichiro")
    assert players._split_name_common("") == ("", "")


def test_players_round_trip(tmp_path):
    roster = [
        players.Player("troutmi01", "Mike", "Trout", 16, 2011, 91.7, "CF", ("CF", "LF")),
        players.Player("benchjo01", "Johnny", "Bench", 17, 1967, 75.1, "C", ("C",)),
    ]
    path = players.write_players(roster, tmp_path / "players.csv")
    assert players.read_players(path) == roster


def test_players_csv_without_positions_column_still_reads(tmp_path):
    """A players.csv from before multi-position eligibility falls back to the primary."""
    path = _write(
        tmp_path / "players.csv",
        ["bbref_id", "first", "last", "seasons", "debut", "war", "position"],
        [["salmoti01", "Tim", "Salmon", 14, 1992, 40.6, "RF"]],
    )
    (salmon,) = players.read_players(path)
    assert salmon.slots == ("RF",)


def _eligible(seasons, games_pitched=0, games_started=0, min_games=100):
    career = players._Career(games_pitched=games_pitched, games_started=games_started)
    return players._eligible_positions(seasons, career, min_games=min_games)


def test_eligible_by_career_games():
    # Mike Trout: 124 games in LF across two seasons, never his main position in either.
    seasons = {2012: {"CF": 110, "LF": 67}, 2013: {"CF": 150, "LF": 57}, 2014: {"CF": 157, "DH": 9}}
    assert _eligible(seasons) == ("CF", "LF")


def test_eligible_by_one_season_as_main_position():
    # Mel Roach: 28 career games at 3B, but 3B was his main position in 1962.
    seasons = {1958: {"2B": 27, "LF": 4}, 1962: {"2B": 9, "3B": 26, "1B": 4}}
    assert _eligible(seasons) == ("2B", "3B")


def test_not_eligible_below_both_rules():
    seasons = {1964: {"RF": 52, "2B": 32, "3B": 10}, 1965: {"1B": 29, "3B": 24}}
    assert "3B" not in _eligible(seasons)


def test_position_player_never_pitches():
    # Sam Rice pitched in 1915 before becoming a right fielder.
    seasons = {1915: {"P": 4, "RF": 1}, 1916: {"P": 5, "RF": 3}, 1920: {"RF": 153}}
    assert _eligible(seasons, games_pitched=9, games_started=0) == ("RF",)


def test_pitcher_keeps_single_role_and_can_gain_field_slots():
    seasons = {1918: {"P": 20, "LF": 59}, 1919: {"P": 17, "LF": 111}, 1920: {"RF": 141}}
    assert _eligible(seasons, games_pitched=37, games_started=30, min_games=100)[0] == "LF"
    pitcher = {2001: {"P": 34}, 2002: {"P": 33}}
    assert _eligible(pitcher, games_pitched=67, games_started=67) == ("SP",)


def test_no_appearances_means_not_eligible():
    assert _eligible({}) == ()


def test_full_name_property():
    player = players.Player("salmoti01", "Tim", "Salmon", 14, 1992, 40.6, "RF")
    assert player.full_name == "Tim Salmon"
