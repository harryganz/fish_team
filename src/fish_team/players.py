"""Build the table of eligible players: name, primary position and career WAR.

Three sources are joined on the Baseball Reference player id:

* the two bbref ``war_daily_*`` files, for career WAR, seasons played and the
  games/games-started counts that separate starters from relievers;
* the Chadwick Bureau register, for a clean first/last name split;
* Lahman's ``Appearances``, for games played at each position.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import sources

log = logging.getLogger(__name__)

DEFAULT_MIN_SEASONS = 6
# The 1903 National Agreement, when the American and National Leagues made peace and
# played the first World Series. Players whose first season precedes it are excluded.
#
# "First season" is bbref's, which counts the Negro Leagues as major leagues from 1920:
# Jackie Robinson's is 1945 (Kansas City Monarchs), so the previous cutoff of 1947,
# meant to mark integration, quietly excluded Robinson himself.
DEFAULT_SINCE = 1903

# Appearances column -> slot label. Order matters only for deterministic tie-breaking.
POSITION_COLUMNS = (
    ("G_p", "P"),
    ("G_c", "C"),
    ("G_1b", "1B"),
    ("G_2b", "2B"),
    ("G_3b", "3B"),
    ("G_ss", "SS"),
    ("G_lf", "LF"),
    ("G_cf", "CF"),
    ("G_rf", "RF"),
    ("G_dh", "DH"),
)

# A player is eligible at any position where he has at least this many career games,
# roughly a season's worth for an everyday player (two thirds of 154 or 162), ...
DEFAULT_MIN_POSITION_GAMES = 100
# ...or which was his most-played position in at least one season.

# A pitcher counts as a starter when at least this share of his appearances were starts.
STARTER_GS_SHARE = 0.5


@dataclass
class Player:
    bbref_id: str
    first: str
    last: str
    seasons: int
    debut: int
    war: float
    position: str
    #: Every slot he is eligible for, primary position first.
    positions: tuple[str, ...] = ()

    @property
    def slots(self) -> tuple[str, ...]:
        return self.positions or (self.position,)

    @property
    def full_name(self) -> str:
        return f"{self.first} {self.last}".strip()


def _as_float(value: str | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: str | None) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


@dataclass
class _Career:
    name_common: str = ""
    war: float = 0.0
    debut: int = 9999
    games_pitched: int = 0
    games_started: int = 0
    years: set[int] = field(default_factory=set)


def _accumulate_war(path: Path, careers: dict[str, _Career], *, pitching: bool) -> int:
    """Fold one ``war_daily_*`` file into ``careers``. Returns rows read."""
    rows = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            career = careers.setdefault(row["player_ID"], _Career())
            # Career bWAR is batting WAR plus pitching WAR, so two-way players such as
            # Ohtani are credited for both halves of their value.
            career.war += _as_float(row.get("WAR"))
            career.years.add(_as_int(row.get("year_ID")))
            career.debut = min(career.debut, _as_int(row.get("year_ID")) or 9999)
            if not career.name_common:
                career.name_common = (row.get("name_common") or "").strip()
            if pitching:
                career.games_pitched += _as_int(row.get("G"))
                career.games_started += _as_int(row.get("GS"))
    return rows


def _load_register(*, refresh: bool = False) -> dict[str, tuple[str, str]]:
    """Map bbref id -> (first, last) from the sharded Chadwick register."""
    names: dict[str, tuple[str, str]] = {}
    for url in sources.REGISTER_SHARDS:
        path = sources.fetch(url, refresh=refresh)
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("key_bbref") or "").strip()
                if key:
                    names[key] = (
                        (row.get("name_first") or "").strip(),
                        (row.get("name_last") or "").strip(),
                    )
    log.info("register: %d bbref ids with names", len(names))
    return names


def _split_name_common(name_common: str) -> tuple[str, str]:
    """Fallback name split for players missing from the register."""
    parts = name_common.split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return " ".join(parts[:-1]), parts[-1]


def _lahman_to_bbref(*, refresh: bool = False) -> dict[str, str]:
    """Map Lahman ``playerID`` -> bbref ``player_ID``.

    The ids are usually identical but diverge for 242 players: bbref encodes an
    initials-style first name with a literal dot ("burnea.01" for A.J. Burnett) while
    Lahman uses letters ("burneaj01"). Joining on the raw id drops exactly those
    players, which is how A.J. Burnett and A.J. Pierzynski first went missing here.
    """
    path = sources.fetch(sources.PEOPLE_URL, refresh=refresh)
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            lahman_id = (row.get("playerID") or "").strip()
            bbref_id = (row.get("bbrefID") or "").strip()
            if lahman_id and bbref_id and lahman_id != bbref_id:
                mapping[lahman_id] = bbref_id
    log.info("people: %d ids where Lahman and bbref disagree", len(mapping))
    return mapping


def _load_positions(
    wanted: set[str], *, refresh: bool = False
) -> dict[str, dict[int, dict[str, int]]]:
    """Games played per position per season, keyed by bbref id then year."""
    translate = _lahman_to_bbref(refresh=refresh)
    path = sources.fetch(sources.APPEARANCES_URL, refresh=refresh)
    games: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            lahman_id = row.get("playerID", "")
            pid = translate.get(lahman_id, lahman_id)
            if pid not in wanted:
                continue
            # Stints with two teams in one year fold into one season.
            season = games[pid][_as_int(row.get("yearID"))]
            for column, label in POSITION_COLUMNS:
                count = _as_int(row.get(column))
                if count:
                    season[label] += count
    return games


def _career_games(seasons: dict[int, dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for season in seasons.values():
        for label, count in season.items():
            totals[label] += count
    return totals


def _most_played(position_games: dict[str, int]) -> str | None:
    """The label with the most games, ties broken in POSITION_COLUMNS order."""
    # max() on (games, label) would tie-break alphabetically; iterate POSITION_COLUMNS
    # order instead so ties resolve deterministically and the same way every run.
    best_label, best_games = None, 0
    for _, label in POSITION_COLUMNS:
        count = position_games.get(label, 0)
        if count > best_games:
            best_label, best_games = label, count
    return best_label


def _pitcher_role(career: _Career) -> str:
    if career.games_pitched <= 0:
        return "SP" if career.games_started else "RP"
    share = career.games_started / career.games_pitched
    return "SP" if share >= STARTER_GS_SHARE else "RP"


def _primary_position(
    position_games: dict[str, int], career: _Career
) -> str | None:
    """The position with the most career games; pitchers split into SP and RP."""
    label = _most_played(position_games)
    return _pitcher_role(career) if label == "P" else label


def _eligible_positions(
    seasons: dict[int, dict[str, int]],
    career: _Career,
    *,
    min_games: int = DEFAULT_MIN_POSITION_GAMES,
) -> tuple[str, ...]:
    """Primary position first, then every other slot the player qualifies for.

    A slot qualifies with ``min_games`` career games there, or by being the player's
    most-played position in at least one season. Pitching slots go only to career
    pitchers, in the one role their career start share gives them.
    """
    totals = _career_games(seasons)
    primary = _primary_position(totals, career)
    if primary is None:
        return ()
    labels = {label for label, count in totals.items() if count >= min_games}
    labels |= {_most_played(season) for season in seasons.values()} - {None}
    # Only career pitchers pitch: without this, Sam Rice's 1915 (nine games on the
    # mound before he became a right fielder) made a 54-WAR outfielder a reliever.
    labels.discard("P")
    slots = labels - {primary}
    order = [label for _, label in POSITION_COLUMNS]
    return (primary, *sorted(slots, key=order.index))


def build_players(
    *,
    min_seasons: int = DEFAULT_MIN_SEASONS,
    since: int = DEFAULT_SINCE,
    min_position_games: int = DEFAULT_MIN_POSITION_GAMES,
    refresh: bool = False,
) -> list[Player]:
    """Fetch the sources and return every player meeting the eligibility filters."""
    careers: dict[str, _Career] = {}
    bat_path = sources.fetch(sources.WAR_BAT_URL, refresh=refresh)
    pitch_path = sources.fetch(sources.WAR_PITCH_URL, refresh=refresh)
    n_bat = _accumulate_war(bat_path, careers, pitching=False)
    n_pitch = _accumulate_war(pitch_path, careers, pitching=True)
    log.info("WAR files: %d batting rows, %d pitching rows", n_bat, n_pitch)

    eligible = {
        pid: career
        for pid, career in careers.items()
        if len(career.years) >= min_seasons and career.debut >= since
    }
    log.info(
        "eligibility: %d of %d players have >=%d seasons and debuted in %d or later",
        len(eligible),
        len(careers),
        min_seasons,
        since,
    )

    register = _load_register(refresh=refresh)
    # Appearances data ends with the 2021 season, which still covers everyone here: a
    # player needs >=6 seasons, and 2022-2026 is only five, so every eligible player
    # has at least one season in 2021 or earlier.
    position_games = _load_positions(set(eligible), refresh=refresh)

    players: list[Player] = []
    unplaced: list[str] = []
    for pid, career in eligible.items():
        positions = _eligible_positions(
            position_games.get(pid, {}), career, min_games=min_position_games
        )
        if not positions:
            unplaced.append(career.name_common or pid)
            continue
        first, last = register.get(pid) or _split_name_common(career.name_common)
        if not last:
            unplaced.append(pid)
            continue
        players.append(
            Player(
                bbref_id=pid,
                first=first,
                last=last,
                seasons=len(career.years),
                debut=career.debut,
                war=round(career.war, 1),
                position=positions[0],
                positions=positions,
            )
        )

    if unplaced:
        log.info(
            "dropped %d eligible players with no Appearances rows (e.g. %s)",
            len(unplaced),
            ", ".join(sorted(unplaced)[:5]),
        )
    players.sort(key=lambda p: (-p.war, p.last, p.first))
    log.info("built %d players with a primary position", len(players))
    return players


def write_players(players: list[Player], path: Path | None = None) -> Path:
    path = path or sources.PLAYERS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["bbref_id", "first", "last", "seasons", "debut", "war", "position", "positions"]
        )
        for p in players:
            writer.writerow(
                [
                    p.bbref_id,
                    p.first,
                    p.last,
                    p.seasons,
                    p.debut,
                    f"{p.war:.1f}",
                    p.position,
                    "|".join(p.slots),
                ]
            )
    log.info("wrote %s (%d players)", path, len(players))
    return path


def read_players(path: Path | None = None) -> list[Player]:
    path = path or sources.PLAYERS_CSV
    players = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            players.append(
                Player(
                    bbref_id=row["bbref_id"],
                    first=row["first"],
                    last=row["last"],
                    seasons=int(row["seasons"]),
                    debut=int(row["debut"]),
                    war=float(row["war"]),
                    position=row["position"],
                    positions=tuple(filter(None, (row.get("positions") or "").split("|"))),
                )
            )
    return players
