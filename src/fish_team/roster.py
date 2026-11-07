"""Fill the eleven roster slots using exact-first tier selection.

For each slot the rule is: take the best tier that has any candidate at all, then take
the highest career WAR within that tier. Match quality wins over WAR, but only as a
tie-break between tiers -- never within one.

A player can be eligible at several slots (Mike Trout at CF, LF and DH), so slots are
not independent: the roster is an assignment problem, solved exactly with the Hungarian
algorithm. Exact-first becomes a lexicographic objective over the whole team -- the
most total match quality first, then the most total WAR -- so no slot is given a worse
tier just to free up a player for a slot that could have used someone else.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from scipy.optimize import linear_sum_assignment

from .embedding import FishAxis, Fishiness
from .matching import DEFAULT_MIN_TIER, FishIndex, Match, tier_rank, tiers_from
from .players import Player

log = logging.getLogger(__name__)

SLOTS: tuple[str, ...] = (
    "SP",
    "RP",
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "LF",
    "CF",
    "RF",
    "DH",
)

ALTERNATES_SHOWN = 5

# Weight of one step of match tier against WAR in the assignment objective. Larger than
# any possible spread in career WAR, so tiers are compared first.
_TIER_STEP = 1000.0
_FORBIDDEN = -1e12

C = TypeVar("C")


def assign(
    pools: dict[str, list[C]],
    weight: Callable[[C], float],
    key: Callable[[C], str],
) -> dict[str, C | None]:
    """Give each slot at most one candidate, never the same player twice, maximising
    the total ``weight``. Slots left without anyone available map to ``None``."""
    slots = [slot for slot in SLOTS if pools.get(slot)]
    people = sorted({key(c) for slot in slots for c in pools[slot]})
    picks: dict[str, C | None] = {slot: None for slot in SLOTS}
    if not slots:
        return picks
    column = {k: j for j, k in enumerate(people)}
    matrix = np.full((len(slots), len(people)), _FORBIDDEN)
    by_cell: dict[tuple[int, int], C] = {}
    for i, slot in enumerate(slots):
        for candidate in pools[slot]:
            j = column[key(candidate)]
            matrix[i, j] = weight(candidate)
            by_cell[i, j] = candidate
    for i, j in zip(*linear_sum_assignment(matrix, maximize=True)):
        if (i, j) in by_cell:
            picks[slots[i]] = by_cell[i, j]
    return picks


@dataclass(frozen=True)
class Candidate:
    """A player who could fill a slot, with the fish word his surname matches."""

    player: Player
    match: Match

    @property
    def war(self) -> float:
        return self.player.war

    @property
    def tier(self) -> str:
        return self.match.tier


@dataclass
class Selection:
    """The outcome for one slot."""

    slot: str
    pick: Candidate | None
    tier_used: str | None
    #: Highest-WAR candidate at any allowed tier. Equal to ``pick`` unless insisting on
    #: the best tier cost us WAR, which is the whole trade-off worth seeing.
    war_max: Candidate | None
    alternates: list[Candidate]

    @property
    def filled(self) -> bool:
        return self.pick is not None

    @property
    def war_forgone(self) -> float:
        if self.pick is None or self.war_max is None:
            return 0.0
        return round(self.war_max.war - self.pick.war, 1)


def collect_candidates(
    players: list[Player],
    index: FishIndex,
    *,
    min_tier: str = DEFAULT_MIN_TIER,
) -> dict[str, list[Candidate]]:
    """Group every player with a good enough fish match under each slot he can play."""
    allowed = set(tiers_from(min_tier))
    by_slot: dict[str, list[Candidate]] = {slot: [] for slot in SLOTS}
    for player in players:
        slots = [slot for slot in player.slots if slot in by_slot]
        if not slots:
            continue
        match = index.best_match(player.last, min_tier=min_tier)
        if match is None or match.tier not in allowed:
            continue
        candidate = Candidate(player=player, match=match)
        for slot in slots:
            by_slot[slot].append(candidate)
    for slot, candidates in by_slot.items():
        candidates.sort(key=lambda c: (tier_rank(c.tier), -c.war))
        log.info("%s: %d candidates", slot, len(candidates))
    return by_slot


def select_roster(
    candidates: dict[str, list[Candidate]],
    *,
    min_tier: str = DEFAULT_MIN_TIER,
) -> dict[str, Selection]:
    """Apply exact-first tier selection across the whole roster."""
    allowed = tiers_from(min_tier)
    pools = {
        slot: [c for c in candidates.get(slot, []) if c.tier in allowed] for slot in SLOTS
    }
    picks = assign(
        pools,
        weight=lambda c: (len(allowed) - tier_rank(c.tier)) * _TIER_STEP + c.war,
        key=lambda c: c.player.bbref_id,
    )
    selections: dict[str, Selection] = {}
    for slot in SLOTS:
        pool = pools[slot]
        pick = picks[slot]
        tier_used = pick.tier if pick else None
        war_max = max(pool, key=lambda c: c.war) if pool else None
        alternates = sorted(pool, key=lambda c: -c.war)[:ALTERNATES_SHOWN]
        selections[slot] = Selection(
            slot=slot,
            pick=pick,
            tier_used=tier_used,
            war_max=war_max,
            alternates=alternates,
        )
    unfilled = [s for s, sel in selections.items() if not sel.filled]
    if unfilled:
        log.warning("no candidate at %s or better for: %s", min_tier, ", ".join(unfilled))
    return selections


def total_war(selections: dict[str, Selection]) -> float:
    return round(sum(s.pick.war for s in selections.values() if s.pick), 1)


# --- Embedding mode ---------------------------------------------------------------
#
# Here every player gets a continuous fishiness score rather than a tier, and a slot is
# filled by a weighted blend of fishiness and WAR. Both are z-scored over the whole
# eligible pool, not per slot, so a blend of 1.0 means the same thing at every position.

DEFAULT_WAR_WEIGHT = 0.5
# Below ~0.30 the ranking is food and animal words (fried, veal, deer) and then noise;
# Carl Yastrzemski scores 0.29 by accident of his rare-word vector.
DEFAULT_MIN_FISHINESS = 0.30


@dataclass(frozen=True)
class BlendCandidate:
    """A player scored by meaning rather than spelling."""

    player: Player
    fishiness: Fishiness
    blend: float

    @property
    def war(self) -> float:
        return self.player.war


@dataclass
class BlendSelection:
    slot: str
    pick: BlendCandidate | None
    #: Every candidate over the floor, best blend first.
    pool: list[BlendCandidate]

    @property
    def filled(self) -> bool:
        return self.pick is not None

    @property
    def fishiest(self) -> BlendCandidate | None:
        return max(self.pool, key=lambda c: c.fishiness.score, default=None)


def _zscorer(values: list[float]):
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 or 1.0
    return lambda v: (v - mean) / sd


def score_players(
    players: list[Player], axis: FishAxis, *, war_weight: float = DEFAULT_WAR_WEIGHT
) -> list[BlendCandidate]:
    """Every player whose surname is in the embedding vocabulary, best blend first."""
    scored = [(p, f) for p in players if (f := axis.score(p.last)) is not None]
    if not scored:
        return []
    z_fish = _zscorer([f.score for _, f in scored])
    z_war = _zscorer([p.war for p, _ in scored])
    out = [
        BlendCandidate(player=p, fishiness=f, blend=z_fish(f.score) + war_weight * z_war(p.war))
        for p, f in scored
    ]
    out.sort(key=lambda c: -c.blend)
    log.info("fishiness scored for %d of %d players", len(out), len(players))
    return out


def select_blended(
    scored: list[BlendCandidate], *, min_fishiness: float = DEFAULT_MIN_FISHINESS
) -> dict[str, BlendSelection]:
    """The highest total blend among players at or above the fishiness floor."""
    pools = {
        slot: [
            c for c in scored
            if slot in c.player.slots and c.fishiness.score >= min_fishiness
        ]  # fmt: skip
        for slot in SLOTS
    }
    picks = assign(pools, weight=lambda c: c.blend, key=lambda c: c.player.bbref_id)
    selections = {
        slot: BlendSelection(slot=slot, pick=picks[slot], pool=pools[slot]) for slot in SLOTS
    }
    unfilled = [s for s, sel in selections.items() if not sel.filled]
    if unfilled:
        log.warning(
            "no candidate with fishiness >= %.2f for: %s", min_fishiness, ", ".join(unfilled)
        )
    return selections


def total_blend_war(selections: dict[str, BlendSelection]) -> float:
    return round(sum(s.pick.war for s in selections.values() if s.pick), 1)
