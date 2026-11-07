"""Command line entry point: ``build`` the team, then ``report`` on it."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import embedding, fish, matching, players, roster, sources

log = logging.getLogger("fish_team")

TEAM_CSV_HEADER = (
    "slot",
    "first",
    "last",
    "war",
    "seasons",
    "debut",
    "tier",
    "score",
    "fish_head_noun",
    "fish_example",
    "fish_n_species",
    "bbref_id",
)


def _name_at(player: players.Player, slot: str) -> str:
    """The player's name, flagged with his primary position when he plays out of it."""
    if player.position == slot:
        return player.full_name
    return f"{player.full_name} ({player.position})"


def _team_rows(selections: dict[str, roster.Selection]) -> list[list[object]]:
    rows: list[list[object]] = []
    for slot in roster.SLOTS:
        selection = selections[slot]
        pick = selection.pick
        if pick is None:
            rows.append([slot, "", "UNFILLED", "", "", "", "", "", "", "", "", ""])
            continue
        rows.append(
            [
                slot,
                pick.player.first,
                pick.player.last,
                f"{pick.war:.1f}",
                pick.player.seasons,
                pick.player.debut,
                pick.tier,
                f"{pick.match.score:.3f}",
                pick.match.head,
                pick.match.example,
                pick.match.n_species,
                pick.player.bbref_id,
            ]
        )
    return rows


def write_team_csv(
    selections: dict[str, roster.Selection], path: Path | None = None
) -> Path:
    path = path or sources.OUTPUT_DIR / "fish_team.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(TEAM_CSV_HEADER)
        writer.writerows(_team_rows(selections))
    log.info("wrote %s", path)
    return path


def format_table(selections: dict[str, roster.Selection]) -> str:
    """The team as a plain-text table, for the terminal."""
    lines = [
        f"{'SLOT':<5} {'PLAYER':<22} {'WAR':>6}  {'TIER':<8} {'FISH':<12} MATCHED NAME",
        "-" * 88,
    ]
    for slot in roster.SLOTS:
        selection = selections[slot]
        pick = selection.pick
        if pick is None:
            lines.append(f"{slot:<5} {'-- UNFILLED --':<22}")
            continue
        lines.append(
            f"{slot:<5} {_name_at(pick.player, slot):<22} {pick.war:>6.1f}  "
            f"{pick.tier:<8} {pick.match.head:<12} {pick.match.example}"
        )
    lines.append("-" * 88)
    lines.append(f"{'':5} {'TOTAL':<22} {roster.total_war(selections):>6.1f}")
    return "\n".join(lines)


def format_report(
    selections: dict[str, roster.Selection],
    *,
    all_players: list[players.Player],
    vocab: dict[str, fish.FishWord],
    candidates: dict[str, list[roster.Candidate]],
    min_tier: str,
) -> str:
    """The full markdown report."""
    filled = [s for s in selections.values() if s.filled]
    unfilled = [s.slot for s in selections.values() if not s.filled]
    tier_counts: dict[str, int] = {}
    for selection in filled:
        tier_counts[selection.tier_used or "?"] = (
            tier_counts.get(selection.tier_used or "?", 0) + 1
        )

    out: list[str] = []
    out.append("# The All-Star Fish Team\n")
    out.append(
        "Every player's surname matches the head noun of a real fish name. Slots are "
        "filled **exact-first**: the best available match tier wins, and the highest "
        "career WAR wins within that tier.\n"
    )
    out.append(f"**Total career WAR: {roster.total_war(selections):.1f}**")
    out.append(
        "Match tiers used: "
        + ", ".join(f"{n} x{c}" for n, c in sorted(tier_counts.items(), key=lambda kv: matching.tier_rank(kv[0])))
        + (f" | unfilled: {', '.join(unfilled)}" if unfilled else "")
        + "\n"
    )

    out.append("## The team\n")
    out.append("| Slot | Player | WAR | Seasons | Tier | Score | Fish | Matched name |")
    out.append("| --- | --- | --: | --: | --- | --: | --- | --- |")
    for slot in roster.SLOTS:
        pick = selections[slot].pick
        if pick is None:
            out.append(f"| **{slot}** | _UNFILLED_ | | | | | | |")
            continue
        out.append(
            f"| **{slot}** | {_name_at(pick.player, slot)} | {pick.war:.1f} | "
            f"{pick.player.seasons} | {pick.tier} | {pick.match.score:.2f} | "
            f"`{pick.match.head}` | {pick.match.example} |"
        )

    costly = [s for s in selections.values() if s.war_forgone > 0]
    out.append("\n## What exact-first costs\n")
    if costly:
        out.append(
            "The highest-WAR candidate at any allowed tier, where that is not the "
            "player selected. Insisting on the better match tier gives up "
            f"**{sum(s.war_forgone for s in costly):.1f} WAR** in total.\n"
        )
        out.append("| Slot | Selected | WAR | Highest-WAR alternative | WAR | Fish | Given up |")
        out.append("| --- | --- | --: | --- | --: | --- | --: |")
        for selection in sorted(costly, key=lambda s: -s.war_forgone):
            pick, alt = selection.pick, selection.war_max
            assert pick is not None and alt is not None
            out.append(
                f"| **{selection.slot}** | {pick.player.full_name} | {pick.war:.1f} | "
                f"{alt.player.full_name} | {alt.war:.1f} | `{alt.match.head}` "
                f"({alt.tier}) | {selection.war_forgone:.1f} |"
            )
    else:
        out.append("Nothing: every slot's best match is also its highest-WAR candidate.\n")

    out.append("\n## Alternates by slot\n")
    for slot in roster.SLOTS:
        selection = selections[slot]
        pool = candidates.get(slot, [])
        out.append(f"**{slot}** — {len(pool)} candidate(s) at `{min_tier}` or better\n")
        if not selection.alternates:
            out.append("_none_\n")
            continue
        out.append("| Player | WAR | Tier | Score | Fish | Matched name |")
        out.append("| --- | --: | --- | --: | --- | --- |")
        for candidate in selection.alternates:
            marker = " **←**" if candidate is selection.pick else ""
            out.append(
                f"| {candidate.player.full_name}{marker} | {candidate.war:.1f} | "
                f"{candidate.tier} | {candidate.match.score:.2f} | "
                f"`{candidate.match.head}` | {candidate.match.example} |"
            )
        out.append("")

    out.append("## How this was built\n")
    out.append(f"- Eligible players: **{len(all_players)}**")
    out.append(f"- Fish head nouns: **{len(vocab)}**")
    out.append(
        f"- Candidates with a match at `{min_tier}` or better: "
        f"**{sum(len(v) for v in candidates.values())}**"
    )
    out.append(
        "- Tier thresholds: "
        + ", ".join(
            f"`{name}` ≥ {threshold:.2f}" if name != "exact" else "`exact` = identical"
            for name, threshold in matching.TIERS
        )
    )
    out.append("\n### Source data\n")
    out.append("| Source | Fetched |")
    out.append("| --- | --- |")
    for label, url in (
        ("bbref WAR (batting)", sources.WAR_BAT_URL),
        ("bbref WAR (pitching)", sources.WAR_PITCH_URL),
        ("Lahman Appearances", sources.APPEARANCES_URL),
        ("Lahman People", sources.PEOPLE_URL),
        ("FAO ASFIS", sources.ASFIS_URL),
    ):
        name = url.split("/")[-1].split("?")[0]
        out.append(f"| {label} | {sources.fetched_at(sources.RAW_DIR / name)} |")
    out.append(
        f"\n_Report generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        "Baseball Reference regenerates its WAR files daily, so totals shift slightly "
        "between runs._\n"
    )
    return "\n".join(out)


def format_blend_table(selections: dict[str, roster.BlendSelection]) -> str:
    """The embedding-mode team as a plain-text table."""
    lines = [
        f"{'SLOT':<5} {'PLAYER':<22} {'WAR':>6}  {'FISHY':>5}  {'BLEND':>5}  NEAREST FISH",
        "-" * 88,
    ]
    for slot in roster.SLOTS:
        pick = selections[slot].pick
        if pick is None:
            lines.append(f"{slot:<5} {'-- UNFILLED --':<22}")
            continue
        lines.append(
            f"{slot:<5} {_name_at(pick.player, slot):<22} {pick.war:>6.1f}  "
            f"{pick.fishiness.score:>5.2f}  {pick.blend:>5.2f}  "
            + ", ".join(pick.fishiness.nearest[:3])
        )
    lines.append("-" * 88)
    lines.append(f"{'':5} {'TOTAL':<22} {roster.total_blend_war(selections):>6.1f}")
    return "\n".join(lines)


def write_fishiness_csv(scored: list[roster.BlendCandidate], path: Path | None = None) -> Path:
    """Every scored player, fishiest first, for inspecting the ranking itself."""
    path = path or sources.OUTPUT_DIR / "fishiness.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("first", "last", "position", "eligible", "war", "fishiness", "blend", "nearest_fish", "bbref_id"))
        for c in sorted(scored, key=lambda c: -c.fishiness.score):
            writer.writerow(
                (
                    c.player.first,
                    c.player.last,
                    c.player.position,
                    "|".join(c.player.slots),
                    f"{c.war:.1f}",
                    f"{c.fishiness.score:.3f}",
                    f"{c.blend:.3f}",
                    " ".join(c.fishiness.nearest),
                    c.player.bbref_id,
                )
            )
    log.info("wrote %s", path)
    return path


def format_blend_report(
    selections: dict[str, roster.BlendSelection],
    *,
    scored: list[roster.BlendCandidate],
    all_players: list[players.Player],
    war_weight: float,
    min_fishiness: float,
) -> str:
    """The embedding-mode markdown report."""
    unfilled = [s.slot for s in selections.values() if not s.filled]
    out: list[str] = []
    out.append("# The All-Star Fish Team (by meaning)\n")
    out.append(
        "Each surname is scored by its cosine similarity, in GloVe word vectors, to the "
        "average of forty common fish words (Trout is scored leave-one-out). A slot goes "
        f"to the highest `z(fishiness) + {war_weight:g}·z(WAR)` among players with "
        f"fishiness ≥ {min_fishiness:.2f}.\n"
    )
    out.append(f"**Total career WAR: {roster.total_blend_war(selections):.1f}**")
    if unfilled:
        out.append(f"Unfilled (nobody over the floor): {', '.join(unfilled)}")
    out.append("\n## The team\n")
    out.append("| Slot | Player | WAR | Seasons | Fishiness | Blend | Nearest fish |")
    out.append("| --- | --- | --: | --: | --: | --: | --- |")
    for slot in roster.SLOTS:
        pick = selections[slot].pick
        if pick is None:
            out.append(f"| **{slot}** | _UNFILLED_ | | | | | |")
            continue
        out.append(
            f"| **{slot}** | {_name_at(pick.player, slot)} | {pick.war:.1f} | {pick.player.seasons} | "
            f"{pick.fishiness.score:.2f} | {pick.blend:.2f} | {', '.join(pick.fishiness.nearest)} |"
        )
    out.append("\n## Fishiest surnames overall\n")
    out.append("| Player | Pos | WAR | Fishiness | Nearest fish |")
    out.append("| --- | --- | --: | --: | --- |")
    for c in sorted(scored, key=lambda c: -c.fishiness.score)[:25]:
        out.append(
            f"| {c.player.full_name} | {c.player.position} | {c.war:.1f} | "
            f"{c.fishiness.score:.2f} | {', '.join(c.fishiness.nearest[:3])} |"
        )
    out.append("\n## How this was built\n")
    out.append(f"- Eligible players: **{len(all_players)}**; in the embedding vocabulary: **{len(scored)}**")
    out.append(f"- Vectors: `glove-wiki-gigaword-300`, fetched {sources.fetched_at(sources.RAW_DIR / embedding.GLOVE_URL.split('/')[-1])}")
    out.append(f"- Seed fish: {', '.join(embedding.SEED_FISH)}")
    out.append(
        f"\n_Report generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}._\n"
    )
    return "\n".join(out)


def _run_embedding(args: argparse.Namespace) -> int:
    if args.command == "report" and sources.PLAYERS_CSV.exists():
        all_players = players.read_players()
    else:
        all_players = players.build_players(
            min_seasons=args.min_seasons,
            since=args.since,
            min_position_games=args.min_position_games,
            refresh=args.refresh,
        )
        players.write_players(all_players)

    axis = embedding.build_axis([p.last for p in all_players])
    scored = roster.score_players(all_players, axis, war_weight=args.war_weight)
    selections = roster.select_blended(scored, min_fishiness=args.min_fishiness)

    csv_path = write_fishiness_csv(scored)
    report = format_blend_report(
        selections,
        scored=scored,
        all_players=all_players,
        war_weight=args.war_weight,
        min_fishiness=args.min_fishiness,
    )
    report_path = sources.OUTPUT_DIR / "report_embedding.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("wrote %s", report_path)

    if args.format == "table":
        print(format_blend_table(selections))
        print(f"\nFull report: {report_path}")
    elif args.format == "md":
        print(report)
    else:
        print(csv_path.read_text(encoding="utf-8"), end="")
    return 0 if all(s.filled for s in selections.values()) else 1


def _load_or_build(
    args: argparse.Namespace,
) -> tuple[
    list[players.Player],
    dict[str, fish.FishWord],
    dict[str, list[roster.Candidate]],
    dict[str, roster.Selection],
]:
    """Return (players, vocab, candidates, selections), building what is missing."""
    if args.command == "report" and sources.PLAYERS_CSV.exists() and sources.FISH_VOCAB_CSV.exists():
        all_players = players.read_players()
        vocab = fish.read_vocabulary()
    else:
        all_players = players.build_players(
            min_seasons=args.min_seasons,
            since=args.since,
            min_position_games=args.min_position_games,
            refresh=args.refresh,
        )
        players.write_players(all_players)
        vocab = fish.build_vocabulary(refresh=args.refresh)
        fish.write_vocabulary(vocab)

    index = matching.FishIndex(vocab)
    index.save()
    candidates = roster.collect_candidates(all_players, index, min_tier=args.min_tier)
    selections = roster.select_roster(candidates, min_tier=args.min_tier)
    return all_players, vocab, candidates, selections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fish-team",
        description="Assemble an all-star baseball team of fish-name surnames.",
    )
    parser.add_argument(
        "command", choices=("build", "report"), help="build downloads and rebuilds; report reuses the cached tables"
    )
    parser.add_argument("--refresh", action="store_true", help="re-download every source")
    parser.add_argument(
        "--min-seasons",
        type=int,
        default=players.DEFAULT_MIN_SEASONS,
        help="minimum major league seasons (default: %(default)s)",
    )
    parser.add_argument(
        "--since",
        type=int,
        default=players.DEFAULT_SINCE,
        help="earliest debut season (default: %(default)s, the AL-NL National Agreement)",
    )
    parser.add_argument(
        "--min-position-games",
        type=int,
        default=players.DEFAULT_MIN_POSITION_GAMES,
        help="career games that make a player eligible at a position; his most-played "
        "position in any single season also counts (default: %(default)s)",
    )
    parser.add_argument(
        "--min-tier",
        choices=matching.TIER_NAMES,
        default=matching.DEFAULT_MIN_TIER,
        help="worst match tier a slot may be filled from (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "md", "csv"),
        default="table",
        help="what to print (default: %(default)s); files are always written",
    )
    parser.add_argument(
        "--matcher",
        choices=("spelling", "embedding"),
        default="spelling",
        help="match surnames by spelling or by word-embedding meaning (default: %(default)s)",
    )
    parser.add_argument(
        "--war-weight",
        type=float,
        default=roster.DEFAULT_WAR_WEIGHT,
        help="embedding mode: weight on z(WAR) against z(fishiness) (default: %(default)s)",
    )
    parser.add_argument(
        "--min-fishiness",
        type=float,
        default=roster.DEFAULT_MIN_FISHINESS,
        help="embedding mode: lowest fishiness a slot may be filled from (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose or args.command == "build" else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )

    if args.matcher == "embedding":
        return _run_embedding(args)

    all_players, vocab, candidates, selections = _load_or_build(args)

    write_team_csv(selections)
    report = format_report(
        selections,
        all_players=all_players,
        vocab=vocab,
        candidates=candidates,
        min_tier=args.min_tier,
    )
    report_path = sources.OUTPUT_DIR / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    log.info("wrote %s", report_path)

    if args.format == "table":
        print(format_table(selections))
        print(f"\nFull report: {report_path}")
    elif args.format == "md":
        print(report)
    else:
        print((sources.OUTPUT_DIR / "fish_team.csv").read_text(encoding="utf-8"), end="")

    return 0 if all(s.filled for s in selections.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
