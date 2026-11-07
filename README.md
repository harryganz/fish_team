# The All-Star Fish Team

Builds a baseball all-star team whose players' **last names match the names of real
fish**, maximising career WAR along the way.

```
SLOT  PLAYER                    WAR  TIER     FISH         MATCHED NAME
----------------------------------------------------------------------------
SP    Dizzy Trout              49.9  exact    trout        Trout
RP    Stu Miller               27.0  exact    miller       Molly miller
C     Rick Dempsey             25.2  exact    dempsey      Jack Dempsey
1B    Dots Miller              18.1  exact    miller       Molly miller
2B    Johnny Ray               24.2  exact    ray          Ray
3B    Nellie Fox (2B)          49.3  exact    fox          Flying fox
SS    Melvin Mora (3B)         28.2  exact    mora         Mora
LF    Jim Rice                 47.7  exact    rice         Japanese rice fish
CF    Mike Trout               91.7  exact    trout        Trout
RF    Sam Rice                 54.7  exact    rice         Japanese rice fish
DH    Tim Salmon (RF)          40.6  exact    salmon       Salmon
```

## Usage

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,phonetic]'

.venv/bin/python -m fish_team build            # download, match, write output/
.venv/bin/python -m fish_team report --format md
```

Options: `--refresh` (re-download everything), `--min-seasons N`, `--since YEAR`,
`--min-position-games N`, `--min-tier {exact,near,close,loose,stretch}`,
`--format {table,md,csv}`, and `--matcher embedding` (with `--war-weight`,
`--min-fishiness`) to match by word-embedding meaning instead of spelling.

Outputs land in `output/fish_team.csv` and `output/report.md`; intermediate tables in
`data/players.csv` and `data/fish_vocab.csv`. Raw downloads are cached in `data/raw/`,
so only the first run needs the network.

## How it works

**Players.** Career WAR, seasons and debut year come from Baseball Reference's daily
`war_daily_bat.txt` and `war_daily_pitch.txt`; batting and pitching WAR are summed, so
two-way players are credited for both halves. Eligibility is **≥6 major league seasons**
and a **debut in 1903 or later** (the National Agreement between the AL and NL), which
leaves 6,954 players. Debut is bbref's first season, which counts the Negro Leagues
(1920–48), so Jackie Robinson debuts in 1945. 425 otherwise eligible players, nearly all
Negro League-only careers such as Alejandro Oms, are dropped because Lahman has no
position data for them. First and last names come from the Chadwick Bureau register.

**Positions.** From Lahman's `Appearances` (games, not innings: Lahman's innings are
missing for 1910–1954 and never exist for DH). A player is eligible at a position with
**≥100 career games** there (`--min-position-games`), or if it was his most-played
position in **any single season**. His primary position is the one with the most career
games. Pitching slots go only to career pitchers, split into SP and RP at a 50%
games-started share, so Sam Rice's nine games on the mound in 1915–16 don't make him a
reliever.

**Fish names.** FAO's ASFIS list (13,966 species) filtered taxonomically to the fishes
via ISSCAAP group, plus Wikipedia's vernacular list for the colloquial North American
names, reduced to ~1,150 **head nouns**.

**Matching.** Surnames are scored against head nouns with a length-guarded blend of
Levenshtein and Jaro-Winkler similarity, bucketed into tiers: `exact`, `near` (≥0.90),
`close` (≥0.85), `loose` (≥0.80), `stretch` (≥0.75). A character-trigram index narrows
each surname to a few dozen plausible fish words before scoring.

**Selection.** Because one player can be eligible at several slots, the roster is
solved as an assignment problem (Hungarian algorithm, `scipy.optimize.linear_sum_assignment`)
so nobody fills two slots. The objective is **exact-first** across the team: the most
total match quality first, then the most total WAR. Players picked out of their primary
position are shown with it in brackets. The report also names the highest-WAR alternative per slot, so the cost of
preferring match quality is visible rather than hidden.

## Design notes

Three things that look like details but decide whether this works at all.

**Only the head noun can match.** A surname is compared to the last word of a fish name
("largemouth bass" → `bass`), never to a modifier. Matching against every word returns
Mickey Mantle and Mike Schmidt (species named after people) and Kevin Brown, Devon White
and Vida Blue (colour words) as fish.

**No substring-tolerant scorer.** `rapidfuzz`'s `WRatio` and `partial_ratio` rate a
short fish word highly against any surname that contains something like it: Al Kaline ~
"line" and Ozzie Smith ~ "saithe" both scored 0.87. The scorer here charges for every
character that fails to line up, and `tests/test_matching.py` keeps those pairs as
regressions.

**The vocabulary needs curating.** Some entries are headed by a colour, a body part or
an aggregate label ("White-eye", "Common silver belly", "Marine fishes NEI"). Those are
dropped via `data/manual/stoplist.txt`, which is plain text and meant to be edited —
Pedro Martinez ~ "marine" scores 0.83 and is kept out by this list rather than by the
scorer.

## Data sources and caveats

| Source | Notes |
| --- | --- |
| [bbref WAR files](https://www.baseball-reference.com/data/war_daily_bat.txt) | Regenerated daily, so career totals drift slightly between runs; fetch times are recorded in the report. |
| [Chadwick Bureau register](https://github.com/chadwickbureau/register) | 16 CSV shards, ~68 MB total. |
| Lahman `Appearances` / `People` | The canonical `chadwickbureau/baseballdatabank` repo **has been deleted** (404); a [fork](https://github.com/xorq-labs/baseballdatabank) is used instead. Its data ends with 2021, which still covers every eligible player: ≥6 seasons means at least one season falls in 2021 or earlier. Fallback if the fork disappears: <https://sabr.org/lahman-database/>. |
| [FAO ASFIS](https://www.fao.org/fishery/en/collection/asfis) | Version 2026.1. |
| [Wikipedia](https://en.wikipedia.org/wiki/List_of_fish_by_common_name) | Fetched as raw wikitext; `#REDIRECT` hops are followed, since the article has been moved before. |

The American Fisheries Society list was the original intent, but
`names.fisheries.org` refuses automated requests (403) and the current edition is a paid
book. To use it, drop a CSV with a common-name column into `data/manual/` and extend
`fish.build_vocabulary`.

`People.csv` is fetched for one reason only: Baseball Reference and Lahman disagree on
242 player ids — bbref writes A.J. Burnett as `burnea.01` where Lahman writes
`burneaj01` — and joining on the raw id silently drops exactly those players.

## Tests

```bash
.venv/bin/pytest
```

No network access required; every test builds its own fixtures.
