"""Download and cache the raw data files the pipeline is built from.

Everything lands in ``data/raw/`` and is reused on later runs unless ``refresh``
is set, so a rebuild after the first run needs no network at all.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANUAL_DIR = DATA_DIR / "manual"
INDEX_DIR = DATA_DIR / "index"
OUTPUT_DIR = PROJECT_ROOT / "output"

PLAYERS_CSV = DATA_DIR / "players.csv"
FISH_VOCAB_CSV = DATA_DIR / "fish_vocab.csv"
STOPLIST_TXT = MANUAL_DIR / "stoplist.txt"

# Baseball Reference regenerates these daily, so career WAR totals drift a little
# between runs; the ``.meta.json`` sidecar records when we fetched them.
WAR_BAT_URL = "https://www.baseball-reference.com/data/war_daily_bat.txt"
WAR_PITCH_URL = "https://www.baseball-reference.com/data/war_daily_pitch.txt"

# Chadwick Bureau register, sharded 16 ways by the first hex digit of key_person.
# Gives a clean name_first/name_last split keyed to bbref ids.
REGISTER_SHARDS = [
    f"https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{c}.csv"
    for c in "0123456789abcdef"
]

# Lahman's Appearances table, for per-position games played.
#
# NOTE: the canonical chadwickbureau/baseballdatabank repo has been deleted (404 as of
# 2026-10); this fork is a live mirror. Its data ends with the 2021 season, which is
# still sufficient here: the >=6-season eligibility filter means every eligible player
# has at least one season in 2021 or earlier (2022-2026 is only five seasons). If the
# fork disappears, the Lahman database is also published at
# https://sabr.org/lahman-database/.
APPEARANCES_URL = (
    "https://raw.githubusercontent.com/xorq-labs/baseballdatabank/master/core/Appearances.csv"
)

# Lahman's People table, used purely to translate its own playerID into a bbref id.
# The two differ for 242 players -- bbref writes A.J. Burnett as "burnea.01" where
# Lahman writes "burneaj01" -- so joining the tables on a bare id silently loses them.
PEOPLE_URL = (
    "https://raw.githubusercontent.com/xorq-labs/baseballdatabank/master/core/People.csv"
)

# FAO ASFIS list of species for fishery statistics purposes: ~14k species with English
# common names and an ISSCAAP group we can use to keep only the fishes.
ASFIS_URL = "https://www.fao.org/fishery/static/ASFIS/ASFIS_sp.zip"

# Wikipedia's vernacular list, fetched as raw wikitext so we need no HTML parser.
# Supplies colloquial North American names (bluegill, albacore) that ASFIS, which is
# oriented toward commercial fisheries, spells differently or omits.
#
# The article has been moved before ("List of fish common names" is now a redirect), so
# fish.py follows #REDIRECT hops rather than trusting this title to stay put.
WIKIPEDIA_TITLE = "List of fish by common name"


def wikitext_url(title: str) -> str:
    """URL for an article's raw wikitext."""
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")) + "?action=raw"

USER_AGENT = (
    "fish-team/0.1 (hobby project assembling a baseball team of fish names; "
    "contact via https://github.com/)"
)

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        # Baseball Reference sits behind Cloudflare and rejects the default
        # python-requests user agent.
        s.headers.update({"User-Agent": USER_AGENT})
        _SESSION = s
    return _SESSION


def _meta_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".meta.json")


def fetch(url: str, filename: str | None = None, *, refresh: bool = False) -> Path:
    """Download ``url`` into ``data/raw/``, returning the local path.

    A cached copy is reused unless ``refresh`` is true.
    """
    name = filename or url.split("/")[-1].split("?")[0]
    dest = RAW_DIR / name
    if dest.exists() and not refresh:
        log.debug("cached %s", dest.name)
        return dest

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _session().get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
    tmp.replace(dest)

    _meta_path(dest).write_text(
        json.dumps(
            {
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bytes": dest.stat().st_size,
            },
            indent=2,
        )
        + "\n"
    )
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def fetched_at(dest: Path) -> str:
    """Return the recorded fetch timestamp for a cached file, or ``"unknown"``."""
    meta = _meta_path(dest)
    if not meta.exists():
        return "unknown"
    try:
        return json.loads(meta.read_text()).get("fetched_at", "unknown")
    except (OSError, json.JSONDecodeError):
        return "unknown"
