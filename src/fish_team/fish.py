"""Build the fish-name vocabulary the matcher scores surnames against.

Two sources are merged:

* **FAO ASFIS** -- ~14k species with English common names, filtered taxonomically to
  the fishes via the ISSCAAP group.
* **Wikipedia's list of fish common names** -- ~2k vernacular names, which covers the
  colloquial North American words (bluegill, albacore) that ASFIS, being oriented
  toward commercial fishery statistics, spells differently or omits.

The unit of comparison is the **head noun**: the last word of a fish name, which is the
word that actually names the fish ("largemouth bass" -> ``bass``). Matching a surname
against *every* word of a fish name does not work -- it reports Mickey Mantle and Mike
Schmidt (species epithets honouring people) and Kevin Brown, Devon White and Vida Blue
(colour modifiers) as fish.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import sources

log = logging.getLogger(__name__)

# ISSCAAP divisions 11-39 are the fishes (freshwater, diadromous, marine and
# cartilaginous). 41+ are crustaceans, molluscs, aquatic mammals, plants and misc.
FISH_ISSCAAP_RANGE = (11, 39)

# Splits a name into independently usable alternates: "Saithe = Pollock",
# "Striped bass, hybrid", "Sprat/brisling".
_ALTERNATE_SPLIT = re.compile(r"\s*[,;/=]\s*")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
# ASFIS writes explicit synonyms as "Saithe(=Pollock)" or
# "Alaska pollock (=walleye pollock)" -- those are usable names, unlike the taxonomic
# parentheticals Wikipedia uses ("Anchovy (family Engraulidae)").
_SYNONYM_PARENTHETICAL = re.compile(r"\(\s*=\s*([^)]*)\)")
_WORD_SPLIT = re.compile(r"[^a-z'\-]+")

# Words already singular despite the -ies/-es ending.
_INVARIANT_PLURALS = frozenset({"species", "series", "facies", "caries", "rabies"})

# Wikipedia wikitext cleanup.
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]")
_WIKI_ITALIC = re.compile(r"'{2,}")
# Footnotes must go contents and all; stripping only the tags leaves the citation text
# behind as part of the name.
_WIKI_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL | re.IGNORECASE)
_WIKI_TAG = re.compile(r"<[^>]+>")
_WIKI_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")
_WIKI_BULLET = re.compile(r"^\*+\s*")


@dataclass
class FishWord:
    """One head noun, with the evidence behind it."""

    head: str
    n_species: int = 0
    example: str = ""
    sources: set[str] = field(default_factory=set)


def deaccent(text: str) -> str:
    """Strip combining marks: ``Rubén`` -> ``Ruben``."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def singularize(word: str) -> str:
    """Best-effort English singular for a fish head noun.

    Deliberately written as explicit suffix rules. Do not reach for
    ``str.rstrip("'s")`` here -- ``rstrip`` strips a character *set*, so it turns
    ``bass`` into ``ba`` and silently loses all 30 bass species.
    """
    if word in _INVARIANT_PLURALS:
        return word
    for suffix, replacement in (
        ("ies", "y"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("sses", "ss"),
        ("xes", "x"),
        ("zes", "z"),
    ):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)] + replacement
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def load_stoplist(path: Path | None = None) -> frozenset[str]:
    """Read the curated non-fish head nouns from ``data/manual/stoplist.txt``."""
    path = path or sources.STOPLIST_TXT
    if not path.exists():
        log.warning("no stoplist at %s; using an empty stoplist", path)
        return frozenset()
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            words.add(singularize(line))
    return frozenset(words)


def _effective_head(token: str) -> str:
    """The grammatical head of a token: the final segment of a hyphenated word.

    ``white-eye`` -> ``eye``, so one stoplist entry covers every ``*-eye`` compound.
    """
    return token.rsplit("-", 1)[-1]


def head_noun(name: str, stoplist: frozenset[str]) -> str | None:
    """Extract the head noun of a single fish name, or ``None`` if there isn't one."""
    cleaned = _PARENTHETICAL.sub(" ", deaccent(name).lower())
    words = [w.strip("'-") for w in _WORD_SPLIT.split(cleaned) if w.strip("'-")]
    # Pop trailing modifiers, aggregates and body parts until a real head is on top.
    while words:
        candidate = singularize(_effective_head(words[-1]))
        if len(candidate) < 3 or candidate in stoplist:
            words.pop()
            continue
        return candidate
    return None


def heads_from_name(name: str, stoplist: frozenset[str]) -> list[str]:
    """Head nouns for every alternate spelling packed into one name field."""
    # Pull "(=synonym)" out as its own name before the rest of the parentheticals
    # are discarded.
    parts = _SYNONYM_PARENTHETICAL.findall(name)
    parts.append(_SYNONYM_PARENTHETICAL.sub(" ", name))

    heads: list[str] = []
    for part in (p for raw in parts for p in _ALTERNATE_SPLIT.split(_PARENTHETICAL.sub(" ", raw))):
        if not part.strip():
            continue
        head = head_noun(part, stoplist)
        if head and head not in heads:
            heads.append(head)
    return heads


def _iter_asfis(path: Path):
    """Yield English common names for fish species in the ASFIS zip."""
    with zipfile.ZipFile(path) as zf:
        csv_names = [i.filename for i in zf.infolist() if i.filename.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"no CSV inside {path}")
        text = zf.read(csv_names[0]).decode("utf-8-sig", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    # ASFIS ships a trailing space in the "ISSCAAP_Group " header.
    reader.fieldnames = [f.strip() for f in (reader.fieldnames or [])]
    lo, hi = FISH_ISSCAAP_RANGE
    for row in reader:
        group = (row.get("ISSCAAP_Group") or "").strip()
        if not group.isdigit() or not (lo <= int(group) <= hi):
            continue
        name = (row.get("English_name") or "").strip()
        if not name:
            continue
        # "NEI" = not elsewhere included: an aggregate bucket, not a fish.
        if "NEI" in name.upper().split():
            continue
        yield name


_WIKI_REDIRECT = re.compile(r"^\s*#REDIRECT\s*\[\[([^\]|#]+)", re.IGNORECASE)


def _fetch_wikitext(title: str, *, refresh: bool = False, max_hops: int = 3) -> Path:
    """Fetch an article's wikitext, following ``#REDIRECT`` hops."""
    for hop in range(max_hops):
        path = sources.fetch(
            sources.wikitext_url(title),
            filename=f"wikipedia_{title.replace(' ', '_')}.wikitext",
            refresh=refresh,
        )
        match = _WIKI_REDIRECT.match(path.read_text(encoding="utf-8", errors="replace"))
        if not match:
            return path
        title = match.group(1).strip()
        log.info("following Wikipedia redirect to %r", title)
    raise RuntimeError(f"too many Wikipedia redirects (last title: {title!r})")


def _iter_wikipedia(path: Path):
    """Yield fish common names from the Wikipedia list's wikitext bullets."""
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith("*"):
            continue
        entry = _WIKI_BULLET.sub("", line)
        entry = _WIKI_REF.sub(" ", entry)
        entry = _WIKI_TEMPLATE.sub(" ", entry)
        entry = _WIKI_TAG.sub(" ", entry)
        entry = _WIKI_LINK.sub(r"\1", entry)
        entry = _WIKI_ITALIC.sub("", entry)
        entry = _PARENTHETICAL.sub(" ", entry)
        entry = entry.split(" - ")[0].split(" – ")[0]
        entry = entry.strip(" *:;,.")
        if not entry or entry.lower().startswith(("see ", "list of")):
            continue
        if len(entry) > 60:  # prose that slipped past the bullet check
            continue
        yield entry


def build_vocabulary(*, refresh: bool = False) -> dict[str, FishWord]:
    """Fetch both sources and reduce them to a head-noun vocabulary."""
    stoplist = load_stoplist()
    vocab: dict[str, FishWord] = {}

    def add(name: str, source: str) -> None:
        for head in heads_from_name(name, stoplist):
            word = vocab.get(head)
            if word is None:
                word = vocab[head] = FishWord(head=head)
            word.n_species += 1
            word.sources.add(source)
            # Prefer a short, recognisable example ("Rock bass" over
            # "Largemouth black bass").
            if not word.example or len(name) < len(word.example):
                word.example = name

    asfis_path = sources.fetch(sources.ASFIS_URL, refresh=refresh)
    n_asfis = 0
    for name in _iter_asfis(asfis_path):
        add(name, "asfis")
        n_asfis += 1

    wiki_path = _fetch_wikitext(sources.WIKIPEDIA_TITLE, refresh=refresh)
    n_wiki = 0
    for name in _iter_wikipedia(wiki_path):
        add(name, "wikipedia")
        n_wiki += 1

    log.info(
        "fish vocabulary: %d head nouns from %d ASFIS species and %d Wikipedia names",
        len(vocab),
        n_asfis,
        n_wiki,
    )
    return vocab


def write_vocabulary(vocab: dict[str, FishWord], path: Path | None = None) -> Path:
    """Persist the vocabulary to ``data/fish_vocab.csv``."""
    path = path or sources.FISH_VOCAB_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["head_noun", "n_species", "example_name", "sources"])
        for head in sorted(vocab):
            word = vocab[head]
            writer.writerow(
                [word.head, word.n_species, word.example, "+".join(sorted(word.sources))]
            )
    log.info("wrote %s (%d head nouns)", path, len(vocab))
    return path


def read_vocabulary(path: Path | None = None) -> dict[str, FishWord]:
    """Load a previously written vocabulary CSV."""
    path = path or sources.FISH_VOCAB_CSV
    vocab: dict[str, FishWord] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vocab[row["head_noun"]] = FishWord(
                head=row["head_noun"],
                n_species=int(row["n_species"]),
                example=row["example_name"],
                sources=set(row["sources"].split("+")) if row["sources"] else set(),
            )
    return vocab
