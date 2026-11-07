"""Vocabulary construction: taxonomic filtering, name cleanup, head-noun extraction."""

from __future__ import annotations

import csv
import io
import zipfile

import pytest

from fish_team import fish

STOPLIST = fish.load_stoplist()

ASFIS_HEADER = [
    "ISSCAAP_Group ",
    "Taxonomic_Code",
    "Alpha3_Code",
    "Scientific_Name",
    "English_name",
    "French_name",
]


def _asfis_zip(tmp_path, rows):
    """Build a miniature ASFIS archive: a CSV inside a zip, as FAO ships it."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(ASFIS_HEADER)
    writer.writerows(rows)
    path = tmp_path / "ASFIS_sp.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ASFIS_sp_test.csv", buffer.getvalue())
    return path


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # The regression that matters: rstrip("'s") turns "bass" into "ba".
        ("bass", "bass"),
        ("anchovies", "anchovy"),
        ("catfishes", "catfish"),
        ("boxes", "box"),
        ("groupers", "grouper"),
        # Already singular despite the ending.
        ("species", "species"),
        ("sole", "sole"),
        ("cod", "cod"),
        ("wrasse", "wrasse"),
    ],
)
def test_singularize(word, expected):
    assert fish.singularize(word) == expected


def test_deaccent():
    assert fish.deaccent("Rubén Sierra") == "Ruben Sierra"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Rock bass", ["bass"]),
        ("Largemouth black bass", ["bass"]),
        ("Golden trout", ["trout"]),
        ("Japanese rice fish", ["rice"]),
        # An explicit synonym in parentheses is a usable second name.
        ("Saithe(=Pollock)", ["pollock", "saithe"]),
        # A taxonomic parenthetical is not.
        ("Anchovy (family Engraulidae)", ["anchovy"]),
        # Comma- and slash-separated alternates both count.
        ("Sprat/brisling", ["sprat", "brisling"]),
    ],
)
def test_heads_from_name(name, expected):
    assert fish.heads_from_name(name, STOPLIST) == expected


@pytest.mark.parametrize(
    "name",
    [
        "White-eye",            # headed by a body part
        "Common silver belly",  # headed by a body part
        "Marine fishes NEI",    # an aggregate, not a species
        "Flying fish",          # headed by a participle once "fish" is dropped
        "Diversity of fish",    # prose from the Wikipedia article
    ],
)
def test_names_with_no_real_head_are_dropped(name):
    assert fish.heads_from_name(name, STOPLIST) == []


def test_stoplist_blocks_the_marine_false_positive():
    """Pedro Martinez ~ "Marine fishes" is prevented here, not by the scorer.

    "martinez" vs "marine" scores 0.83, which clears the default floor -- the pair is
    kept out by never admitting "marine" as a fish word in the first place.
    """
    assert "marine" in STOPLIST
    assert fish.heads_from_name("Marine fishes", STOPLIST) == []


def test_stoplist_covers_hyphenated_compounds():
    """One "eye" entry should cover every *-eye compound."""
    assert fish.head_noun("Blue-eye", STOPLIST) is None
    assert fish.head_noun("Shortnose greeneye", STOPLIST) == "greeneye"


def test_asfis_keeps_fishes_and_excludes_everything_else(tmp_path):
    path = _asfis_zip(
        tmp_path,
        [
            ["13", "1", "RBS", "Ambloplites rupestris", "Rock bass", ""],
            ["33", "2", "TUN", "Thunnus alalunga", "Albacore", ""],
            # ISSCAAP 45 is crustaceans, 56 molluscs, 61 aquatic mammals, 71 plants.
            ["45", "3", "CRA", "Cancer pagurus", "Edible crab", ""],
            ["56", "4", "SQU", "Loligo vulgaris", "European squid", ""],
            ["61", "5", "WHA", "Balaenoptera musculus", "Blue whale", ""],
            ["71", "6", "KEL", "Laminaria digitata", "Kelp", ""],
        ],
    )
    names = list(fish._iter_asfis(path))
    assert names == ["Rock bass", "Albacore"]


def test_asfis_drops_nei_aggregates(tmp_path):
    path = _asfis_zip(
        tmp_path,
        [
            ["33", "1", "GRX", "Epinephelus spp", "Groupers NEI", ""],
            ["33", "2", "TUN", "Thunnus alalunga", "Albacore", ""],
            ["33", "3", "XXX", "Nothing", "", ""],  # no English name
        ],
    )
    assert list(fish._iter_asfis(path)) == ["Albacore"]


def test_wikipedia_wikitext_parsing(tmp_path):
    path = tmp_path / "wiki.wikitext"
    path.write_text(
        "\n".join(
            [
                "== A ==",
                "* [[Albacore]]",
                "* [[African glass catfish]] (''Pareutropius debauwi'')",
                "* [[Bluegill|bluegill]]<ref name=x>cite</ref>",
                "* {{anchor|M}}[[Muskellunge]]",
                "Not a bullet line.",
                "* See [[List of fish families]]",
            ]
        ),
        encoding="utf-8",
    )
    assert list(fish._iter_wikipedia(path)) == [
        "Albacore",
        "African glass catfish",
        "bluegill",
        "Muskellunge",
    ]


def test_wikipedia_redirects_are_detected():
    assert fish._WIKI_REDIRECT.match("#REDIRECT [[List of fish by common name]]")
    assert not fish._WIKI_REDIRECT.match("* [[Albacore]]")


def test_vocabulary_round_trip(tmp_path):
    vocab = {
        "bass": fish.FishWord("bass", n_species=30, example="Rock bass", sources={"asfis"}),
        "trout": fish.FishWord(
            "trout", n_species=28, example="Trout", sources={"asfis", "wikipedia"}
        ),
    }
    path = fish.write_vocabulary(vocab, tmp_path / "vocab.csv")
    assert fish.read_vocabulary(path) == vocab
