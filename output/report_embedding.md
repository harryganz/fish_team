# The All-Star Fish Team (by meaning)

Each surname is scored by its cosine similarity, in GloVe word vectors, to the average of forty common fish words (Trout is scored leave-one-out). A slot goes to the highest `z(fishiness) + 0.5·z(WAR)` among players with fishiness ≥ 0.30.

**Total career WAR: 216.0**

## The team

| Slot | Player | WAR | Seasons | Fishiness | Blend | Nearest fish |
| --- | --- | --: | --: | --: | --: | --- |
| **SP** | Dizzy Trout | 49.9 | 15 | 0.71 | 12.42 | salmon, sturgeon, walleye, catfish, whitefish |
| **RP** | Art Herring | 3.2 | 11 | 0.60 | 9.16 | mackerel, salmon, cod, halibut, sturgeon |
| **C** | Hank Conger | 1.9 | 7 | 0.33 | 4.48 | eel, chub, herring, dace, grouper |
| **1B** | Sid Bream | 11.1 | 12 | 0.61 | 9.58 | carp, crappie, bluegill, tench, perch |
| **2B** | Chico Salmon | 1.0 | 9 | 0.58 | 8.78 | trout, tuna, halibut, herring, sturgeon |
| **3B** | Mel Roach (2B) | -1.5 | 8 | 0.33 | 4.50 | bream, tench, trout, dace, carp |
| **SS** | Bobby Sturgeon | 1.4 | 6 | 0.57 | 8.59 | trout, salmon, herring, whitefish, shad |
| **LF** | Mike Carp (1B) | 1.9 | 6 | 0.57 | 8.66 | bream, trout, catfish, tilapia, perch |
| **CF** | Mike Trout | 91.7 | 16 | 0.71 | 13.64 | salmon, sturgeon, walleye, catfish, whitefish |
| **RF** | Kevin Bass | 14.8 | 14 | 0.40 | 6.07 | crappie, trout, bluegill, catfish, walleye |
| **DH** | Tim Salmon (RF) | 40.6 | 14 | 0.58 | 9.93 | trout, tuna, halibut, herring, sturgeon |

## Fishiest surnames overall

| Player | Pos | WAR | Fishiness | Nearest fish |
| --- | --- | --: | --: | --- |
| Mike Trout | CF | 91.7 | 0.71 | salmon, sturgeon, walleye |
| Dizzy Trout | SP | 49.9 | 0.71 | salmon, sturgeon, walleye |
| Steve Trout | SP | 13.3 | 0.71 | salmon, sturgeon, walleye |
| Sid Bream | 1B | 11.1 | 0.61 | carp, crappie, bluegill |
| Art Herring | RP | 3.2 | 0.60 | mackerel, salmon, cod |
| Tim Salmon | RF | 40.6 | 0.58 | trout, tuna, halibut |
| Chico Salmon | 2B | 1.0 | 0.58 | trout, tuna, halibut |
| Mike Carp | 1B | 1.9 | 0.57 | bream, trout, catfish |
| Bobby Sturgeon | SS | 1.4 | 0.57 | trout, salmon, herring |
| Kevin Bass | RF | 14.8 | 0.40 | crappie, trout, bluegill |
| Anthony Bass | RP | 3.8 | 0.40 | crappie, trout, bluegill |
| Randy Bass | 1B | -1.1 | 0.40 | crappie, trout, bluegill |
| Max Fried | SP | 32.3 | 0.36 | catfish, tuna, salmon |
| Coot Veal | SS | -0.0 | 0.35 | halibut, tuna, swordfish |
| Mel Roach | 2B | -1.5 | 0.33 | bream, tench, trout |
| Hank Conger | C | 1.9 | 0.33 | eel, chub, herring |
| Rob Deer | RF | 13.8 | 0.32 | trout, salmon, sturgeon |
| Carl Yastrzemski | LF | 96.4 | 0.29 | haddock, bream, snapper |
| Mike Yastrzemski | RF | 17.6 | 0.29 | haddock, bream, snapper |
| Preacher Roe | SP | 29.7 | 0.28 | salmon, shad, herring |
| Chaz Roe | RP | 1.2 | 0.28 | salmon, shad, herring |
| Bob Moose | SP | 7.6 | 0.28 | trout, sturgeon, herring |
| Brad Peacock | RP | 4.7 | 0.27 | perch, bass, bream |
| Johnny Peacock | C | 1.1 | 0.27 | perch, bass, bream |
| Chet Lemon | CF | 55.6 | 0.27 | anchovy, tuna, salmon |

## How this was built

- Eligible players: **6954**; in the embedding vocabulary: **6378**
- Vectors: `glove-wiki-gigaword-300`, fetched 2026-10-29T23:15:33+00:00
- Seed fish: trout, salmon, bass, pike, carp, cod, haddock, halibut, herring, mackerel, sardine, anchovy, tuna, perch, walleye, catfish, sturgeon, flounder, sole, snapper, grouper, minnow, eel, shad, smelt, tilapia, marlin, swordfish, barracuda, bluegill, crappie, pollock, whitefish, char, bream, tench, roach, chub, dace, goby

_Report generated 2026-10-29T23:29:19+00:00._
