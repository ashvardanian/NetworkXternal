"""The graphs a run is measured on: real edge lists from the catalogue, or a generated stream for a smoke test.

Nothing ships in the repository, and a graph is read from exactly one place: `data/<name>`, a symlink
the machine points at storage with room for it. Names are lowercase with dashes and the scale last,
so the link says what it holds. The generated graphs stay for quick checks, and their skew is nothing
like a real graph's — one vertex holds a third of the edges, which flatters everything but hubs.
"""

from __future__ import annotations

import random
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from networkxternal.parsing import yield_edges

type WeightedEdge = tuple[int, int, float | None]

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
"""The only place the benchmark looks for a graph: `data/<name>`, which git ignores but for its notes."""


@dataclass(frozen=True)
class Catalogued:
    """A real graph that can be fetched, with the shape it has once unpacked."""

    name: str
    """Lowercase, dash-separated, scale last, as every dataset on the shared filesystem is named."""

    url: str
    """Where the edge list is downloaded from."""

    vertices: int
    """How many vertices the published graph holds."""

    edges: int
    """How many edges the published graph holds."""

    download_bytes: int
    """The compressed download size, measured rather than quoted."""

    fetched: bool = True
    """Whether `download` will fetch it, or it is listed for a machine with room for it."""

    @property
    def path(self) -> Path:
        """Where the edge list is read from, which is a `data/<name>` symlink to wherever it is stored."""
        return DATA_ROOT / self.name / Path(self.url).name


CATALOGUE = (
    Catalogued("facebook-88k", "https://snap.stanford.edu/data/facebook_combined.txt.gz", 4_039, 88_234, 218_154),
    Catalogued("astroph-198k", "https://snap.stanford.edu/data/ca-AstroPh.txt.gz", 18_772, 198_110, 1_484_474),
    Catalogued("enron-184k", "https://snap.stanford.edu/data/email-Enron.txt.gz", 36_692, 183_831, 1_111_264),
    Catalogued(
        "amazon-1m",
        "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        334_863,
        925_872,
        4_496_694,
    ),
    Catalogued(
        "youtube-3m",
        "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        1_134_890,
        2_987_624,
        10_611_968,
    ),
    Catalogued("google-5m", "https://snap.stanford.edu/data/web-Google.txt.gz", 875_713, 5_105_039, 21_168_784),
    Catalogued(
        "pokec-31m",
        "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
        1_632_803,
        30_622_564,
        132_484_918,
    ),
    Catalogued(
        "livejournal-35m",
        "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
        3_997_962,
        34_681_189,
        124_271_581,
    ),
    Catalogued(
        "orkut-117m",
        "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
        3_072_441,
        117_185_083,
        447_303_744,
    ),
    Catalogued(
        "friendster-1806m",
        "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz",
        65_608_366,
        1_806_067_135,
        9_371_303_564,
        fetched=False,
    ),
    Catalogued(
        "sinaweibo-261m",
        "https://nrvis.com/download/data/soc/soc-sinaweibo.zip",
        58_655_849,
        261_321_071,
        1_700_000_000,
        fetched=False,
    ),
    Catalogued(
        "clueweb09-7940m",
        "https://nrvis.com/download/data/web/web-ClueWeb09.zip",
        1_684_868_322,
        7_939_635_651,
        40_000_000_000,
        fetched=False,
    ),
    Catalogued(
        "wdc2012-128b",
        "https://data.commoncrawl.org/projects/hyperlinkgraph/cc-main-2012/index.html",
        3_563_000_000,
        128_000_000_000,
        2_500_000_000_000,
        fetched=False,
    ),
)
"""Every graph a run can be pointed at, from a laptop smoke test up to the multi-terabyte tier."""


@dataclass(frozen=True)
class Dataset:
    """One graph to measure against, named for the report and for the store's database."""

    name: str
    """Lowercase, dash-separated, scale last."""

    nodes: int
    """How many distinct vertices a generated graph draws from."""

    edges: int
    """How many edges a generated stream yields."""

    path: Path | None = None
    """The edge list on disk, or `None` when the graph is generated."""

    seed: int = 42
    """The seed a generated graph is reproducible by."""

    def stream(self) -> Iterator[WeightedEdge]:
        """Yields the edges of the graph, holding one at a time either way."""
        if self.path is not None:
            yield from yield_edges(self.path)
            return
        generator = random.Random(self.seed)
        for _ in range(self.edges):
            # A Pareto draw for the source, so degrees come out uneven — far more so than a real graph's.
            source = min(int(generator.paretovariate(1.2)), self.nodes)
            target = generator.randrange(1, self.nodes + 1)
            if source != target:
                yield source, target, round(generator.uniform(0.1, 10.0), 3)


SYNTHETIC = (
    Dataset("synthetic-10k", nodes=2_000, edges=10_000),
    Dataset("synthetic-100k", nodes=20_000, edges=100_000),
)
"""Generated graphs for a smoke test, whose skew is a hub stress case rather than a realistic one."""


BY_NAME = {entry.name: entry for entry in CATALOGUE}
"""Every catalogued graph by its name, so a lookup is a lookup."""


def find(name: str) -> Catalogued:
    """The catalogue entry a name addresses."""
    if name not in BY_NAME:
        raise KeyError(f"No catalogued dataset is named {name!r}; known: {', '.join(BY_NAME)}")
    return BY_NAME[name]


def download(name: str) -> Path:
    """Fetches a catalogued graph into the storage `data/<name>` already points at.

    Where the bytes live is the machine's decision, not the repository's, so the link comes first.
    """
    entry = find(name)
    if not entry.fetched:
        raise ValueError(f"{name} is listed for a machine with room for it; fetch it yourself from {entry.url}")
    if not entry.path.parent.exists():
        raise FileNotFoundError(wiring_instructions(entry))
    if not entry.path.exists():
        urllib.request.urlretrieve(entry.url, entry.path)
    return entry.path


def wiring_instructions(entry: Catalogued) -> str:
    """What to run so `data/<name>` points at storage with room for the graph."""
    return (
        f"data/{entry.name} is not wired up. Point it at storage holding "
        f"{entry.download_bytes / 1e6:,.0f} MB compressed:\n"
        f"    mkdir -p <storage>/{entry.name}\n"
        f"    ln -s <storage>/{entry.name} {DATA_ROOT / entry.name}"
    )


def from_catalogue(name: str) -> Dataset:
    """A dataset backed by a catalogued graph, downloaded into the storage its link names."""
    entry = find(name)
    return Dataset(entry.name, nodes=entry.vertices, edges=entry.edges, path=download(name))


def from_path(path: str | Path) -> Dataset:
    """A dataset backed by an edge list on disk, named after the file it comes from."""
    path = Path(path)
    return Dataset(path.stem.lower(), nodes=0, edges=0, path=path)
