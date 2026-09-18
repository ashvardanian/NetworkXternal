"""The graphs a run is measured on: a synthetic stream generated on the fly, or an edge list on disk.

Nothing ships in the repository — a real dataset is a `data/<name>` symlink into the shared filesystem,
and everything smaller is generated, so no fixture file has to be kept in sync with the code.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from networkxternal.parsing import yield_edges

type WeightedEdge = tuple[int, int, float | None]


@dataclass(frozen=True)
class Dataset:
    """One graph to measure against, named for the report and for the store's database."""

    name: str
    """Lowercase, dash-separated, scale last, as every dataset on the shared filesystem is named."""

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
            # A Pareto draw for the source, so degrees come out as uneven as a real graph's.
            source = min(int(generator.paretovariate(1.2)), self.nodes)
            target = generator.randrange(1, self.nodes + 1)
            if source != target:
                yield source, target, round(generator.uniform(0.1, 10.0), 3)


SYNTHETIC = (
    Dataset("synthetic-10k", nodes=2_000, edges=10_000),
    Dataset("synthetic-100k", nodes=20_000, edges=100_000),
)
"""The generated graphs a default run measures, small enough for a laptop or a capped container."""


def from_path(path: str | Path) -> Dataset:
    """A dataset backed by an edge list on disk, named after the file it comes from."""
    path = Path(path)
    return Dataset(path.stem.lower(), nodes=0, edges=0, path=path)
