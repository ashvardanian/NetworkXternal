"""Streaming import of adjacency-list files, one page of edges per round-trip and never a whole file in RAM."""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from itertools import batched
from pathlib import Path
from typing import IO

from networkxternal.base_api import BaseGraph

type WeightedEdge = tuple[int, int, float | None]


def open_text(path: str | Path) -> IO[str]:
    """Opens an edge list, unpacking it on the way through when the name says it is gzipped."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def yield_edges(path: str | Path) -> Iterator[WeightedEdge]:
    """Yields `(source, target, weight)` per row, with `None` where a row carries no weight.

    Fields are separated by whitespace or commas, `#` starts a comment, and a row whose two leading
    fields are not integers is skipped — which is how a header, a blank line and a preamble all go.
    """
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(("#", "%")):
                continue
            fields = line.replace(",", " ").split()
            if len(fields) < 2:
                continue
            try:
                source, target = int(fields[0]), int(fields[1])
                weight = float(fields[2]) if len(fields) > 2 else None
            except ValueError:
                continue
            yield source, target, weight


def import_edges(graph: BaseGraph, path: str | Path, weight: str = "weight") -> int:
    """Imports every edge of a file into `graph`, in pages of `graph.PAGE`.

    Returns:
        The number of edges read from the file.
    """
    imported = 0
    for page in batched(yield_edges(path), graph.PAGE):
        sources = [source for source, _, _ in page]
        targets = [target for _, target, _ in page]
        weights = [held for _, _, held in page]
        columns = {weight: weights} if any(held is not None for held in weights) else None
        graph.add_edges_from_arrays(sources, targets, columns=columns)
        imported += len(page)
    return imported
