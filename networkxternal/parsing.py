"""Streaming import of adjacency-list files, one page of edges per round-trip and never a whole file in RAM."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from itertools import batched
from pathlib import Path

from networkxternal.base_api import BaseGraph

type WeightedEdge = tuple[int, int, float | None]


def yield_edges(path: str | Path, delimiter: str = ",") -> Iterator[WeightedEdge]:
    """Yields `(source, target, weight)` per row, with `None` where a row carries no weight.

    A first row whose two leading fields are not integers is taken for a header and skipped.
    """
    with open(path, newline="") as handle:
        rows = csv.reader(handle, delimiter=delimiter)
        for row in rows:
            if len(row) < 2:
                continue
            try:
                source, target = int(row[0]), int(row[1])
            except ValueError:
                continue
            weight = float(row[2]) if len(row) > 2 and row[2] else None
            yield source, target, weight


def import_edges(
    graph: BaseGraph,
    path: str | Path,
    delimiter: str = ",",
    weight: str = "weight",
) -> int:
    """Imports every edge of a file into `graph`, in pages of `graph.PAGE`.

    Returns:
        The number of edges read from the file.
    """
    imported = 0
    for page in batched(yield_edges(path, delimiter), graph.PAGE):
        sources = [source for source, _, _ in page]
        targets = [target for _, target, _ in page]
        weights = [held for _, _, held in page]
        columns = {weight: weights} if any(held is not None for held in weights) else None
        graph.add_edges_from_arrays(sources, targets, columns=columns)
        imported += len(page)
    return imported
