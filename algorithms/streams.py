"""The edge streams every algorithm consumes, and the one-pass sampler two of them share."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched, islice

from networkxternal.base_api import BaseGraph, Role

type Arc = tuple[int, int, float]
"""A source, a target and a weight, which is what every pass consumes."""


def asked(nodes: Iterable[int]) -> Sequence[int]:
    """The vertices as something indexable, without boxing an `array` into a list of Python integers."""
    return nodes if isinstance(nodes, Sequence) else list(nodes)


def adjacency(graph: BaseGraph, nodes: Iterable[int], role: Role) -> Iterator[tuple[int, int, int]]:
    """Yields `(vertex, neighbour, edge)` for every edge incident to `nodes`, one page of rows at a time."""
    for node, (source, target, edge) in graph.adjacent_edges(asked(nodes), role):
        yield node, (target if source == node else source), edge


def weighted_adjacency(graph: BaseGraph, nodes: Iterable[int], role: Role, weight: str | None) -> Iterator[Arc]:
    """Yields `(vertex, neighbour, weight)` for every incident edge, one page of weights at a time.

    With `weight` unset every edge counts as one and no attribute is read, which is decided here
    rather than inside the loop.
    """
    for page in batched(graph.adjacent_edges(asked(nodes), role), graph.PAGE):
        reached = [(node, target if source == node else source) for node, (source, target, _) in page]
        if weight is None:
            yield from ((node, other, 1.0) for node, other in reached)
            continue
        weights = graph.edge_weights([edge for _, (_, _, edge) in page], weight)
        for (node, other), held in zip(reached, weights, strict=True):
            yield node, other, 1.0 if held is None else held


def scan_unit_edges(graph: BaseGraph) -> Iterator[Arc]:
    """Yields `(source, target, 1.0)` for every stored edge, reading no attribute document at all."""
    for source, target, _ in graph.scan_edges():
        yield source, target, 1.0


def scan_weighted_edges(graph: BaseGraph, weight: str) -> Iterator[Arc]:
    """Yields `(source, target, weight)` for every stored edge, a page of weights at a time.

    The weight comes through `edge_weights`, which a store holding it in a typed column answers from
    that column rather than by parsing a document per edge.
    """
    for page in batched(graph.scan_edges(), graph.PAGE):
        weights = graph.edge_weights([edge for _, _, edge in page], weight)
        for (source, target, _), held in zip(page, weights, strict=True):
            yield source, target, 1.0 if held is None else held


def mirrored(stream: Iterable[Arc]) -> Iterator[Arc]:
    """Yields every edge of a stream in both orientations, a self-loop once."""
    for source, target, held in stream:
        yield source, target, held
        if source != target:
            yield target, source, held


def scan_arcs(graph: BaseGraph, weight: str | None) -> Iterator[Arc]:
    """Every edge as an arc pushing mass, mirrored unless the graph is directed, chosen before the sweep."""
    stream = scan_unit_edges(graph) if weight is None else scan_weighted_edges(graph, weight)
    return stream if graph.is_directed() else mirrored(stream)


def reservoir[Item](stream: Iterable[Item], count: int, seed: int | None) -> list[Item]:
    """A uniform sample of `count` items over one pass, holding the reservoir and nothing else.

    The reservoir fills before the loop, so the walk carries no test for whether it is full yet.
    """
    generator = random.Random(seed)
    walk = iter(stream)
    held = list(islice(walk, count))
    for seen, item in enumerate(walk, start=count):
        index = generator.randrange(seen + 1)
        if index < count:
            held[index] = item
    return held
