"""Semi-external graph algorithms: vertex state stays in RAM, adjacency is pulled from the store page by page.

NetworkX walks one vertex at a time, which costs one round-trip per step against an external store.
Every traversal here expands a whole frontier in one `find_edges` call instead, so a run costs a
round-trip per level rather than per vertex, and holds at most a frontier and one value per vertex.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched

from networkxternal.base_api import AttributeStore, BaseGraph, DegreeView, Role


def adjacency_pages(graph: BaseGraph, nodes: Iterable[int], role: Role) -> Iterator[tuple[int, list[tuple[int, int]]]]:
    """Yields every vertex of `nodes` with the `(neighbour, edge)` pairs its edges reach, one page at a time."""
    for page in batched(nodes, graph.PAGE):
        found = graph.find_edges(page, role)
        for node, triples in zip(page, found, strict=True):
            yield node, [(target if source == node else source, edge) for source, target, edge in triples]


def weighted_adjacency_pages(
    graph: BaseGraph, nodes: Iterable[int], role: Role, weight: str | None
) -> Iterator[tuple[int, list[tuple[int, float]]]]:
    """Yields every vertex with the `(neighbour, weight)` pairs its edges reach, one page of reads per page.

    With `weight` unset every edge counts as one, and no attribute document is read at all.
    """
    for page in batched(nodes, graph.PAGE):
        found = graph.find_edges(page, role)
        if weight is None:
            for node, triples in zip(page, found, strict=True):
                yield node, [(target if source == node else source, 1.0) for source, target, _ in triples]
            continue
        identifiers = [edge for triples in found for _, _, edge in triples]
        attributes = iter(graph.read_documents(AttributeStore.EDGES, identifiers))
        for node, triples in zip(page, found, strict=True):
            reached = [
                (target if source == node else source, float(next(attributes).get(weight, 1)))
                for source, target, _ in triples
            ]
            yield node, reached


def breadth_first_layers(
    graph: BaseGraph,
    sources: int | Iterable[int],
    cutoff: int | None = None,
) -> Iterator[set[int]]:
    """Yields the vertices at distance 0, 1, 2 … from `sources`, one round-trip per layer.

    Holds the visited set and the current frontier, which is proportional to the reached component.
    """
    frontier = {sources} if isinstance(sources, int) else set(sources)
    visited = set(frontier)
    depth = 0
    while frontier:
        yield frontier
        depth += 1
        if cutoff is not None and depth > cutoff:
            return
        reached: set[int] = set()
        for _, neighbours in adjacency_pages(graph, frontier, graph.outgoing_role):
            reached.update(neighbour for neighbour, _ in neighbours if neighbour not in visited)
        visited |= reached
        frontier = reached


def shortest_path_lengths(
    graph: BaseGraph,
    source: int,
    cutoff: int | None = None,
) -> dict[int, int]:
    """The number of edges from `source` to every vertex it reaches, within `cutoff` steps."""
    lengths: dict[int, int] = {}
    for depth, layer in enumerate(breadth_first_layers(graph, source, cutoff)):
        lengths.update(dict.fromkeys(layer, depth))
    return lengths


def neighbors_of_neighbors(graph: BaseGraph, node: int, include_neighbors: bool = False) -> set[int]:
    """The vertices two steps from `node`, with or without the ones a single step reaches."""
    layers = list(breadth_first_layers(graph, node, cutoff=2))
    reached = layers[2] if len(layers) > 2 else set()
    if include_neighbors and len(layers) > 1:
        reached |= layers[1]
    return reached - {node}


def connected_components(graph: BaseGraph, iterations: int = 1024) -> dict[int, int]:
    """The component every vertex belongs to, named by the smallest vertex in it.

    Label propagation over the whole vertex set, one sweep per round-trip page, until labels settle.
    Holds one label per vertex, never an adjacency list of the graph, and raises rather than reporting
    labels that have not settled within `iterations` sweeps.
    """
    labels = {node: node for node in graph.scan_nodes()}
    nodes = list(labels)
    for _ in range(iterations):
        changed = False
        for node, neighbours in adjacency_pages(graph, nodes, Role.ANY):
            smallest = min(
                (labels[neighbour] for neighbour, _ in neighbours if neighbour in labels), default=labels[node]
            )
            if smallest < labels[node]:
                labels[node] = smallest
                changed = True
        if not changed:
            return labels
    raise RuntimeError(f"Labels did not settle within {iterations} sweeps")


def pagerank(
    graph: BaseGraph,
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-6,
    weight: str | None = "weight",
) -> dict[int, float]:
    """The PageRank of every vertex, pushed along out-edges, one round-trip page per sweep.

    Holds two floats per vertex; the edges themselves are never materialized. With `weight` unset
    every edge carries the same mass, which is what NetworkX calls `weight=None`.
    """
    ranks = {node: 0.0 for node in graph.scan_nodes()}
    if not ranks:
        return ranks
    share = 1.0 / len(ranks)
    ranks = dict.fromkeys(ranks, share)
    nodes = list(ranks)
    for _ in range(iterations):
        pushed = dict.fromkeys(ranks, 0.0)
        dangling = 0.0
        for node, neighbours in weighted_adjacency_pages(graph, nodes, graph.outgoing_role, weight):
            total = sum(held for _, held in neighbours)
            if not total:
                dangling += ranks[node]
                continue
            for neighbour, held in neighbours:
                if neighbour in pushed:
                    pushed[neighbour] += ranks[node] * held / total
        leaked = damping * dangling * share + (1.0 - damping) * share
        updated = {node: damping * mass + leaked for node, mass in pushed.items()}
        drift = sum(abs(updated[node] - ranks[node]) for node in ranks)
        ranks = updated
        if drift < len(ranks) * tolerance:
            break
    return ranks


def core_numbers(graph: BaseGraph) -> dict[int, int]:
    """The largest `k` whose k-core holds every vertex, by peeling the lowest degree first.

    Holds one degree per vertex and pulls the neighbourhood of a peeled vertex only.
    """
    remaining = {node: degree for node, degree in graph.degree}
    cores: dict[int, int] = {}
    level = 0
    while remaining:
        level = max(level, min(remaining.values()))
        peeled = [node for node, degree in remaining.items() if degree <= level]
        for node in peeled:
            cores[node] = level
            del remaining[node]
        for _, neighbours in adjacency_pages(graph, peeled, Role.ANY):
            for neighbour, _ in neighbours:
                if neighbour in remaining:
                    remaining[neighbour] -= 1
    return cores


def triangle_counts(graph: BaseGraph, nodes: Iterable[int] | None = None) -> dict[int, int]:
    """How many triangles every vertex takes part in, by intersecting neighbourhoods a page at a time.

    Holds the neighbourhood of the wanted vertices and of the vertices one step away from them.
    """
    wanted = set(graph.scan_nodes()) if nodes is None else set(nodes)
    neighbourhoods = {
        node: {neighbour for neighbour, _ in neighbours} - {node}
        for node, neighbours in adjacency_pages(graph, wanted, Role.ANY)
    }
    reached = {neighbour for neighbours in neighbourhoods.values() for neighbour in neighbours}
    missing = [neighbour for neighbour in reached if neighbour not in neighbourhoods]
    for node, neighbours in adjacency_pages(graph, missing, Role.ANY):
        neighbourhoods[node] = {neighbour for neighbour, _ in neighbours} - {node}
    return {
        node: sum(len(neighbours & neighbourhoods[neighbour]) for neighbour in neighbours) // 2
        for node, neighbours in neighbourhoods.items()
        if node in wanted
    }


def degree_histogram(graph: BaseGraph, role: Role = Role.ANY) -> dict[int, int]:
    """How many vertices hold each degree, streamed rather than sorted into a list."""
    histogram: dict[int, int] = {}
    for _, degree in DegreeView(graph, role):
        histogram[degree] = histogram.get(degree, 0) + 1
    return histogram


def sample_nodes(graph: BaseGraph, count: int, seed: int | None = None) -> list[int]:
    """A uniform sample of `count` vertices, reservoir-sampled over one pass of the vertex scan."""
    generator = random.Random(seed)
    reservoir: list[int] = []
    for seen, node in enumerate(graph.scan_nodes()):
        if len(reservoir) < count:
            reservoir.append(node)
            continue
        index = generator.randrange(seen + 1)
        if index < count:
            reservoir[index] = node
    return reservoir


def sample_edges(graph: BaseGraph, count: int, seed: int | None = None) -> list[tuple[int, int, int]]:
    """A uniform sample of `count` edges, reservoir-sampled over one pass of the edge scan."""
    generator = random.Random(seed)
    reservoir: list[tuple[int, int, int]] = []
    for seen, triple in enumerate(graph.edge_triples(None)):
        if len(reservoir) < count:
            reservoir.append(triple)
            continue
        index = generator.randrange(seen + 1)
        if index < count:
            reservoir[index] = triple
    return reservoir


def relabel_dense(nodes: Sequence[int]) -> dict[int, int]:
    """A dense 0-based numbering of sparse vertex identifiers, for matrices and adjacency arrays."""
    return {node: index for index, node in enumerate(nodes)}
