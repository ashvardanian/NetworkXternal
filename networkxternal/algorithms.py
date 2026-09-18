"""Semi-external graph algorithms: vertex state stays in RAM, adjacency is pulled from the store page by page.

NetworkX walks one vertex at a time, which costs one round-trip per step against an external store.
Every traversal here streams the edges incident to a whole frontier instead, so a run costs a
round-trip per page of edges rather than per vertex, and holds vertex state and one page of rows.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched

from networkxternal.base_api import AttributeStore, BaseGraph, DegreeView, Role


def adjacency(graph: BaseGraph, nodes: Iterable[int], role: Role) -> Iterator[tuple[int, int, int]]:
    """Yields `(vertex, neighbour, edge)` for every edge incident to `nodes`, one page of rows at a time."""
    for node, (source, target, edge) in graph.adjacent_edges(list(nodes), role):
        yield node, (target if source == node else source), edge


def weighted_adjacency(
    graph: BaseGraph, nodes: Iterable[int], role: Role, weight: str | None
) -> Iterator[tuple[int, int, float]]:
    """Yields `(vertex, neighbour, weight)` for every incident edge, reading one page of documents at a time.

    With `weight` unset every edge counts as one, and no attribute document is read at all.
    """
    for page in batched(graph.adjacent_edges(list(nodes), role), graph.PAGE):
        if weight is None:
            for node, (source, target, _) in page:
                yield node, (target if source == node else source), 1.0
            continue
        attributes = graph.read_documents(AttributeStore.EDGES, [edge for _, (_, _, edge) in page])
        for (node, (source, target, _)), found in zip(page, attributes, strict=True):
            yield node, (target if source == node else source), float(found.get(weight, 1))


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
        reached = {
            neighbour for _, neighbour, _ in adjacency(graph, frontier, graph.outgoing_role) if neighbour not in visited
        }
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
        for node, neighbour, _ in adjacency(graph, nodes, Role.ANY):
            if neighbour in labels and labels[neighbour] < labels[node]:
                labels[node] = labels[neighbour]
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
        totals = dict.fromkeys(ranks, 0.0)
        for node, _, held in weighted_adjacency(graph, nodes, graph.outgoing_role, weight):
            totals[node] += held
        dangling = sum(ranks[node] for node, total in totals.items() if not total)
        for node, neighbour, held in weighted_adjacency(graph, nodes, graph.outgoing_role, weight):
            if totals[node] and neighbour in pushed:
                pushed[neighbour] += ranks[node] * held / totals[node]
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
        for _, neighbour, _ in adjacency(graph, peeled, Role.ANY):
            if neighbour in remaining:
                remaining[neighbour] -= 1
    return cores


def triangle_counts(graph: BaseGraph, nodes: Iterable[int] | None = None) -> dict[int, int]:
    """How many triangles every vertex takes part in, by intersecting neighbourhoods a page at a time.

    Holds the neighbourhood of the wanted vertices and of the vertices one step away from them.
    """
    wanted = set(graph.scan_nodes()) if nodes is None else set(nodes)
    neighbourhoods: dict[int, set[int]] = {node: set() for node in wanted}
    for node, neighbour, _ in adjacency(graph, list(wanted), Role.ANY):
        if neighbour != node:
            neighbourhoods[node].add(neighbour)
    reached = {neighbour for neighbours in neighbourhoods.values() for neighbour in neighbours}
    missing = [neighbour for neighbour in reached if neighbour not in neighbourhoods]
    for node in missing:
        neighbourhoods[node] = set()
    for node, neighbour, _ in adjacency(graph, missing, Role.ANY):
        if neighbour != node:
            neighbourhoods[node].add(neighbour)
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
