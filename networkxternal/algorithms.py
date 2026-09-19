"""Semi-external graph algorithms: vertex state stays in RAM, edges stream out of the store in stored order.

NetworkX walks one vertex at a time, which costs one round-trip per step against an external store.
Every algorithm here is a scatter over an edge stream instead, so a sweep costs one sequential pass over
the edges and holds a fixed number of bytes per vertex — two `array` slots rather than a dict of floats.
"""

from __future__ import annotations

import random
from array import array
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from enum import StrEnum
from itertools import batched, compress, islice

from networkxternal.base_api import BaseGraph, DegreeView, Role, Triple

BOTTOM_UP_FRACTION = 0.1
"""The share of the vertex count a frontier must pass before a layer is found by scanning every edge instead."""

BOTTOM_UP_FLOOR = 4_096
"""How large a frontier must be before the vertex count is worth asking for, which some stores scan to answer."""


# region Edge Streams


def adjacency(graph: BaseGraph, nodes: Iterable[int], role: Role) -> Iterator[tuple[int, int, int]]:
    """Yields `(vertex, neighbour, edge)` for every edge incident to `nodes`, one page of rows at a time."""
    for node, (source, target, edge) in graph.adjacent_edges(list(nodes), role):
        yield node, (target if source == node else source), edge


def scan_unit_edges(graph: BaseGraph) -> Iterator[tuple[int, int, float]]:
    """Yields `(source, target, 1.0)` for every stored edge, reading no attribute document at all."""
    for source, target, _ in graph.scan_edges():
        yield source, target, 1.0


def scan_weighted_edges(graph: BaseGraph, weight: str) -> Iterator[tuple[int, int, float]]:
    """Yields `(source, target, weight)` for every stored edge, a page of weights at a time.

    The weight comes through `edge_weights`, which a store holding it in a typed column answers from
    that column rather than by parsing a document per edge.
    """
    for page in batched(graph.scan_edges(), graph.PAGE):
        weights = graph.edge_weights([edge for _, _, edge in page], weight)
        for (source, target, _), held in zip(page, weights, strict=True):
            yield source, target, 1.0 if held is None else held


def mirrored(stream: Iterable[tuple[int, int, float]]) -> Iterator[tuple[int, int, float]]:
    """Yields every edge of a stream in both orientations, a self-loop once."""
    for source, target, held in stream:
        yield source, target, held
        if source != target:
            yield target, source, held


def scan_arcs(graph: BaseGraph, weight: str | None) -> Iterator[tuple[int, int, float]]:
    """Every edge as an arc pushing mass, mirrored unless the graph is directed, chosen before the sweep."""
    stream = scan_unit_edges(graph) if weight is None else scan_weighted_edges(graph, weight)
    return stream if graph.is_directed() else mirrored(stream)


# endregion Edge Streams

# region Traversal


def expand_top_down(graph: BaseGraph, frontier: set[int], visited: set[int]) -> set[int]:
    """The unvisited neighbours of a small frontier, pulled by adjacency, one page of rows at a time."""
    return {
        neighbour for _, neighbour, _ in adjacency(graph, frontier, graph.outgoing_role) if neighbour not in visited
    }


def expand_bottom_up(graph: BaseGraph, frontier: set[int], visited: set[int]) -> set[int]:
    """The unvisited neighbours of a wide frontier, found in one sequential pass over every edge."""
    return {target for source, target, _ in scan_arcs(graph, None) if source in frontier and target not in visited}


def breadth_first_layers(
    graph: BaseGraph,
    sources: int | Iterable[int],
    cutoff: int | None = None,
) -> Iterator[set[int]]:
    """Yields the vertices at distance 0, 1, 2 … from `sources`, one round-trip page per layer.

    Holds the visited set and the current frontier, both proportional to the reached component, and
    switches from pulling a frontier's adjacency to scanning every edge once the frontier passes
    `BOTTOM_UP_FRACTION` of the vertex count.
    """
    frontier = {sources} if isinstance(sources, int) else set(sources)
    visited = set(frontier)
    threshold: float | None = None
    depth = 0
    while frontier:
        yield frontier
        depth += 1
        if cutoff is not None and depth > cutoff:
            return
        # The vertex count is asked for only once a frontier is large enough for the answer to matter,
        # since a store without a counter answers it by scanning every key.
        if threshold is None and len(frontier) >= BOTTOM_UP_FLOOR:
            threshold = BOTTOM_UP_FRACTION * graph.number_of_nodes()
        large = threshold is not None and len(frontier) > threshold
        expand = expand_bottom_up if large else expand_top_down
        reached = expand(graph, frontier, visited)
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


class Reach(StrEnum):
    """Which vertices a two-step lookup reports."""

    SECOND_STEP = "second-step"
    """Only the vertices exactly two steps away."""

    BOTH_STEPS = "both-steps"
    """The vertices one or two steps away."""


def neighbors_of_neighbors(graph: BaseGraph, node: int, reach: Reach = Reach.SECOND_STEP) -> set[int]:
    """The vertices two steps from `node`, and the ones a single step reaches when `reach` asks for them."""
    layers = list(breadth_first_layers(graph, node, cutoff=2))
    wanted = layers[1:3] if reach is Reach.BOTH_STEPS else layers[2:3]
    return set().union(*wanted) - {node} if wanted else set()


# endregion Traversal

# region Components


def find_root(parents: array, index: int) -> int:
    """The representative of the set holding `index`, compressing the path walked to reach it."""
    root = index
    while parents[root] != root:
        root = parents[root]
    while parents[index] != root:
        parents[index], index = root, parents[index]
    return root


def join_sets(parents: array, depths: array, left: int, right: int) -> None:
    """Joins the two sets holding `left` and `right`, hanging the shallower tree under the deeper one."""
    left, right = find_root(parents, left), find_root(parents, right)
    if left == right:
        return
    if depths[left] < depths[right]:
        left, right = right, left
    parents[right] = left
    if depths[left] == depths[right]:
        depths[left] += 1


def connected_components(graph: BaseGraph) -> dict[int, int]:
    """The component every vertex belongs to, named by the smallest vertex in it.

    Union-find over one sequential pass of the edge stream, which converges in that single pass however
    long the paths are. Holds three 8-byte slots per vertex and one page of edge rows, never an adjacency
    list and never a second sweep.
    """
    order = relabel_dense(list(graph.scan_nodes()))
    count = len(order)
    names = array("q", order)
    parents = array("q", range(count))
    depths = array("q", [0]) * count
    for source, target, _ in graph.scan_edges():
        join_sets(parents, depths, order[source], order[target])
    smallest = array("q", names)
    for index in range(count):
        root = find_root(parents, index)
        if names[index] < smallest[root]:
            smallest[root] = names[index]
    return {names[index]: smallest[find_root(parents, index)] for index in range(count)}


# endregion Components

# region PageRank


def outgoing_scale(graph: BaseGraph, order: dict[int, int], weight: str | None) -> array:
    """The reciprocal of every vertex's outgoing weight, zero where it has none, in one edge pass."""
    totals = array("d", [0.0]) * len(order)
    for source, _, held in scan_arcs(graph, weight):
        totals[order[source]] += held
    return array("d", [1.0 / total if total else 0.0 for total in totals])


def settle_ranks(ranks: array, pushed: array, damping: float, leaked: float) -> float:
    """Folds a sweep's pushed mass back into the ranks and returns how far they moved."""
    drift = 0.0
    for index, mass in enumerate(pushed):
        updated = damping * mass + leaked
        drift += abs(updated - ranks[index])
        ranks[index] = updated
    return drift


def pagerank(
    graph: BaseGraph,
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-6,
    weight: str | None = "weight",
) -> dict[int, float]:
    """The PageRank of every vertex, scattered along out-edges, one sequential edge pass per sweep.

    Holds three 8-byte slots and one index entry per vertex, and one page of edge rows; a weighted run
    also holds one page of attribute documents, which is where its weights come from. With `weight` unset
    every edge carries the same mass, which is what NetworkX calls `weight=None`.
    """
    order = relabel_dense(list(graph.scan_nodes()))
    count = len(order)
    if not count:
        return {}
    share = 1.0 / count
    ranks = array("d", [share]) * count
    scale = outgoing_scale(graph, order, weight)
    # Which vertices lead nowhere never changes, so the sum below walks only those.
    dangling_at = [position for position, spread in enumerate(scale) if not spread]
    for _ in range(iterations):
        pushed = array("d", [0.0]) * count
        for source, target, held in scan_arcs(graph, weight):
            position = order[source]
            pushed[order[target]] += ranks[position] * held * scale[position]
        dangling = sum(map(ranks.__getitem__, dangling_at))
        leaked = damping * dangling * share + (1.0 - damping) * share
        if settle_ranks(ranks, pushed, damping, leaked) < count * tolerance:
            break
    return {node: ranks[index] for node, index in order.items()}


# endregion PageRank

# region Peeling


def total_degrees(graph: BaseGraph, order: dict[int, int]) -> array:
    """How many edge ends every vertex holds, counted in one sequential pass, self-loops excluded.

    A self-loop joins a vertex to itself and so carries no vertex into a core; NetworkX refuses a graph
    that has one, and peeling reads it the same way.
    """
    degrees = array("q", [0]) * len(order)
    for source, target, _ in graph.scan_edges():
        if source == target:
            continue
        degrees[order[source]] += 1
        degrees[order[target]] += 1
    return degrees


def peel_below(degrees: array, alive: bytearray, cores: array, level: int) -> bytearray:
    """Marks every surviving vertex at or under `level` as peeled at that level, and reports which."""
    peeled = bytearray(len(alive))
    for index, degree in enumerate(degrees):
        if not alive[index] or degree > level:
            continue
        peeled[index] = 1
        alive[index] = 0
        cores[index] = level
    return peeled


def lower_across_peeled(graph: BaseGraph, order: dict[int, int], degrees: array, peeled: bytearray) -> None:
    """Lowers the degree of every vertex reached from one peeled this round, in one edge pass."""
    for source, target, _ in graph.scan_edges():
        if source == target:
            continue
        left, right = order[source], order[target]
        degrees[right] -= peeled[left]
        degrees[left] -= peeled[right]


def core_numbers(graph: BaseGraph) -> dict[int, int]:
    """The largest `k` whose k-core holds every vertex, by peeling the lowest degree first.

    Holds one degree, one core number and one liveness byte per vertex, and spends one sequential edge
    pass per peeling round — never the neighbourhood of a peeled vertex, which is what the store would
    have to seek for. Self-loops carry no vertex into a core and are skipped, as NetworkX has it.
    """
    order = relabel_dense(list(graph.scan_nodes()))
    count = len(order)
    degrees = total_degrees(graph, order)
    cores = array("q", [0]) * count
    alive = bytearray(b"\x01") * count
    level = 0
    remaining = count
    while remaining:
        level = max(level, min(compress(degrees, alive)))
        peeled = peel_below(degrees, alive, cores, level)
        remaining -= peeled.count(1)
        lower_across_peeled(graph, order, degrees, peeled)
    return {node: cores[index] for node, index in order.items()}


# endregion Peeling

# region Triangles


def compact_run(neighbours: array, start: int, stop: int) -> int:
    """Sorts one vertex's neighbours in place, dropping repeats, and reports where its run now ends."""
    unique = sorted(set(neighbours[start:stop]))
    neighbours[start : start + len(unique)] = array("q", unique)
    return start + len(unique)


def sorted_adjacency(graph: BaseGraph, order: dict[int, int]) -> tuple[array, array, array]:
    """Every vertex's neighbours as one sorted run, packed at 8 bytes an edge end, in two edge passes."""
    count = len(order)
    starts = array("q", [0]) * (count + 1)
    for source, target, _ in graph.scan_edges():
        starts[order[source] + 1] += source != target
        starts[order[target] + 1] += source != target
    for index in range(count):
        starts[index + 1] += starts[index]
    neighbours = array("q", [0]) * starts[count]
    cursors = array("q", starts[:count])
    for source, target, _ in graph.scan_edges():
        if source == target:
            continue
        left, right = order[source], order[target]
        neighbours[cursors[left]] = right
        neighbours[cursors[right]] = left
        cursors[left] += 1
        cursors[right] += 1
    ends = array("q", [compact_run(neighbours, starts[index], cursors[index]) for index in range(count)])
    return starts, ends, neighbours


def shared_count(left: Sequence[int], right: Sequence[int]) -> int:
    """How many vertices two sorted adjacency runs share, by walking each of them once."""
    left_index = right_index = shared = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            shared += 1
            left_index += 1
            right_index += 1
        elif left[left_index] < right[right_index]:
            left_index += 1
        else:
            right_index += 1
    return shared


def accumulate_shared(starts: array, ends: array, neighbours: array, counts: array) -> None:
    """Adds to both ends of every edge how many neighbours the two ends share, run against sorted run.

    Each run is sorted, so the neighbours below the vertex itself are skipped by a search rather than
    one at a time — every unordered pair is then visited exactly once.
    """
    runs = memoryview(neighbours)
    for index in range(len(counts)):
        run = runs[starts[index] : ends[index]]
        for other in run[bisect_left(run, index) :]:
            shared = shared_count(run, runs[starts[other] : ends[other]])
            counts[index] += shared
            counts[other] += shared


def triangle_counts(graph: BaseGraph, nodes: Iterable[int] | None = None) -> dict[int, int]:
    """How many triangles every vertex takes part in, by intersecting two sorted adjacency runs at a time.

    Holds one packed adjacency of 8 bytes per edge end plus three 8-byte slots per vertex, built in two
    sequential edge passes; a self-loop and a repeated edge are dropped as the runs are sorted.
    """
    order = relabel_dense(list(graph.scan_nodes()))
    starts, ends, neighbours = sorted_adjacency(graph, order)
    counts = array("q", [0]) * len(order)
    accumulate_shared(starts, ends, neighbours, counts)
    wanted = order if nodes is None else set(nodes)
    return {node: counts[order[node]] // 2 for node in wanted}


# endregion Triangles

# region Sampling


def degree_histogram(graph: BaseGraph, role: Role = Role.ANY) -> Counter[int]:
    """How many vertices hold each degree, streamed rather than sorted into a list."""
    return Counter(degree for _, degree in DegreeView(graph, role))


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


def sample_nodes(graph: BaseGraph, count: int, seed: int | None = None) -> list[int]:
    """A uniform sample of `count` vertices, over one pass of the vertex scan."""
    return reservoir(graph.scan_nodes(), count, seed)


def sample_edges(graph: BaseGraph, count: int, seed: int | None = None) -> list[Triple]:
    """A uniform sample of `count` edges, over one pass of the edge stream."""
    return reservoir(graph.scan_edges(), count, seed)


def relabel_dense(nodes: Sequence[int]) -> dict[int, int]:
    """A dense 0-based numbering of sparse vertex identifiers, for matrices and adjacency arrays."""
    return {node: index for index, node in enumerate(nodes)}


# endregion Sampling
