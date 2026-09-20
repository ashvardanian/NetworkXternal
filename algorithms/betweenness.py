"""Betweenness centrality by Brandes' accumulation, over sampled sources."""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from typing import Any

from algorithms.base import Bounds, VertexOriented
from algorithms.results import DenseIndex, VertexMap
from algorithms.streams import adjacency, reservoir
from networkxternal.base_api import BaseGraph


class BetweennessCentrality(VertexOriented[VertexMap[float]]):
    """How often each vertex lies on a shortest path, summed over the sources asked for.

    The layered walk records a distance and a path count per vertex, and the backward pass reads the
    adjacency again rather than holding predecessors, so a source costs two round trips per layer and
    no state proportional to the edges. The sources are independent of each other.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=32, passes="per layer per source")

    def __init__(
        self,
        graph: BaseGraph,
        *,
        sources: Iterable[int] | None = None,
        seed: int | None = None,
        samples: int | None = None,
    ) -> None:
        super().__init__(graph)
        self.sources = sources
        """Which vertices to walk from, or every vertex when unasked."""

        self.seed = seed
        """What makes a sampled run repeatable."""

        self.samples = samples
        """How many sources to keep, or all of them when unasked."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """Four sampled sources, which is what the declared bound is measured at."""
        return {"samples": 4, "seed": 42}

    def chosen(self) -> list[int]:
        """The sources this walk starts from, sampled in one pass where a sample was asked for."""
        held = list(self.order) if self.sources is None else list(self.sources)
        if self.samples is not None and self.samples < len(held):
            return reservoir(held, self.samples, self.seed)
        return held

    def shortest_path_counts(self, source: int) -> tuple[list[list[int]], array, array]:
        """The layers reached from `source`, how many shortest paths reach each vertex, and how far."""
        sigma = array("d", [0.0]) * self.count
        distance = array("q", [-1]) * self.count
        sigma[self.order[source]] = 1.0
        distance[self.order[source]] = 0
        layers = [[source]]
        depth = 0
        while layers[-1]:
            beyond: list[int] = []
            for node, other, _ in adjacency(self.graph, layers[-1], self.graph.outgoing_role):
                position = self.order[other]
                if distance[position] < 0:
                    distance[position] = depth + 1
                    beyond.append(other)
                if distance[position] == depth + 1:
                    sigma[position] += sigma[self.order[node]]
            layers.append(beyond)
            depth += 1
        layers.pop()
        return layers, sigma, distance

    def accumulate_dependencies(self, layers: list[list[int]], sigma: array, distance: array) -> None:
        """Folds each layer's dependency back into the one before it, deepest first, into the scores."""
        dependency = array("d", [0.0]) * self.count
        for layer in reversed(layers[1:]):
            for node, other, _ in adjacency(self.graph, layer, self.graph.outgoing_role):
                here, there = self.order[node], self.order[other]
                if distance[there] != distance[here] - 1 or not sigma[here]:
                    continue
                dependency[there] += sigma[there] / sigma[here] * (1.0 + dependency[here])
            for node in layer:
                self.scores[self.order[node]] += dependency[self.order[node]]

    def run(self) -> VertexMap[float]:
        self.order = DenseIndex(self.graph.scan_nodes())
        self.count = len(self.order)
        self.scores = array("d", [0.0]) * self.count
        for source in self.chosen():
            layers, sigma, distance = self.shortest_path_counts(source)
            self.accumulate_dependencies(layers, sigma, distance)
        halved = 1.0 if self.graph.is_directed() else 0.5
        return VertexMap(self.order, array("d", [score * halved for score in self.scores]))


def betweenness_centrality(
    graph: BaseGraph,
    sources: Iterable[int] | None = None,
    seed: int | None = None,
    samples: int | None = None,
) -> VertexMap[float]:
    """How often each vertex lies on a shortest path, over every source or a sample of them."""
    return BetweennessCentrality.on(graph, sources=sources, seed=seed, samples=samples).run()
