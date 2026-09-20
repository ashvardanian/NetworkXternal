"""Vertices reached one or two steps from a node, by a breadth-first walk capped at depth two."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from algorithms.base import Bounds, VertexOriented
from algorithms.breadth_first import BreadthFirstLayers
from networkxternal.base_api import BaseGraph


class Reach(StrEnum):
    """Which vertices a two-step lookup reports."""

    SECOND_STEP = "second-step"
    """Only the vertices exactly two steps away."""

    BOTH_STEPS = "both-steps"
    """The vertices one or two steps away."""


class NeighborsOfNeighbors(VertexOriented[set[int]]):
    """The vertices two steps from `node`, and the ones a single step reaches when `reach` asks for them."""

    BOUNDS = Bounds(retained_per_vertex=0, working_per_vertex=32, passes="per layer")

    def __init__(self, graph: BaseGraph, *, node: int, reach: Reach = Reach.SECOND_STEP) -> None:
        super().__init__(graph)
        self.node = node
        """The vertex the walk starts from."""

        self.reach = reach
        """Which of the reached layers the answer holds."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """One node, which is what the declared bound assumes."""
        return {"node": next(iter(graph.scan_nodes()), 0)}

    def run(self) -> set[int]:
        layers = list(BreadthFirstLayers(self.graph, sources=self.node, cutoff=2).layers())
        wanted = layers[1:3] if self.reach is Reach.BOTH_STEPS else layers[2:3]
        return set().union(*wanted) - {self.node} if wanted else set()


def neighbors_of_neighbors(graph: BaseGraph, node: int, reach: Reach = Reach.SECOND_STEP) -> set[int]:
    """The vertices two steps from `node`, and the ones a single step reaches when `reach` asks for them."""
    return NeighborsOfNeighbors.on(graph, node=node, reach=reach).run()
