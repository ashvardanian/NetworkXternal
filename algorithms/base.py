"""What every algorithm is: a walk bound to one graph, declaring what it holds and how it reads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Any, ClassVar, Self

from algorithms.streams import Arc, scan_arcs, weighted_adjacency
from networkxternal.base_api import BaseGraph, Orientation, Role


@dataclass(frozen=True)
class Bounds:
    """What one variant holds and how often it reads the graph, as a contract `test/bounds.py` enforces.

    The bytes are of vertex state in the Python heap, measured on a store whose own buffers
    `tracemalloc` cannot see, and they are what the growth between two graph sizes is checked against.
    """

    retained_per_vertex: int
    """What the answer itself costs per vertex once the call returns."""

    working_per_vertex: int
    """What the walk holds per vertex while it runs, in arrays, and gives back when it returns."""

    passes: str
    """How often the edge set is read: `one`, `two`, `per sweep`, `per layer`, `per round`."""

    retained_per_edge: int = 0
    """What the answer costs per edge, for the walks whose state a vertex count cannot describe."""


REGISTRY: dict[str, type[Algorithm]] = {}
"""Every concrete algorithm, filled as its class is defined, which the bounds suite parametrizes over."""


class Algorithm[Result](ABC):
    """One algorithm bound to one graph and one set of parameters, driven by `run`.

    An instance is a single walk and keeps its arrays afterwards, which is what makes a partially
    driven walk inspectable without a second call.
    """

    BOUNDS: ClassVar[Bounds]
    """What this variant costs; a concrete class that omits it does not import."""

    ORIENTATION: ClassVar[Orientation]
    """Which way this variant drives the store."""

    VARIANTS: ClassVar[dict[Orientation, type[Algorithm]]] = {}
    """The sibling classes `on` chooses between; empty where only one orientation makes sense."""

    NEEDS_DIRECTION: ClassVar[bool] = False
    """Whether the question only makes sense on a directed graph, as a topological order does."""

    def __init__(self, graph: BaseGraph) -> None:
        self.graph = graph

    def __init_subclass__(cls, abstract: bool = False, **rest: Any) -> None:
        super().__init_subclass__(**rest)
        if abstract:
            return
        if "BOUNDS" not in vars(cls):
            raise TypeError(f"{cls.__name__} lands without a Bounds declaration")
        REGISTRY[cls.__name__] = cls

    @classmethod
    def on(cls, graph: BaseGraph, /, orientation: Orientation | None = None, **parameters: Any) -> Algorithm[Result]:
        """The variant this store prefers, or the one asked for, built with the parameters given."""
        return cls.VARIANTS.get(orientation or graph.SCAN_ORIENTATION, cls)(graph, **parameters)

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """The arguments a declared bound assumes, which the bounds suite and the variant tests reuse."""
        return {}

    @classmethod
    def probe(cls, graph: BaseGraph) -> Self:
        """The instance the bounds suite drives, built with the arguments its bound is declared for."""
        return cls(graph, **cls.probing(graph))

    @abstractmethod
    def run(self) -> Result:
        """Drives the walk to completion and answers."""


class EdgeOriented[Result](Algorithm[Result], abstract=True):
    """Scatters: `scan_edges` in stored order, one page in flight, never a seek per vertex."""

    ORIENTATION = Orientation.EDGE

    def arcs(self, weight: str | None = None) -> Iterator[Arc]:
        """Every edge as an arc, mirrored unless the graph is directed."""
        return scan_arcs(self.graph, weight)


class VertexOriented[Result](Algorithm[Result], abstract=True):
    """Gathers: `adjacent_edges` a page of vertices at a time, which is a graph engine's own unit."""

    ORIENTATION = Orientation.VERTEX

    def pages(self, nodes: Iterable[int] | None = None) -> Iterator[Sequence[int]]:
        """The vertices in pages the store answers one round trip for, the whole graph when unasked.

        An `array` is sliced rather than batched, so a page of a `DenseIndex` stays eight bytes a
        vertex instead of boxing every identifier into a Python integer.
        """
        if nodes is None:
            return batched(self.graph.scan_nodes(), self.graph.PAGE)
        if isinstance(nodes, array):
            return (nodes[start : start + self.graph.PAGE] for start in range(0, len(nodes), self.graph.PAGE))
        return batched(nodes, self.graph.PAGE)

    def arcs_of(self, nodes: Sequence[int], role: Role, weight: str | None = None) -> Iterator[Arc]:
        """`(vertex, neighbour, weight)` for every edge incident to the page, in that role."""
        return weighted_adjacency(self.graph, nodes, role, weight)


class Sweeping[Result](Algorithm[Result], abstract=True):
    """A walk repeating one pass until the answer stops moving, with every sweep observable."""

    def __init__(self, graph: BaseGraph, *, iterations: int = 100, tolerance: float = 1e-6) -> None:
        super().__init__(graph)
        self.iterations = iterations
        """How many sweeps a graph that never settles is allowed."""

        self.tolerance = tolerance
        """How little the state must move, per vertex, for the walk to call itself settled."""

        self.count = 0
        """How many vertices the graph holds, which `start` fills in."""

    @abstractmethod
    def start(self) -> None:
        """Allocates the per-vertex arrays and whatever one setup pass produces, and sets `count`."""

    @abstractmethod
    def sweep(self) -> float:
        """One pass over the graph; answers how far the state moved."""

    @abstractmethod
    def result(self) -> Result:
        """A view of the state as it stands, live rather than copied; `VertexMap.freeze` copies."""

    def settled(self, drift: float) -> bool:
        """Whether that much movement counts as done; a walk over labels rather than mass overrides it."""
        return drift < self.tolerance * self.count

    def sweeps(self) -> Iterator[Result]:
        """Yields the answer after each sweep, stopping once the state has settled."""
        self.start()
        if not self.count:
            return
        for _ in range(self.iterations):
            drift = self.sweep()
            yield self.result()
            if self.settled(drift):
                return

    def run(self) -> Result:
        for _ in self.sweeps():
            pass
        return self.result()
