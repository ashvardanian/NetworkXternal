"""What a walk numbers its vertices with, and what it answers with once it is done."""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections.abc import Iterable, Iterator, Mapping


class DenseIndex:
    """A dense 0-based numbering of sparse vertex identifiers, at eight bytes per vertex.

    Every backend scans its vertices in key order, so the identifiers are already sorted and the
    position of one is a binary search. A dict would answer in constant time and cost about a hundred
    bytes per vertex, which is four times the state every algorithm here holds for its own results.
    """

    __slots__ = ("nodes",)

    def __init__(self, nodes: Iterable[int]) -> None:
        self.nodes = array("q", nodes)
        """Every vertex identifier in key order, which is the order the store answered in."""

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[int]:
        return iter(self.nodes)

    def __getitem__(self, node: int) -> int:
        position = bisect_left(self.nodes, node)
        if position == len(self.nodes) or self.nodes[position] != node:
            raise KeyError(node)
        return position

    def __contains__(self, node: object) -> bool:
        if not isinstance(node, int):
            return False
        position = bisect_left(self.nodes, node)
        return position < len(self.nodes) and self.nodes[position] == node

    def items(self) -> Iterator[tuple[int, int]]:
        """Every vertex with its position, in key order."""
        return ((node, index) for index, node in enumerate(self.nodes))


class VertexMap[Value](Mapping[int, Value]):
    """An answer per vertex as two arrays: the `DenseIndex` the walk already held, and its values.

    Sixteen bytes per vertex where a dict costs seventy, and the index is shared with the walk that
    produced it, so an answer outliving its algorithm costs eight. A lookup is a binary search rather
    than a hash, which is the price.

    A walk still running hands out a view of its live array, so `freeze` is what a caller holding an
    intermediate answer across further sweeps wants.
    """

    __slots__ = ("held", "order")

    def __init__(self, order: DenseIndex, held: array) -> None:
        self.order = order
        """The vertex identifiers, in the order the values are laid out in."""

        self.held = held
        """One value per vertex, positionally matched to the index; `values` is `Mapping`'s own verb."""

    def __getitem__(self, node: int) -> Value:
        return self.held[self.order[node]]

    def __iter__(self) -> Iterator[int]:
        return iter(self.order.nodes)

    def __len__(self) -> int:
        return len(self.order)

    def __contains__(self, node: object) -> bool:
        return node in self.order

    def __eq__(self, other: object) -> bool:
        """Equal to any mapping holding the same pairs, which is how a test compares this to NetworkX."""
        if not isinstance(other, Mapping):
            return NotImplemented
        return len(self) == len(other) and all(node in other and other[node] == value for node, value in self.items())

    __hash__ = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} vertices)"

    def array(self) -> array:
        """The values in index order, uncopied, for NumPy and for writing out."""
        return self.held

    def freeze(self) -> VertexMap[Value]:
        """A copy no further sweep will move, sharing the index the walk already holds."""
        return VertexMap(self.order, array(self.held.typecode, self.held))

    def to_dict(self) -> dict[int, Value]:
        """The plain dict, at about seventy bytes a vertex, when a caller asks for one by name."""
        return dict(zip(self.order.nodes, self.held, strict=True))


def gathered[Value](order: DenseIndex, held: array, nodes: Iterable[int] | None) -> VertexMap[Value]:
    """A map over every vertex, or over the subset asked for, which needs an index of its own."""
    if nodes is None:
        return VertexMap(order, held)
    wanted = DenseIndex(sorted(set(nodes)))
    return VertexMap(wanted, array(held.typecode, (held[order[node]] for node in wanted.nodes)))


def sparsely[Value](settled: Mapping[int, Value], typecode: str) -> VertexMap[Value]:
    """A map over the vertices a walk actually reached, which is smaller than the vertex set.

    One sort and one transient dict at peak, for an answer four times smaller than the dict it came
    from — which is the trade for a walk whose reach is a fraction of the graph.
    """
    order = DenseIndex(sorted(settled))
    return VertexMap(order, array(typecode, map(settled.__getitem__, order.nodes)))
