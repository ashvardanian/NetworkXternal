"""NetworkX-shaped views and writes over an external store, on the storage verbs every backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import StrEnum
from itertools import batched, islice
from threading import Lock
from typing import Any, ClassVar

type Attributes = dict[str, Any]
type Triple = tuple[int, int, int]
type NodeBunch = int | Iterable[int] | None
type Cursor = Triple | None

FIRST_EDGE_ID = 1
"""The edge identifier a graph starts handing out from."""


class NetworkXternalError(Exception):
    """Raised where NetworkX raises `NetworkXError`: a vertex or edge the graph does not hold."""


class Role(StrEnum):
    """The end of an edge a vertex is looked up by."""

    SOURCE = "source"
    TARGET = "target"
    ANY = "any"


class AttributeStore(StrEnum):
    """Which of the two attribute stores a document read or write addresses."""

    NODES = "nodes"
    EDGES = "edges"


class EdgeLayout(StrEnum):
    """How a backend stores an undirected edge, which decides what a reverse lookup costs."""

    NATIVE = "native"
    """The engine reaches a relationship from either end on its own: UStore, Neo4J and Memgraph."""

    CANONICAL = "canonical"
    """One row per edge, holding `source <= target` because the write normalized it: SQL and MongoDB."""

    MIRRORED = "mirrored"
    """Two rows sharing one edge identifier, one ordered by each end: ClickHouse, which has no secondary index."""


def numeric_weight(value: Any) -> float | None:
    """A weight as a float, or `None` where the attribute is missing, boolean, or not a number."""
    return float(value) if type(value) is int or type(value) is float else None


class NodeView:
    """The vertices of a graph: iterable, sized, and callable for their attributes, as `Graph.nodes` is."""

    def __init__(self, graph: BaseGraph) -> None:
        self.graph = graph

    def __iter__(self) -> Iterator[int]:
        return iter(self.graph)

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    def __contains__(self, node: int) -> bool:
        return self.graph.has_node(node)

    def __getitem__(self, node: int) -> Attributes:
        if not self.graph.has_node(node):
            raise KeyError(node)
        return self.graph.read_documents(AttributeStore.NODES, [node])[0]

    def __call__(self, data: bool | str = False, default: Any = None) -> Iterator[int] | Iterator[tuple[int, Any]]:
        if data is False:
            return iter(self.graph)
        return self._with_data(data, default)

    def _with_data(self, data: bool | str, default: Any) -> Iterator[tuple[int, Any]]:
        for page in batched(self.graph, self.graph.PAGE):
            attributes = self.graph.read_documents(AttributeStore.NODES, page)
            for node, found in zip(page, attributes, strict=True):
                yield (node, found) if data is True else (node, found.get(data, default))


class EdgeView:
    """The edges of a graph: iterable, sized, and callable for keys and attributes, as `Graph.edges` is."""

    def __init__(self, graph: BaseGraph) -> None:
        self.graph = graph

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """Every edge as its two ends, followed by its key in a multigraph."""
        return self(keys=self.graph.MULTIGRAPH)

    def __len__(self) -> int:
        return self.graph.number_of_edges()

    def __contains__(self, edge: tuple[int, int]) -> bool:
        return self.graph.has_edge(*edge)

    def __call__(
        self,
        nbunch: NodeBunch = None,
        data: bool | str = False,
        keys: bool = False,
        default: Any = None,
    ) -> Iterator[tuple[Any, ...]]:
        if keys and not self.graph.MULTIGRAPH:
            raise TypeError("Only a multigraph reports edge keys")
        for page in batched(self.graph.edge_triples(nbunch), self.graph.PAGE):
            identifiers = [edge for _, _, edge in page]
            attributes = (
                self.graph.read_documents(AttributeStore.EDGES, identifiers)
                if data is not False
                else [None] * len(page)
            )
            for (source, target, edge), found in zip(page, attributes, strict=True):
                head = (source, target, edge) if keys else (source, target)
                if data is False:
                    yield head
                elif data is True:
                    yield (*head, found)
                else:
                    yield (*head, found.get(data, default))


class DegreeView:
    """Degrees of vertices in one role: indexable by vertex, iterable as pairs, callable with weights."""

    def __init__(
        self,
        graph: BaseGraph,
        role: Role,
        weight: str | None = None,
        nodes: list[int] | None = None,
    ) -> None:
        self.graph = graph
        self.role = role
        self.weight = weight
        self.nodes = nodes

    def __getitem__(self, node: int) -> int | float:
        if not self.graph.has_node(node):
            raise KeyError(node)
        return next(self._degrees([node]))[1]

    def __iter__(self) -> Iterator[tuple[int, int | float]]:
        pages = batched(self.graph if self.nodes is None else self.nodes, self.graph.PAGE)
        for page in pages:
            yield from self._degrees(list(page))

    def __call__(self, nbunch: NodeBunch = None, weight: str | None = None) -> int | float | DegreeView:
        if not isinstance(nbunch, Iterable | None):
            return DegreeView(self.graph, self.role, weight)[nbunch]
        return DegreeView(self.graph, self.role, weight, None if nbunch is None else list(nbunch))

    def _degrees(self, nodes: list[int]) -> Iterator[tuple[int, int | float]]:
        if self.weight is None:
            counts = self.graph.degrees(nodes, self.role)
            yield from ((node, count or 0) for node, count in zip(nodes, counts, strict=True))
            return
        totals = dict.fromkeys(nodes, 0.0)
        for page in batched(self.graph.adjacent_edges(nodes, self.role), self.graph.PAGE):
            self._add_weights(totals, page)
        yield from ((node, totals[node]) for node in nodes)

    def _add_weights(self, totals: dict[int, float], page: Sequence[tuple[int, Triple]]) -> None:
        """Adds one page of incident edges to the running totals, reading their documents in one call."""
        attributes = self.graph.read_documents(AttributeStore.EDGES, [edge for _, (_, _, edge) in page])
        for (node, (source, target, _)), found in zip(page, attributes, strict=True):
            weight = found.get(self.weight, 1)
            # NetworkX counts an undirected self-loop at both of its ends.
            totals[node] += weight * 2 if source == target and self.role is Role.ANY else weight


class BaseGraph(ABC):
    """An undirected graph holding at most one edge between two vertices, as `networkx.Graph` does.

    A backend implements the storage verbs of the region below and inherits every view, traversal and
    attribute map from here. Every read reaches the store, so a graph holds nothing that could go stale.
    """

    DIRECTED: ClassVar[bool] = False
    MULTIGRAPH: ClassVar[bool] = False

    LAYOUT: ClassVar[EdgeLayout] = EdgeLayout.NATIVE
    """How this backend stores an undirected edge, which a reverse lookup and a removal both consult."""

    PAGE: ClassVar[int] = 1 << 12
    """How many keys one round-trip carries; a backend lowers it where the wire format is heavy."""

    CONCURRENT: ClassVar[bool] = True
    """Whether one instance serves several threads, which a free-threaded interpreter exploits."""

    __networkx_backend__: ClassVar[str] = "networkxternal"
    """The name NetworkX dispatches by, so `nx.pagerank(graph, backend="networkxternal")` finds this graph."""

    def __init__(self) -> None:
        self.next_edge_id: int | None = None
        self.graph: dict[str, Any] = {}
        self.edge_ids_lock = Lock()
        """Guards the identifier counter, which several threads sharing one instance would otherwise race."""

    @property
    def name(self) -> str:
        return self.graph.get("name", "")

    @name.setter
    def name(self, name: str) -> None:
        self.graph["name"] = name

    def __str__(self) -> str:
        kind = type(self).__name__
        return f"{self.name or kind} with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges"

    def is_directed(self) -> bool:
        return self.DIRECTED

    def is_multigraph(self) -> bool:
        return self.MULTIGRAPH

    # region Storage Verbs

    @abstractmethod
    def scan_nodes(self) -> Iterator[int]:
        """Yields every vertex of the graph, holding at most one page of them at a time."""

    @abstractmethod
    def has_node(self, node: int) -> bool:
        """Whether the graph holds that vertex."""

    @abstractmethod
    def number_of_nodes(self) -> int:
        """How many vertices the graph holds."""

    @abstractmethod
    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        """Inserts vertices, leaving the ones already stored as they are."""

    @abstractmethod
    def drop_nodes(self, nodes: Sequence[int]) -> None:
        """Removes vertices together with every edge they take part in, but not their attributes."""

    @abstractmethod
    def scan_edges(self, after: Cursor = None, limit: int | None = None) -> Iterator[Triple]:
        """Every edge of the graph in stored order, streamed, resuming after the edge `after` names.

        A backend holds one page of rows at a time, whatever the degree of any vertex in it.
        """

    @abstractmethod
    def adjacent_edges(self, nodes: Sequence[int], role: Role) -> Iterator[tuple[int, Triple]]:
        """Every edge incident to a given vertex in that role, streamed as `(vertex, edge)` pairs.

        An edge joining two of `nodes` is reported once per vertex; a self-loop is reported once.
        The stream is bounded by a page of rows, so a vertex of any degree is never materialized.
        """

    @abstractmethod
    def find_pairs(self, sources: Sequence[int], targets: Sequence[int]) -> Iterator[tuple[int, Triple]]:
        """Every edge stored between a given pair, streamed as `(position, edge)` in the pairs' own order.

        A pair holding no edge yields nothing; a pair given twice yields its edges twice.
        An undirected graph answers a pair in either orientation, a directed one only as asked.
        """

    @abstractmethod
    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        """How many edges every vertex takes part in, in that role."""

    @abstractmethod
    def upsert_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        """Inserts edges with their vertices, overwriting the ones stored under the same identifier."""

    @abstractmethod
    def drop_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        """Removes edges, skipping the ones that are not stored, and keeps their vertices."""

    def edge_weights(self, edges: Sequence[int], name: str = "weight") -> list[float | None]:
        """The numeric weight of every edge under `name`, `None` where it is missing or not a number.

        A store holding weights in a typed column or field overrides this read of whole documents.
        """
        return [numeric_weight(found.get(name)) for found in self.read_documents(AttributeStore.EDGES, edges)]

    def count_edges(self) -> int:
        """How many edges the graph holds; a store that can count them outright overrides this walk."""
        return sum(degree for _, degree in DegreeView(self, Role.SOURCE))

    def close(self) -> None:
        """Releases the connections or handles the backend holds; a store needing none overrides nothing."""
        return None

    def __enter__(self) -> BaseGraph:
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()

    @abstractmethod
    def biggest_edge_id(self) -> int:
        """The largest edge identifier the graph holds, or zero when it holds none."""

    @abstractmethod
    def read_documents(self, store: AttributeStore, keys: Sequence[int]) -> list[Attributes]:
        """The attribute document of every key, an empty one where there is none."""

    @abstractmethod
    def merge_documents(self, store: AttributeStore, keys: Sequence[int], entries: Sequence[Attributes]) -> None:
        """Merges one attribute document into every key, or one per key, as RFC 7386 does."""

    @abstractmethod
    def drop_documents(self, store: AttributeStore, keys: Sequence[int]) -> None:
        """Removes the attribute document of every key."""

    @abstractmethod
    def clear(self) -> None:
        """Removes every vertex, every edge, and the attributes of both."""

    # endregion Storage Verbs

    # region Vertices

    def __iter__(self) -> Iterator[int]:
        return self.scan_nodes()

    def __len__(self) -> int:
        return self.number_of_nodes()

    def __contains__(self, node: int) -> bool:
        return self.has_node(node)

    def __getitem__(self, node: int) -> dict[int, Any]:
        """The neighbours of a vertex with the attributes of the edges reaching them, keyed by edge in a multigraph."""
        if not self.has_node(node):
            raise KeyError(node)
        triples = list(self.edge_triples(node))
        attributes = self.read_documents(AttributeStore.EDGES, [edge for _, _, edge in triples])
        adjacency: dict[int, Any] = {}
        for (_, neighbour, edge), found in zip(triples, attributes, strict=True):
            if self.MULTIGRAPH:
                adjacency.setdefault(neighbour, {})[edge] = found
            else:
                adjacency[neighbour] = found
        return adjacency

    @property
    def nodes(self) -> NodeView:
        return NodeView(self)

    def order(self) -> int:
        return self.number_of_nodes()

    def add_node(self, node: int, **attributes: Any) -> None:
        self.add_nodes_from([node], **attributes)

    def add_nodes_from(self, nodes: Iterable[Any], **attributes: Any) -> None:
        """Inserts vertices, or `(vertex, attributes)` pairs, merging `attributes` into every one of them."""
        for page in batched(nodes, self.PAGE):
            keys = [entry[0] if isinstance(entry, tuple) else entry for entry in page]
            own = [dict(entry[1]) if isinstance(entry, tuple) else {} for entry in page]
            self.upsert_nodes(keys)
            if attributes or any(own):
                self.merge_documents(AttributeStore.NODES, keys, [{**attributes, **entry} for entry in own])

    def remove_node(self, node: int) -> None:
        if not self.has_node(node):
            raise NetworkXternalError(f"The graph holds no vertex {node}")
        self.remove_nodes_from([node])

    def remove_nodes_from(self, nodes: Iterable[int]) -> None:
        """Removes vertices with every edge they take part in and the attributes of both."""
        for page in batched(nodes, self.PAGE):
            for incident in batched(self.adjacent_edges(page, Role.ANY), self.PAGE):
                self.drop_documents(AttributeStore.EDGES, [edge for _, (_, _, edge) in incident])
            self.drop_nodes(page)
            self.drop_documents(AttributeStore.NODES, page)

    # endregion Vertices

    # region Edges

    @property
    def outgoing_role(self) -> Role:
        """The role a vertex plays in the edges it reaches its neighbours by."""
        return Role.SOURCE if self.DIRECTED else Role.ANY

    def canonical_pair(self, source: int, target: int) -> tuple[int, int]:
        """The pair both orientations of an undirected edge share."""
        return (source, target) if self.DIRECTED else (min(source, target), max(source, target))

    def edges_of_pairs(self, sources: Sequence[int], targets: Sequence[int]) -> list[list[Triple]]:
        """The edges between each pair, one list per pair, holding no more than the batch asked about."""
        grouped: list[list[Triple]] = [[] for _ in sources]
        for position, triple in self.find_pairs(sources, targets):
            grouped[position].append(triple)
        return grouped

    def edges_of_pair(self, source: int, target: int) -> list[Triple]:
        """The edges between one pair, which is bounded by the multiplicity of that pair alone."""
        return [triple for _, triple in self.find_pairs([source], [target])]

    def allocate_edge_ids(self, count: int) -> list[int]:
        """Hands out identifiers past every one the graph already holds, to one thread at a time.

        The lock covers the threads sharing this instance; two processes still need a durable claim.
        """
        with self.edge_ids_lock:
            if self.next_edge_id is None:
                self.next_edge_id = max(self.biggest_edge_id() + 1, FIRST_EDGE_ID)
            first = self.next_edge_id
            self.next_edge_id += count
        return list(range(first, first + count))

    def add_edges_from_arrays(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        keys: Sequence[int | None] | None = None,
        *,
        columns: Mapping[str, Sequence[Any]] | None = None,
        entries: Sequence[Attributes] | None = None,
        **attributes: Any,
    ) -> list[int]:
        """Inserts a batch of edges, merging `attributes` into all of them, and `columns` or `entries` edge by edge.

        Returns:
            The identifier of every edge, in the order the edges were given.
        """
        sources = [int(source) for source in sources]
        targets = [int(target) for target in targets]
        if len(sources) != len(targets):
            raise ValueError(f"Got {len(sources)} sources and {len(targets)} targets")
        wanted = [None] * len(sources) if keys is None else [None if key is None else int(key) for key in keys]
        if not self.MULTIGRAPH and any(key is not None for key in wanted):
            raise TypeError("Only a multigraph takes edge keys")

        identifiers: list[int] = []
        upserted: list[Triple] = []
        if self.MULTIGRAPH:
            fresh = iter(self.allocate_edge_ids(sum(key is None for key in wanted)))
            identifiers = [next(fresh) if key is None else key for key in wanted]
            # A key the caller chose is never handed out again.
            self.next_edge_id = max([self.next_edge_id or 0, *(key + 1 for key in wanted if key is not None)])
            upserted = list(zip(sources, targets, identifiers, strict=True))
        else:
            stored = self.edges_of_pairs(sources, targets)
            pending: dict[tuple[int, int], int] = {}
            for position, (source, target) in enumerate(zip(sources, targets, strict=True)):
                pair = self.canonical_pair(source, target)
                if stored[position]:
                    identifiers.append(stored[position][0][2])
                    continue
                if pair not in pending:
                    pending[pair] = self.allocate_edge_ids(1)[0]
                    upserted.append((source, target, pending[pair]))
                identifiers.append(pending[pair])

        if upserted:
            upsert_sources, upsert_targets, upsert_edges = zip(*upserted, strict=True)
            self.upsert_edges(upsert_sources, upsert_targets, upsert_edges)
        if attributes or columns or entries:
            per_edge = [
                {
                    **attributes,
                    **{name: values[index] for name, values in (columns or {}).items()},
                    **((entries or [{}] * len(identifiers))[index]),
                }
                for index in range(len(identifiers))
            ]
            shared = not columns and not entries
            self.merge_documents(AttributeStore.EDGES, identifiers, per_edge[:1] if shared else per_edge)
        return identifiers

    def add_edge(self, source: int, target: int, **attributes: Any) -> None:
        self.add_edges_from_arrays([source], [target], **attributes)

    def add_edges_from(self, ebunch: Iterable[tuple[Any, ...]], **attributes: Any) -> None:
        """Inserts (source, target) pairs, (source, target, attributes) triples, or keyed quadruples."""
        for page in batched(ebunch, self.PAGE):
            sources: list[int] = []
            targets: list[int] = []
            wanted: list[int | None] = []
            entries: list[Attributes] = []
            for edge in page:
                source, target, *rest = edge
                key = rest[0] if self.MULTIGRAPH and rest and not isinstance(rest[0], Mapping) else None
                sources.append(source)
                targets.append(target)
                wanted.append(key)
                entries.append(dict(rest[-1]) if rest and isinstance(rest[-1], Mapping) else {})
            self.add_edges_from_arrays(
                sources,
                targets,
                wanted if self.MULTIGRAPH else None,
                entries=entries if any(entries) else None,
                **attributes,
            )

    def add_weighted_edges_from(
        self, ebunch: Iterable[tuple[int, int, float]], weight: str = "weight", **attributes: Any
    ) -> None:
        """Inserts `(source, target, weight)` triples, storing the weight under the attribute `weight` names."""
        for page in batched(ebunch, self.PAGE):
            self.add_edges_from_arrays(
                [source for source, _, _ in page],
                [target for _, target, _ in page],
                columns={weight: [held for _, _, held in page]},
                **attributes,
            )

    def remove_edges_from_arrays(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        keys: Sequence[int | None] | None = None,
    ) -> int:
        """Removes a batch of edges, the newest one of a pair where a multigraph names no key.

        Returns:
            The number of edges removed; pairs holding no edge are skipped.
        """
        sources = [int(source) for source in sources]
        targets = [int(target) for target in targets]
        wanted = [None] * len(sources) if keys is None else [None if key is None else int(key) for key in keys]
        stored = self.edges_of_pairs(sources, targets)
        removed: list[Triple] = []
        taken: set[Triple] = set()
        for position, key in enumerate(wanted):
            remaining = [triple for triple in stored[position] if triple not in taken]
            if not self.MULTIGRAPH:
                removed.extend(remaining)
            elif key is not None:
                removed.extend(triple for triple in remaining if triple[2] == key)
            elif remaining:
                removed.append(max(remaining, key=lambda triple: triple[2]))
            taken.update(removed)
        if removed:
            removed_sources, removed_targets, removed_edges = zip(*removed, strict=True)
            self.drop_edges(removed_sources, removed_targets, removed_edges)
            self.drop_documents(AttributeStore.EDGES, removed_edges)
        return len(removed)

    def remove_edge(self, source: int, target: int) -> None:
        if self.remove_edges_from_arrays([source], [target]) == 0:
            raise NetworkXternalError(f"The graph holds no edge between {source} and {target}")

    def remove_edges_from(self, ebunch: Iterable[tuple[int, ...]]) -> None:
        for page in batched(ebunch, self.PAGE):
            sources = [edge[0] for edge in page]
            targets = [edge[1] for edge in page]
            wanted = [edge[2] if self.MULTIGRAPH and len(edge) > 2 else None for edge in page]
            self.remove_edges_from_arrays(sources, targets, wanted)

    def edge_triples(self, nbunch: NodeBunch) -> Iterator[Triple]:
        """Every edge once, as stored for the whole graph and oriented away from `nbunch` otherwise."""
        if nbunch is None:
            yield from self.scan_edges()
            return
        nodes = list(nbunch) if isinstance(nbunch, Iterable) else [nbunch]
        wanted = set(nodes)
        for node, (source, target, edge) in self.adjacent_edges(nodes, self.outgoing_role):
            other = target if source == node else source
            # An edge joining two vertices of the bunch is reported by the lower of them alone.
            if not self.DIRECTED and other in wanted and other < node:
                continue
            yield (source, target, edge) if source == node else (target, source, edge)

    @property
    def edges(self) -> EdgeView:
        return EdgeView(self)

    def has_edge(self, source: int, target: int, key: int | None = None) -> bool:
        return any(key is None or edge == key for _, _, edge in self.edges_of_pair(source, target))

    def number_of_edges(self, source: int | None = None, target: int | None = None) -> int:
        if source is None or target is None:
            return self.count_edges()
        return len(self.edges_of_pair(source, target))

    def size(self, weight: str | None = None) -> int | float:
        if weight is None:
            return self.number_of_edges()
        return sum(attributes.get(weight, 1) for _, _, attributes in self.edges(data=True))

    def neighbors(self, node: int) -> Iterator[int]:
        """The distinct vertices an edge leads to from `node`, in ascending order."""
        if not self.has_node(node):
            raise NetworkXternalError(f"The graph holds no vertex {node}")
        return iter(sorted({target for _, target, _ in self.edge_triples(node)}))

    def nbunch_iter(self, nbunch: NodeBunch = None) -> Iterator[int]:
        """The vertices of `nbunch` the graph actually holds, as NetworkX filters a bunch."""
        if nbunch is None:
            return self.scan_nodes()
        nodes = nbunch if isinstance(nbunch, Iterable) else [nbunch]
        return (node for node in nodes if self.has_node(node))

    @property
    def adj(self) -> dict[int, Any]:
        """A snapshot of the adjacency of every vertex; writing into it does not reach the store."""
        return {node: self[node] for node in self}

    def adjacency(self) -> Iterator[tuple[int, dict[int, Any]]]:
        """Yields every vertex with the attributes of the edges reaching its neighbours, page by page."""
        for page in batched(self, self.PAGE):
            for node in page:
                yield node, self[node]

    def nodes_with_selfloops(self) -> Iterator[int]:
        return (source for source, target, _ in self.edge_triples(None) if source == target)

    def selfloop_edges(self) -> Iterator[Triple]:
        return (triple for triple in self.edge_triples(None) if triple[0] == triple[1])

    def number_of_selfloops(self) -> int:
        return sum(1 for _ in self.nodes_with_selfloops())

    def neighbors_of_group(self, nodes: Iterable[int]) -> set[int]:
        """The distinct vertices an edge leads to from any of `nodes`, minus `nodes` themselves."""
        given = set(nodes)
        return {target for _, target, _ in self.edge_triples(given)} - given

    def get_edge_data(self, source: int, target: int, default: Any = None) -> Any:
        between = self.edges_of_pair(source, target)
        if not between:
            return default
        return self.read_documents(AttributeStore.EDGES, [between[0][2]])[0]

    @property
    def degree(self) -> DegreeView:
        return DegreeView(self, Role.ANY)

    def clear_edges(self) -> None:
        """Removes every edge and its attributes, keeping the vertices."""
        while page := list(islice(self.scan_edges(), self.PAGE)):
            sources, targets, identifiers = zip(*page, strict=True)
            self.drop_edges(sources, targets, identifiers)
            self.drop_documents(AttributeStore.EDGES, identifiers)
        self.next_edge_id = None

    # endregion Edges

    # region Attribute Maps

    def get_node_attributes(self, name: str) -> dict[int, Any]:
        return {node: found[name] for node, found in self.nodes(data=True) if name in found}

    def set_node_attributes(self, values: Any, name: str | None = None) -> None:
        """Merges `values` into vertices: one value for all of them, a value per vertex, or a document per vertex."""
        if name is None:
            nodes = list(values)
            self.merge_documents(AttributeStore.NODES, nodes, [dict(values[node]) for node in nodes])
        elif isinstance(values, Mapping):
            nodes = list(values)
            self.merge_documents(AttributeStore.NODES, nodes, [{name: values[node]} for node in nodes])
        else:
            for page in batched(self, self.PAGE):
                self.merge_documents(AttributeStore.NODES, page, [{name: values}])

    def edge_label(self, source: int, target: int, edge: int) -> tuple[int, ...]:
        """How NetworkX names an edge: with its key in a multigraph, by its ends otherwise."""
        return (source, target, edge) if self.MULTIGRAPH else (source, target)

    def edge_identifier(self, label: tuple[int, ...]) -> int:
        """The identifier an edge label addresses."""
        if self.MULTIGRAPH:
            return label[2]
        between = self.edges_of_pair(label[0], label[1])
        if not between:
            raise KeyError(label)
        return between[0][2]

    def get_edge_attributes(self, name: str) -> dict[tuple[int, ...], Any]:
        labelled: dict[tuple[int, ...], Any] = {}
        for page in batched(self.edge_triples(None), self.PAGE):
            attributes = self.read_documents(AttributeStore.EDGES, [edge for _, _, edge in page])
            for (source, target, edge), found in zip(page, attributes, strict=True):
                if name in found:
                    labelled[self.edge_label(source, target, edge)] = found[name]
        return labelled

    def set_edge_attributes(self, values: Any, name: str | None = None) -> None:
        """Merges `values` into edges: one value for all of them, a value per edge, or a document per edge."""
        if name is None:
            labels = list(values)
            entries = [dict(values[label]) for label in labels]
        elif isinstance(values, Mapping):
            labels = list(values)
            entries = [{name: values[label]} for label in labels]
        else:
            for page in batched(self.edge_triples(None), self.PAGE):
                self.merge_documents(AttributeStore.EDGES, [edge for _, _, edge in page], [{name: values}])
            return
        identifiers = [self.edge_identifier(label) for label in labels]
        self.merge_documents(AttributeStore.EDGES, identifiers, entries)

    # endregion Attribute Maps


class BaseDiGraph(BaseGraph):
    """A directed graph holding at most one edge from one vertex to another, as `networkx.DiGraph` does."""

    DIRECTED = True

    def successors(self, node: int) -> Iterator[int]:
        return self.neighbors(node)

    def has_successor(self, source: int, target: int) -> bool:
        return self.has_edge(source, target)

    def has_predecessor(self, source: int, target: int) -> bool:
        return self.has_edge(target, source)

    @property
    def succ(self) -> dict[int, Any]:
        """A snapshot of every vertex's successors; writing into it does not reach the store."""
        return self.adj

    @property
    def pred(self) -> dict[int, Any]:
        """A snapshot of every vertex's predecessors; writing into it does not reach the store."""
        return {node: {source: {} for source in self.predecessors(node)} for node in self}

    def predecessors(self, node: int) -> Iterator[int]:
        return iter(sorted({source for _, (source, _, _) in self.adjacent_edges([node], Role.TARGET)}))

    @property
    def in_degree(self) -> DegreeView:
        return DegreeView(self, Role.TARGET)

    @property
    def out_degree(self) -> DegreeView:
        return DegreeView(self, Role.SOURCE)


class BaseMultiGraph(BaseGraph):
    """An undirected graph holding any number of keyed edges between two vertices, as `networkx.MultiGraph` does."""

    MULTIGRAPH = True

    def add_edge(self, source: int, target: int, key: int | None = None, **attributes: Any) -> int:
        """Inserts one edge, or merges `attributes` into the one `key` names, and answers its key."""
        return self.add_edges_from_arrays([source], [target], [key], **attributes)[0]

    def remove_edge(self, source: int, target: int, key: int | None = None) -> None:
        """Removes the edge `key` names, or the newest edge between the pair."""
        if self.remove_edges_from_arrays([source], [target], [key]) == 0:
            raise NetworkXternalError(f"The graph holds no edge {key} between {source} and {target}")

    def get_edge_data(self, source: int, target: int, key: int | None = None, default: Any = None) -> Any:
        between = self.edges_of_pair(source, target)
        identifiers = [edge for _, _, edge in between if key is None or edge == key]
        if not identifiers:
            return default
        attributes = self.read_documents(AttributeStore.EDGES, identifiers)
        return attributes[0] if key is not None else dict(zip(identifiers, attributes, strict=True))


class BaseMultiDiGraph(BaseMultiGraph, BaseDiGraph):
    """A directed graph holding any number of keyed edges between two vertices, as `networkx.MultiDiGraph` does."""

    DIRECTED = True
    MULTIGRAPH = True
