"""UStore backend: vertices and edges in one graph collection, attributes in two document collections.

UStore is the only store here that speaks graphs natively, so a lookup of many vertices costs one
call rather than one query per vertex, and attribute documents are merged in place by the engine.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import Any

from ustore.keys import scan_keys
from ustore.lib import Database, Transaction

from networkxternal.base_api import (
    Attributes,
    AttributeStore,
    BaseDiGraph,
    BaseGraph,
    BaseMultiDiGraph,
    BaseMultiGraph,
    Role,
    Triple,
)

type View = Database | Transaction


class UStoreGraph(BaseGraph):
    """An undirected simple graph stored in UStore, as `networkx.Graph` is in RAM."""

    PAGE = 1 << 16
    """UStore takes whole batches in one call, so a page is as large as the engine's own."""

    def __init__(
        self,
        url: str | None = None,
        *,
        view: View | None = None,
        graph: str = "graph",
        nodes: str = "graph_nodes",
        edges: str = "graph_edges",
    ) -> None:
        """Opens a graph in the collection `graph`, with vertex and edge attributes beside it.

        Either a directory `url` to open a `Database` in, or an already open `view` to borrow.
        """
        super().__init__()
        if (url is None) == (view is None):
            raise ValueError("Pass either a directory to open or an open view to borrow")
        self.view = view if view is not None else Database(url)
        self.graph_collection = self._collection(graph)
        self.nodes_collection = self._collection(nodes)
        self.edges_collection = self._collection(edges)

    def _collection(self, name: str) -> int:
        database = self.view if isinstance(self.view, Database) else self.view.database
        if database.collection_contains(name):
            return database.collection_find(name)
        return database.collection_create(name)

    def _store(self, store: AttributeStore) -> int:
        return self.nodes_collection if store is AttributeStore.NODES else self.edges_collection

    # region Storage Verbs

    def scan_nodes(self) -> Iterator[int]:
        return scan_keys(self.view, self.graph_collection, self.PAGE)

    def has_node(self, node: int) -> bool:
        return self.view.contains(node, collection=self.graph_collection)

    def number_of_nodes(self) -> int:
        return sum(1 for _ in self.scan_nodes())

    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        # A batch whose every vertex already exists is refused by the C ABI, so only the absent ones travel.
        keys = list(nodes)
        missing = [key for key in keys if not self.view.contains(key, collection=self.graph_collection)]
        if missing:
            self.view.graph_upsert_vertices(missing, collection=self.graph_collection)

    def drop_nodes(self, nodes: Sequence[int]) -> None:
        if len(nodes):
            self.view.graph_remove_vertices(list(nodes), collection=self.graph_collection)

    def find_edges(self, nodes: Sequence[int], role: Role) -> list[list[Triple]]:
        keys = list(nodes)
        if not keys:
            return []
        found = self.view.graph_find_edges(keys, role=role.value, collection=self.graph_collection)
        return [self._triples(found, index) for index in range(len(keys))]

    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        keys = list(nodes)
        if not keys:
            return []
        counts = self.view.graph_degrees(keys, role=role.value, collection=self.graph_collection)
        return [count or 0 for count in counts]

    def upsert_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        if len(sources):
            self.view.graph_upsert_edges(
                list(sources), list(targets), edges=list(edges), collection=self.graph_collection
            )

    def drop_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        if len(sources):
            self.view.graph_remove_edges(
                list(sources), list(targets), edges=list(edges), collection=self.graph_collection
            )

    def biggest_edge_id(self) -> int:
        return max((edge for _, _, edge in self.edge_triples(None)), default=0)

    def read_documents(self, store: AttributeStore, keys: Sequence[int]) -> list[Attributes]:
        keys = list(keys)
        if not keys:
            return []
        found = self.view.docs_read(keys, collection=self._store(store))
        return [{} if document is None else json.loads(bytes(document)) for document in found]

    def merge_documents(self, store: AttributeStore, keys: Sequence[int], entries: Sequence[Attributes]) -> None:
        keys = list(keys)
        if not keys:
            return
        encoded: Any = json.dumps(entries[0]) if len(entries) == 1 else [json.dumps(entry) for entry in entries]
        self.view.docs_write(keys, encoded, modification="merge", collection=self._store(store))

    def drop_documents(self, store: AttributeStore, keys: Sequence[int]) -> None:
        keys = list(keys)
        if keys:
            self.view.pop(keys, collection=self._store(store))

    def close(self) -> None:
        """A borrowed view belongs to its owner; a database this graph opened is closed with it."""
        if isinstance(self.view, Database):
            self.view.close()

    def clear(self) -> None:
        while page := list(islice(self.scan_nodes(), self.PAGE)):
            self.view.graph_remove_vertices(page, collection=self.graph_collection)
        for collection in (self.nodes_collection, self.edges_collection):
            while page := list(islice(scan_keys(self.view, collection, self.PAGE), self.PAGE)):
                self.view.pop(page, collection=collection)
        self.next_edge_id = None

    # endregion Storage Verbs

    # region Helpers

    @staticmethod
    def _triples(found: Any, index: int) -> list[Triple]:
        """The (source, target, edge) triples one vertex of a lookup found."""
        entry = found[index]
        return [] if entry is None else [(source, target, edge) for source, target, edge in entry.tolist()]

    # endregion Helpers


class UStoreDiGraph(UStoreGraph, BaseDiGraph):
    """A directed simple graph stored in UStore, as `networkx.DiGraph` is in RAM."""


class UStoreMultiGraph(UStoreGraph, BaseMultiGraph):
    """An undirected multigraph stored in UStore, as `networkx.MultiGraph` is in RAM."""


class UStoreMultiDiGraph(UStoreGraph, BaseMultiDiGraph):
    """A directed multigraph stored in UStore, as `networkx.MultiDiGraph` is in RAM."""
