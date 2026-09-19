"""UStore backend: vertices and edges in one graph collection, attributes in two document collections.

UStore is the only store here that speaks graphs natively, so a lookup of many vertices costs one
call rather than one query per vertex, and attribute documents are merged in place by the engine.

Two choices decide what the numbers mean. Persistence is made here: a directory opens a database on
disk, and `IN_MEMORY` opens one that never touches it. The storage engine behind either is chosen
when the extension is linked, one per build, and `engines()` answers which one this build carries —
rebuild with `CMAKE_ARGS="-DUSTORE_USE_ROCKSDB=ON -DUSTORE_PYTHON_ENGINE=rocksdb" pip install .`
for a log-structured engine instead of the in-memory default.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from itertools import batched
from pathlib import Path
from typing import Any

from ustore.keys import KEY_MAX, KEY_MIN, scan_keys
from ustore.lib import Database, Transaction

from networkxternal.base_api import (
    FIRST_EDGE_ID,
    Attributes,
    AttributeStore,
    BaseDiGraph,
    BaseGraph,
    BaseMultiDiGraph,
    BaseMultiGraph,
    Cursor,
    NetworkXternalError,
    Role,
    Triple,
)

type View = Database | Transaction

IN_MEMORY = ":memory:"
"""The directory that opens a database holding everything in RAM, as SQLite spells the same choice."""


def engines() -> tuple[str, ...]:
    """The storage engines the installed extension was linked with, such as `ram` or `rocksdb`."""
    from ustore.lib import engines_compiled

    return tuple(engines_compiled())


class UStoreGraph(BaseGraph):
    """An undirected simple graph stored in UStore, as `networkx.Graph` is in RAM."""

    SHARES_STORE = False
    """An embedded engine keeps its own state per handle, so two objects on one directory are two graphs."""

    RETRIES = 8
    """How many times a conflicted claim is retried before it gives up, since the transaction is optimistic."""

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

        Either a directory `url` to open a `Database` in, `IN_MEMORY` for one that never reaches disk,
        or an already open `view` to borrow.
        """
        super().__init__()
        if (url is None) == (view is None):
            raise ValueError("Pass either a directory to open, `IN_MEMORY`, or an open view to borrow")
        if view is None and url != IN_MEMORY:
            # The engine opens a directory rather than creating one, so an empty graph starts here.
            Path(url).mkdir(parents=True, exist_ok=True)
        self.view = view if view is not None else Database(None if url == IN_MEMORY else url)
        self.graph_collection = self._collection(graph)
        self.nodes_collection = self._collection(nodes)
        self.edges_collection = self._collection(edges)
        self.meta_collection = self._collection(f"{graph}_meta")
        """One document holding the shape this graph was written as, and its identifier counter."""

        try:
            self._seed_meta()
        except Exception:
            self.close()
            raise

    def _collection(self, name: str) -> int:
        database = self.view if isinstance(self.view, Database) else self.view.database
        if database.collection_contains(name):
            return database.collection_find(name)
        return database.collection_create(name)

    META = 0
    """The key the meta document is stored under, in a collection holding nothing else."""

    def claim_edge_ids(self, count: int) -> int:
        """Claims a run inside a transaction, which is the one embedded store here that has them."""
        for _ in range(self.RETRIES):
            try:
                with self.view.transaction() if isinstance(self.view, Database) else nullcontext(self.view) as writer:
                    held = self._read_meta(writer)
                    held["next_edge"] += count
                    self._write_meta(writer, held)
                return held["next_edge"] - count
            except Exception as failure:
                if "conflict" not in str(failure).lower():
                    raise
        raise NetworkXternalError(f"Could not claim {count} identifiers within {self.RETRIES} attempts")

    def raise_edge_floor(self, floor: int) -> None:
        held = self._read_meta(self.view)
        if held["next_edge"] < floor:
            held["next_edge"] = floor
            self._write_meta(self.view, held)

    def _seed_meta(self) -> None:
        """Writes the meta document on first use, and refuses a graph written as another shape."""
        held = self._read_meta(self.view, seed=False)
        if held is None:
            first = max(self.biggest_edge_id() + 1, FIRST_EDGE_ID)
            shape = {"directed": self.DIRECTED, "multigraph": self.MULTIGRAPH, "next_edge": first}
            self._write_meta(self.view, shape)
            return
        if (held["directed"], held["multigraph"]) != (self.DIRECTED, self.MULTIGRAPH):
            raise NetworkXternalError(
                f"This graph was written as directed={held['directed']}, multigraph={held['multigraph']}, "
                f"and is being opened as directed={self.DIRECTED}, multigraph={self.MULTIGRAPH}"
            )

    def _read_meta(self, view: View, seed: bool = True) -> Attributes | None:
        """The meta document, or `None` where the graph has never been written to."""
        found = view.docs_read([self.META], collection=self.meta_collection)[0]
        if found is None:
            return (
                {"directed": self.DIRECTED, "multigraph": self.MULTIGRAPH, "next_edge": FIRST_EDGE_ID} if seed else None
            )
        return json.loads(bytes(found))

    def _write_meta(self, view: View, held: Attributes) -> None:
        view.docs_write([self.META], json.dumps(held), collection=self.meta_collection)

    def _store(self, store: AttributeStore) -> int:
        return self.nodes_collection if store is AttributeStore.NODES else self.edges_collection

    # region Storage Verbs

    def scan_nodes(self) -> Iterator[int]:
        return scan_keys(self.view, self.graph_collection, self.PAGE)

    def has_node(self, node: int) -> bool:
        return self.view.contains(node, collection=self.graph_collection)

    def number_of_nodes(self) -> int:
        """The engine's own count where it is exact, and a scan where the engine can only bound it."""
        held = self.view.measure(KEY_MIN, KEY_MAX, collection=self.graph_collection)
        if held["min_cardinality"] == held["max_cardinality"]:
            return held["min_cardinality"]
        return sum(1 for _ in self.scan_nodes())

    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        """Inserts the vertices the graph does not hold, asking which those are in one call.

        A batch whose every vertex already exists is refused by the C ABI, and a presence check per key
        was one round trip per key on the import path; one batched scan answers the whole page.
        """
        keys = list(dict.fromkeys(int(node) for node in nodes))
        if not keys:
            return
        found = self.view.scan_batch(keys, [1] * len(keys), collection=self.graph_collection)
        held = {run[0] for index in range(len(keys)) if (run := found[index]) is not None and len(run)}
        missing = [key for key in keys if key not in held]
        if missing:
            self.view.graph_upsert_vertices(missing, collection=self.graph_collection)

    def drop_nodes(self, nodes: Sequence[int]) -> None:
        if len(nodes):
            self.view.graph_remove_vertices(list(nodes), collection=self.graph_collection)

    def scan_edges(self, after: Cursor = None, limit: int | None = None) -> Iterator[Triple]:
        """Walks the vertices in key order and reports the edges leaving each, which is how the engine stores them."""
        taken = 0
        start = None if after is None else after[0]
        for page in batched(scan_keys(self.view, self.graph_collection, self.PAGE), self.PAGE):
            keys = [key for key in page if start is None or key >= start]
            if not keys:
                continue
            found = self.view.graph_find_edges(keys, role=Role.SOURCE.value, collection=self.graph_collection)
            for index, key in enumerate(keys):
                for triple in self._triples(found, index):
                    if after is not None and (key, triple[1], triple[2]) <= after:
                        continue
                    yield triple
                    taken += 1
                    if limit is not None and taken >= limit:
                        return

    def adjacent_edges(self, nodes: Sequence[int], role: Role) -> Iterator[tuple[int, Triple]]:
        """One call per page of vertices, which is the engine's own unit — a vertex's edges arrive together.

        A lookup in either role finds a self-loop at both of its ends, and the verb reports it once.
        """
        keys = list(dict.fromkeys(int(node) for node in nodes))
        for page in batched(keys, self.PAGE):
            found = self.view.graph_find_edges(list(page), role=role.value, collection=self.graph_collection)
            for index, key in enumerate(page):
                yield from ((key, triple) for triple in dict.fromkeys(self._triples(found, index)))

    def find_pairs(self, sources: Sequence[int], targets: Sequence[int]) -> Iterator[tuple[int, Triple]]:
        """One lookup of the distinct sources, filtered to the pairs asked about."""
        positions = self.group_pairs(sources, targets)
        if not positions:
            return
        wanted = list({source for source, _ in positions})
        reported: set[tuple[int, Triple]] = set()
        for _, triple in self.adjacent_edges(wanted, self.outgoing_role):
            for found in self.fan_out(positions, triple):
                if found not in reported:
                    reported.add(found)
                    yield found

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

    def clear_storage(self) -> None:
        """Empties every collection in one call each, rather than walking and popping key by key."""
        database = self.view if isinstance(self.view, Database) else self.view.database
        for collection in (self.graph_collection, self.nodes_collection, self.edges_collection):
            database.collection_drop(collection, mode="pairs")
        self._write_meta(
            self.view, {"directed": self.DIRECTED, "multigraph": self.MULTIGRAPH, "next_edge": FIRST_EDGE_ID}
        )

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
