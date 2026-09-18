"""MongoDB backend: edges as documents indexed by both ends, attributes in two collections beside them.

Vertices live in a collection of their own, so an isolated vertex survives the removal of its last
edge, and both attribute stores are plain document collections that `$set` merges field by field.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import batched
from urllib.parse import urlparse

import pymongo
from pymongo import MongoClient, UpdateOne

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


def database_name(url: str, default: str = "graph") -> str:
    """The database a connection string addresses, or `default` when it names none."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[0].lower() if parts else default


class MongoGraph(BaseGraph):
    """An undirected simple graph stored in MongoDB, as `networkx.Graph` is in RAM."""

    PAGE = 10_000
    """Batch writes beyond this size bring no further throughput and cost a lot of memory on the server."""

    def __init__(self, url: str = "mongodb://localhost:27017/graph") -> None:
        super().__init__()
        self.client = MongoClient(url)
        database = self.client[database_name(url)]
        self.nodes_collection = database["nodes"]
        self.edges_collection = database["edges"]
        self.stores = {
            AttributeStore.NODES: database["nodes_attributes"],
            AttributeStore.EDGES: database["edges_attributes"],
        }
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Compound indexes, so an adjacency read is answered from the index without touching a document."""
        self.edges_collection.create_index([("source", 1), ("target", 1)])
        self.edges_collection.create_index([("target", 1), ("source", 1)])

    @staticmethod
    def _ends(source: int, target: int, role: Role) -> tuple[int, ...]:
        """Which ends of a found edge the lookup was asking about."""
        if role is Role.SOURCE:
            return (source,)
        if role is Role.TARGET:
            return (target,)
        return (source,) if source == target else (source, target)

    def _role_filter(self, keys: Sequence[int], role: Role) -> dict:
        if role is Role.SOURCE:
            return {"source": {"$in": list(keys)}}
        if role is Role.TARGET:
            return {"target": {"$in": list(keys)}}
        return {"$or": [{"source": {"$in": list(keys)}}, {"target": {"$in": list(keys)}}]}

    # region Storage Verbs

    def scan_nodes(self) -> Iterator[int]:
        after: int | None = None
        while True:
            wanted = {} if after is None else {"_id": {"$gt": after}}
            page = list(self.nodes_collection.find(wanted, {"_id": 1}).sort("_id", pymongo.ASCENDING).limit(self.PAGE))
            for document in page:
                yield document["_id"]
            if len(page) < self.PAGE:
                return
            after = page[-1]["_id"]

    def has_node(self, node: int) -> bool:
        return self.nodes_collection.find_one({"_id": node}, {"_id": 1}) is not None

    def number_of_nodes(self) -> int:
        return self.nodes_collection.count_documents({})

    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        if not len(nodes):
            return
        writes = [UpdateOne({"_id": node}, {"$setOnInsert": {"_id": node}}, upsert=True) for node in nodes]
        self.nodes_collection.bulk_write(writes, ordered=False)

    def drop_nodes(self, nodes: Sequence[int]) -> None:
        if not len(nodes):
            return
        self.edges_collection.delete_many(self._role_filter(nodes, Role.ANY))
        self.nodes_collection.delete_many({"_id": {"$in": list(nodes)}})

    def find_edges(self, nodes: Sequence[int], role: Role) -> list[list[Triple]]:
        keys = list(nodes)
        if not keys:
            return []
        grouped: dict[int, list[Triple]] = {key: [] for key in keys}
        found = self.edges_collection.find(self._role_filter(keys, role), {"source": 1, "target": 1})
        for document in found:
            source, target = document["source"], document["target"]
            for end in self._ends(source, target, role):
                if end in grouped:
                    grouped[end].append((source, target, document["_id"]))
        return [grouped[key] for key in keys]

    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        keys = list(nodes)
        if not keys:
            return []
        counts = dict.fromkeys(keys, 0)
        fields = ("source", "target") if role is Role.ANY else (role.value,)
        for field in fields:
            pipeline = [
                {"$match": {field: {"$in": keys}}},
                {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
            ]
            for entry in self.edges_collection.aggregate(pipeline, allowDiskUse=True):
                if entry["_id"] in counts:
                    counts[entry["_id"]] += entry["count"]
        return [counts[key] for key in keys]

    def upsert_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        if not len(sources):
            return
        triples = list(zip(sources, targets, edges, strict=True))
        writes = [
            UpdateOne({"_id": edge}, {"$set": {"source": source, "target": target}}, upsert=True)
            for source, target, edge in triples
        ]
        self.edges_collection.bulk_write(writes, ordered=False)
        self.upsert_nodes(list({end for source, target, _ in triples for end in (source, target)}))

    def drop_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        if len(edges):
            self.edges_collection.delete_many({"_id": {"$in": list(edges)}})

    def count_edges(self) -> int:
        return self.edges_collection.count_documents({})

    def biggest_edge_id(self) -> int:
        found = self.edges_collection.find({}, {"_id": 1}).sort("_id", pymongo.DESCENDING).limit(1)
        return next((document["_id"] for document in found), 0)

    def read_documents(self, store: AttributeStore, keys: Sequence[int]) -> list[Attributes]:
        keys = list(keys)
        if not keys:
            return []
        found = {document.pop("_id"): document for document in self.stores[store].find({"_id": {"$in": keys}})}
        return [found.get(key, {}) for key in keys]

    def merge_documents(self, store: AttributeStore, keys: Sequence[int], entries: Sequence[Attributes]) -> None:
        keys = list(keys)
        if not keys:
            return
        for entry in entries:
            for name in entry:
                if name.startswith("$") or "." in name:
                    raise ValueError(f"MongoDB reads {name!r} as a path or an operator, not as a field name")
        shared = entries[0] if len(entries) == 1 else None
        writes = [
            UpdateOne({"_id": key}, {"$set": shared if shared is not None else entries[index]}, upsert=True)
            for index, key in enumerate(keys)
        ]
        for page in batched(writes, self.PAGE):
            self.stores[store].bulk_write(list(page), ordered=False)

    def drop_documents(self, store: AttributeStore, keys: Sequence[int]) -> None:
        if len(keys):
            self.stores[store].delete_many({"_id": {"$in": list(keys)}})

    def clear(self) -> None:
        self.edges_collection.drop()
        self.nodes_collection.drop()
        for collection in self.stores.values():
            collection.drop()
        self._create_indexes()
        self.next_edge_id = None

    def close(self) -> None:
        self.client.close()

    # endregion Storage Verbs


class MongoDiGraph(MongoGraph, BaseDiGraph):
    """A directed simple graph stored in MongoDB, as `networkx.DiGraph` is in RAM."""


class MongoMultiGraph(MongoGraph, BaseMultiGraph):
    """An undirected multigraph stored in MongoDB, as `networkx.MultiGraph` is in RAM."""


class MongoMultiDiGraph(MongoGraph, BaseMultiDiGraph):
    """A directed multigraph stored in MongoDB, as `networkx.MultiDiGraph` is in RAM."""
