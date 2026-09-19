"""Neo4J backend: vertices and relationships carry their attributes as properties, so nothing is repacked.

Community Neo4J serves one database per instance, so a graph is named by its labels rather than by a
database: vertices carry `<name>` and relationships `<name>_EDGE`, which keeps disjoint graphs apart
in one server.

A label cannot be a parameter, so every statement is a template rendered once when the graph opens.
The templates interpolate `%vertex`, `%edge` and `%meta`, leaving `$parameter` to Cypher and `{map}`
to its maps — which is why they are `Template`s rather than f-strings, whose braces would all double.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from string import Template
from urllib.parse import urlparse

from neo4j import GraphDatabase

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
    per_key,
)

BATCH = 1_000
"""Larger batches exhaust the server's Java heap long before they pay off in throughput."""


class CypherText(Template):
    """Cypher text with the labels left to fill in, taking `%name` so `$name` stays Cypher's own."""

    delimiter = "%"


PATTERNS = {
    Role.SOURCE: "(v:%vertex {id: key})-[e:%edge]->(other:%vertex)",
    Role.TARGET: "(other:%vertex)-[e:%edge]->(v:%vertex {id: key})",
    Role.ANY: "(v:%vertex {id: key})-[e:%edge]-(other:%vertex)",
}
"""The match that binds `v` to the asked vertex in each role, before the labels are filled in."""

LOOPS = "COUNT { MATCH (v:%vertex {id: key})-[e:%edge]-(v) }"
"""The second end an undirected self-loop contributes, which the pattern above finds once."""


@dataclass(frozen=True)
class Queries:
    """Every statement this graph runs, rendered once from the labels it was opened with."""

    scan_nodes: str
    """Every vertex of the graph in key order."""

    has_node: str
    """Whether the graph holds one vertex."""

    count_nodes: str
    """How many vertices the graph holds."""

    upsert_nodes: str
    """Inserts a batch of vertices, leaving the ones already stored alone."""

    drop_nodes: str
    """Removes a batch of vertices with every relationship they take part in."""

    scan_edges: str
    """One page of relationships in identifier order, resumed after a cursor."""

    scan_edges_first: str
    """The first page of relationships, which has no cursor to resume from."""

    adjacent: dict[Role, str]
    """Every relationship incident to a batch of vertices, per role."""

    degrees: dict[Role, str]
    """How many relationship ends a batch of vertices holds, per role."""

    find_pairs: str
    """Every relationship between a batch of asked pairs."""

    upsert_edges: str
    """Inserts a batch of relationships with the vertices they join."""

    drop_edges: str
    """Removes a batch of relationships by identifier."""

    count_edges: str
    """How many relationships the graph holds."""

    biggest_edge: str
    """The largest relationship identifier the graph holds."""

    read_nodes: str
    """The properties of a batch of vertices."""

    read_edges: str
    """The properties of a batch of relationships."""

    merge_nodes: str
    """Merges a document into each of a batch of vertices."""

    merge_edges: str
    """Merges a document into each of a batch of relationships."""

    strip_nodes: str
    """Leaves a batch of vertices with their identifier alone."""

    strip_edges: str
    """Leaves a batch of relationships with their identifier alone."""

    clear: str
    """Removes every vertex of this graph, in batches the server commits separately."""

    rewind: str
    """Returns the identifier counter to its first value."""

    seed_meta: str
    """Writes the meta node on first use and answers the shape it holds."""

    claim: str
    """Claims a run of identifiers from the meta node."""

    raise_floor: str
    """Lifts the counter to a floor a caller-chosen key requires."""


def graph_name(url: str, default: str = "Graph") -> str:
    """The label a connection string names; a backtick would end the quoting the label is written in."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    name = parts[0] if parts else default
    if "`" in name:
        raise ValueError(f"A graph name may not hold a backtick, got {name!r}")
    return name[0].upper() + name[1:]


class Neo4JGraph(BaseGraph):
    """An undirected simple graph stored in Neo4J, as `networkx.Graph` is in RAM."""

    PAGE = BATCH

    def __init__(self, url: str = "bolt://localhost:7687/graph") -> None:
        super().__init__()
        address = urlparse(url)
        self.driver = GraphDatabase.driver(
            f"{address.scheme}://{address.hostname}:{address.port}",
            auth=(address.username, address.password) if address.username else None,
        )
        self.vertex = f"`{graph_name(url)}`"
        self.meta = f"`{graph_name(url)}_META`"
        """The label of the one node holding this graph's shape and its identifier counter."""

        self.edge = f"`{graph_name(url).upper()}_EDGE`"
        self.queries = self._build_queries()
        """Every statement this graph runs, rendered from its labels when it opened."""

        try:
            self._create_indexes()
            self._seed_meta()
        except Exception:
            # A refused graph leaves no object to close, so its driver is released here.
            self.driver.close()
            raise

    def _build_queries(self) -> Queries:
        """Renders every statement from this graph's labels, once, so no call site interpolates."""
        labels = {"vertex": self.vertex, "edge": self.edge, "meta": self.meta}
        fill = lambda body: CypherText(body).substitute(labels)  # noqa: E731
        patterns = {role: fill(body) for role, body in PATTERNS.items()}
        loops = fill(LOOPS)
        return Queries(
            scan_nodes=fill("MATCH (v:%vertex) RETURN v.id AS id ORDER BY id"),
            has_node=fill("MATCH (v:%vertex {id: $id}) RETURN count(v) > 0 AS found"),
            count_nodes=fill("MATCH (v:%vertex) RETURN count(v) AS count"),
            upsert_nodes=fill("UNWIND $keys AS key MERGE (:%vertex {id: key})"),
            drop_nodes=fill("UNWIND $keys AS key MATCH (v:%vertex {id: key}) DETACH DELETE v"),
            scan_edges=fill(
                "MATCH (source:%vertex)-[e:%edge]->(target:%vertex) WHERE e.id > $after "
                "RETURN source.id AS source, target.id AS target, e.id AS edge ORDER BY e.id LIMIT $width"
            ),
            scan_edges_first=fill(
                "MATCH (source:%vertex)-[e:%edge]->(target:%vertex) "
                "RETURN source.id AS source, target.id AS target, e.id AS edge ORDER BY e.id LIMIT $width"
            ),
            adjacent={
                role: (
                    "UNWIND $keys AS key MATCH " + pattern + " "
                    "RETURN key AS key, startNode(e).id AS source, endNode(e).id AS target, e.id AS edge"
                )
                for role, pattern in patterns.items()
            },
            degrees={
                role: (
                    "UNWIND $keys AS key RETURN key AS key, COUNT { MATCH "
                    + pattern
                    + " }"
                    + (" + " + loops if role is Role.ANY else "")
                    + " AS degree"
                )
                for role, pattern in patterns.items()
            },
            find_pairs=fill(
                "UNWIND range(0, size($pairs) - 1) AS position "
                "MATCH (a:%vertex {id: $pairs[position][0]})-[e:%edge]"
                + ("->" if self.DIRECTED else "-")
                + "(b:%vertex) WHERE b.id = $pairs[position][1] "
                "RETURN position AS position, startNode(e).id AS source, endNode(e).id AS target, e.id AS edge"
            ),
            upsert_edges=fill(
                "UNWIND $rows AS row MERGE (source:%vertex {id: row.source}) "
                "MERGE (target:%vertex {id: row.target}) "
                "MERGE (source)-[:%edge {id: row.edge}]->(target)"
            ),
            drop_edges=fill("UNWIND $keys AS key MATCH ()-[e:%edge {id: key}]-() DELETE e"),
            count_edges=fill("MATCH ()-[e:%edge]->() RETURN count(e) AS count"),
            biggest_edge=fill("MATCH ()-[e:%edge]->() RETURN max(e.id) AS biggest"),
            read_nodes=fill("UNWIND $keys AS key MATCH (v:%vertex {id: key}) RETURN key AS key, properties(v) AS held"),
            read_edges=fill(
                "UNWIND $keys AS key MATCH ()-[e:%edge {id: key}]-() RETURN key AS key, properties(e) AS held"
            ),
            merge_nodes=fill("UNWIND $rows AS row MATCH (v:%vertex {id: row.key}) SET v += row.held"),
            merge_edges=fill("UNWIND $rows AS row MATCH ()-[e:%edge {id: row.key}]-() SET e += row.held"),
            strip_nodes=fill("UNWIND $keys AS key MATCH (v:%vertex {id: key}) SET v = {id: key}"),
            strip_edges=fill("UNWIND $keys AS key MATCH ()-[e:%edge {id: key}]-() SET e = {id: key}"),
            clear=fill("MATCH (v:%vertex) CALL (v) { DETACH DELETE v } IN TRANSACTIONS OF " + str(BATCH) + " ROWS"),
            rewind=fill("MATCH (m:%meta) SET m.next_edge = $first"),
            seed_meta=fill(
                "MERGE (m:%meta {graph: 1}) "
                "ON CREATE SET m.next_edge = $first, m.directed = $directed, m.multigraph = $multigraph "
                "RETURN m.directed AS directed, m.multigraph AS multigraph"
            ),
            claim=fill("MATCH (m:%meta) SET m.next_edge = m.next_edge + $count RETURN m.next_edge - $count AS first"),
            raise_floor=fill(
                "MATCH (m:%meta) SET m.next_edge = CASE WHEN m.next_edge < $floor THEN $floor ELSE m.next_edge END"
            ),
        )

    def _create_indexes(self) -> None:
        with self.driver.session() as session:
            name = self.vertex.strip("`")
            session.run(f"CREATE INDEX `{name}_id` IF NOT EXISTS FOR (v:{self.vertex}) ON (v.id)")
            session.run(f"CREATE INDEX `{name}_edge` IF NOT EXISTS FOR ()-[e:{self.edge}]-() ON (e.id)")

    def claim_edge_ids(self, count: int) -> int:
        """Claims a run inside a write transaction, which both servers serialize on the meta node."""
        self._seed_meta()
        query = self.queries.claim
        with self.driver.session() as session:
            return session.execute_write(lambda transaction: transaction.run(query, count=count).single()["first"])

    def raise_edge_floor(self, floor: int) -> None:
        self._seed_meta()
        query = self.queries.raise_floor
        with self.driver.session() as session:
            session.execute_write(lambda transaction: transaction.run(query, floor=floor).consume())

    def _seed_meta(self) -> None:
        """Writes the meta node on first use, and refuses a graph written as another shape."""
        first = max(self.biggest_edge_id() + 1, FIRST_EDGE_ID)
        query = self.queries.seed_meta
        with self.driver.session() as session:
            held = session.execute_write(
                lambda transaction: transaction.run(
                    query, first=first, directed=self.DIRECTED, multigraph=self.MULTIGRAPH
                ).single()
            )
        if (held["directed"], held["multigraph"]) != (self.DIRECTED, self.MULTIGRAPH):
            raise NetworkXternalError(
                f"This graph was written as directed={held['directed']}, multigraph={held['multigraph']}, "
                f"and is being opened as directed={self.DIRECTED}, multigraph={self.MULTIGRAPH}"
            )

    def scan_nodes(self) -> Iterator[int]:
        with self.driver.session() as session:
            result = session.run(self.queries.scan_nodes)
            for record in result:
                yield record["id"]

    def has_node(self, node: int) -> bool:
        with self.driver.session() as session:
            found = session.run(self.queries.has_node, id=node).single()
        return bool(found["found"])

    def number_of_nodes(self) -> int:
        with self.driver.session() as session:
            return session.run(self.queries.count_nodes).single()["count"]

    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        if not len(nodes):
            return
        with self.driver.session() as session:
            session.run(self.queries.upsert_nodes, keys=list(nodes))

    def drop_nodes(self, nodes: Sequence[int]) -> None:
        if not len(nodes):
            return
        with self.driver.session() as session:
            session.run(self.queries.drop_nodes, keys=list(nodes))

    def scan_edges(self, after: Cursor = None, limit: int | None = None) -> Iterator[Triple]:
        """Walks the relationships in identifier order, one page per query, resumed by the last identifier."""
        remaining = limit
        cursor = after
        while remaining is None or remaining > 0:
            width = self.PAGE if remaining is None else min(self.PAGE, remaining)
            query = self.queries.scan_edges if cursor is not None else self.queries.scan_edges_first
            with self.driver.session() as session:
                page = [
                    (record["source"], record["target"], record["edge"])
                    for record in session.run(query, after=None if cursor is None else cursor[2], width=width)
                ]
            yield from page
            if len(page) < width:
                return
            cursor = page[-1]
            if remaining is not None:
                remaining -= len(page)

    def adjacent_edges(self, nodes: Sequence[int], role: Role) -> Iterator[tuple[int, Triple]]:
        keys = list(dict.fromkeys(int(node) for node in nodes))
        query = self.queries.adjacent[role]
        for page in batched(keys, self.PAGE):
            with self.driver.session() as session:
                found = [
                    (record["key"], (record["source"], record["target"], record["edge"]))
                    for record in session.run(query, keys=list(page))
                ]
            # An undirected match reaches a self-loop from both of its ends.
            yield from dict.fromkeys(found)

    def find_pairs(self, sources: Sequence[int], targets: Sequence[int]) -> Iterator[tuple[int, Triple]]:
        pairs = [[int(source), int(target)] for source, target in zip(sources, targets, strict=True)]
        if not pairs:
            return
        query = self.queries.find_pairs
        for page in batched(enumerate(pairs), self.PAGE):
            with self.driver.session() as session:
                found = [
                    (record["position"], (record["source"], record["target"], record["edge"]))
                    for record in session.run(query, pairs=[pair for _, pair in page])
                ]
            offset = page[0][0]
            # A self-loop matches in both directions, so the same edge is reported once per position.
            for position, triple in dict.fromkeys(found):
                yield offset + position, triple

    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        keys = list(nodes)
        if not keys:
            return []
        query = self.queries.degrees[role]
        with self.driver.session() as session:
            counts = {record["key"]: record["degree"] for record in session.run(query, keys=keys)}
        return [counts.get(key, 0) for key in keys]

    def upsert_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        rows = [
            {"source": int(source), "target": int(target), "edge": int(edge)}
            for source, target, edge in zip(sources, targets, edges, strict=True)
        ]
        if not rows:
            return
        query = self.queries.upsert_edges
        with self.driver.session() as session:
            session.run(query, rows=rows)

    def drop_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        keys = list(edges)
        if not keys:
            return
        query = self.queries.drop_edges
        with self.driver.session() as session:
            session.run(query, keys=keys)

    def count_edges(self) -> int:
        with self.driver.session() as session:
            return session.run(self.queries.count_edges).single()["count"]

    def biggest_edge_id(self) -> int:
        with self.driver.session() as session:
            found = session.run(self.queries.biggest_edge).single()
        return found["biggest"] or 0

    def read_documents(self, store: AttributeStore, keys: Sequence[int]) -> list[Attributes]:
        """Attributes are properties of the vertex or relationship itself, minus the `id` we key by."""
        keys = list(keys)
        if not keys:
            return []
        query = self.queries.read_nodes if store is AttributeStore.NODES else self.queries.read_edges
        with self.driver.session() as session:
            found = {record["key"]: dict(record["held"]) for record in session.run(query, keys=keys)}
        return [{name: value for name, value in found.get(key, {}).items() if name != "id"} for key in keys]

    def merge_documents(self, store: AttributeStore, keys: Sequence[int], entries: Sequence[Attributes]) -> None:
        keys = list(keys)
        if not keys:
            return
        rows = [{"key": int(key), "held": entry} for key, entry in zip(keys, per_key(keys, entries), strict=True)]
        query = self.queries.merge_nodes if store is AttributeStore.NODES else self.queries.merge_edges
        with self.driver.session() as session:
            session.run(query, rows=rows)

    def drop_documents(self, store: AttributeStore, keys: Sequence[int]) -> None:
        """Strips every property but the identifier, from a vertex or from a relationship that outlives it."""
        keys = list(keys)
        if not keys:
            return
        query = self.queries.strip_nodes if store is AttributeStore.NODES else self.queries.strip_edges
        with self.driver.session() as session:
            session.run(query, keys=keys)

    def clear_storage(self) -> None:
        with self.driver.session() as session:
            session.run(self.queries.clear)
            session.run(self.queries.rewind, first=FIRST_EDGE_ID)

    def close(self) -> None:
        self.driver.close()

    # endregion Storage Verbs


class Neo4JDiGraph(Neo4JGraph, BaseDiGraph):
    """A directed simple graph stored in Neo4J, as `networkx.DiGraph` is in RAM."""


class Neo4JMultiGraph(Neo4JGraph, BaseMultiGraph):
    """An undirected multigraph stored in Neo4J, as `networkx.MultiGraph` is in RAM."""


class Neo4JMultiDiGraph(Neo4JGraph, BaseMultiDiGraph):
    """A directed multigraph stored in Neo4J, as `networkx.MultiDiGraph` is in RAM."""
