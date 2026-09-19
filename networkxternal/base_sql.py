"""SQL backend: four tables — vertices, edges indexed by both ends, and one attribute table per store.

Every statement goes through SQLAlchemy Core rather than the ORM, since mapping rows into objects
costs more than the query itself on graph-shaped workloads. Attribute documents are stored as JSON
text and merged read-modify-write inside one transaction, which every dialect here supports.

An undirected edge is one row holding `source <= target`, so a pair lookup is a single equality probe
while adjacency still reads both indexes. The one attribute the algorithms read, `weight`, sits in a
typed column that both indexes carry, and everything else stays in the document beside it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from itertools import batched
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    Index,
    MetaData,
    Table,
    Text,
    bindparam,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    literal,
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlalchemy_utils import create_database, database_exists

from networkxternal.base_api import (
    Attributes,
    AttributeStore,
    BaseDiGraph,
    BaseGraph,
    BaseMultiDiGraph,
    BaseMultiGraph,
    Cursor,
    EdgeLayout,
    NetworkXternalError,
    Role,
    Triple,
    numeric_weight,
)

PARAMETER_CAPS = {"mysql": 65_535, "mariadb": 65_535, "postgresql": 65_535, "sqlite": 32_766}
"""How many parameters one statement may bind, per dialect, which caps how long an `IN` list may grow."""

WEIGHT = "weight"
"""The one edge attribute that lives in a typed column rather than in the attribute document."""

metadata = MetaData()

nodes_table = Table("nodes", metadata, Column("node", BigInteger, primary_key=True))

edges_table = Table(
    "edges",
    metadata,
    Column("edge", BigInteger, primary_key=True),
    Column("source", BigInteger, nullable=False),
    Column("target", BigInteger, nullable=False),
    Column("weight", Float, nullable=True),
)

node_attributes_table = Table(
    "node_attributes",
    metadata,
    Column("node", BigInteger, primary_key=True),
    Column("document", Text, nullable=False),
)

edge_attributes_table = Table(
    "edge_attributes",
    metadata,
    Column("edge", BigInteger, primary_key=True),
    Column("document", Text, nullable=False),
)

Index("edges_by_source", edges_table.c.source, edges_table.c.target, edges_table.c.edge, edges_table.c.weight)
Index("edges_by_target", edges_table.c.target, edges_table.c.source, edges_table.c.edge, edges_table.c.weight)


def without_numeric_weight(document: Attributes) -> Attributes:
    """The document minus a weight the typed column now holds, or the document unchanged."""
    if numeric_weight(document.get(WEIGHT)) is None:
        return document
    return {name: value for name, value in document.items() if name != WEIGHT}


class SQLGraph(BaseGraph):
    """An undirected simple graph stored in a SQL database, as `networkx.Graph` is in RAM."""

    LAYOUT = EdgeLayout.CANONICAL
    """One row holds an undirected edge with `source <= target`, so a pair is one equality probe."""

    PAGE = 10_000
    """One statement carries this many rows; dialects cap the number of bound parameters well above it."""

    PAIRS = 1_024
    """How many pairs one row-constructor `IN` carries, which a planner's stack limits long before the parameter cap."""

    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        super().__init__()
        if not database_exists(url):
            create_database(url)
        shared = url.endswith(":memory:")
        self.engine: Engine = create_engine(
            url,
            poolclass=StaticPool if shared else None,
            connect_args={"check_same_thread": False} if shared else {},
        )
        metadata.create_all(self.engine)
        self.check_schema()
        self.tune()

    def check_schema(self) -> None:
        """Refuses a database whose tables predate this schema, since `create_all` never alters one.

        The project ships no migration, so an older database is dropped and written again.
        """
        held = {column["name"] for column in inspect(self.engine).get_columns(edges_table.name)}
        missing = {column.name for column in edges_table.columns} - held
        if missing:
            raise NetworkXternalError(
                f"This database predates the current schema: `edges` has no {', '.join(sorted(missing))}. "
                f"Drop its tables and write it again; no migration ships."
                ""
            )

    def tune(self) -> None:
        """Applies the dialect's performance settings; the base dialect needs none."""

    def _attributes_table(self, store: AttributeStore) -> Table:
        return node_attributes_table if store is AttributeStore.NODES else edge_attributes_table

    def _key_column(self, store: AttributeStore) -> Column:
        return self._attributes_table(store).c.node if store is AttributeStore.NODES else edge_attributes_table.c.edge

    def _role_column(self, role: Role) -> Column:
        return edges_table.c.source if role is Role.SOURCE else edges_table.c.target

    @staticmethod
    def _ends(source: int, target: int, role: Role) -> tuple[int, ...]:
        """Which ends of a found edge the lookup was asking about."""
        if role is Role.SOURCE:
            return (source,)
        if role is Role.TARGET:
            return (target,)
        return (source,) if source == target else (source, target)

    def _role_clause(self, keys: Sequence[int], role: Role):
        if role is Role.SOURCE:
            return edges_table.c.source.in_(keys)
        if role is Role.TARGET:
            return edges_table.c.target.in_(keys)
        return or_(edges_table.c.source.in_(keys), edges_table.c.target.in_(keys))

    def insert_ignore(self, table: Table):
        """An insert that leaves a row already holding the primary key untouched, in this dialect's spelling."""
        match self.engine.dialect.name:
            case "sqlite":
                return sqlite_insert(table).on_conflict_do_nothing()
            case "postgresql":
                return postgres_insert(table).on_conflict_do_nothing()
            case "mysql" | "mariadb":
                return mysql_insert(table).prefix_with("IGNORE")
            case other:
                raise NotImplementedError(f"No conflict-free insert is spelled out for {other}")

    def upsert_ends(self, table: Table):
        """An insert that rewrites only the ends of a row already stored, leaving its weight as it is."""
        match self.engine.dialect.name:
            case "sqlite":
                statement = sqlite_insert(table)
                ends = {"source": statement.excluded.source, "target": statement.excluded.target}
                return statement.on_conflict_do_update(index_elements=[table.c.edge], set_=ends)
            case "postgresql":
                statement = postgres_insert(table)
                ends = {"source": statement.excluded.source, "target": statement.excluded.target}
                return statement.on_conflict_do_update(index_elements=[table.c.edge], set_=ends)
            case "mysql" | "mariadb":
                statement = mysql_insert(table)
                return statement.on_duplicate_key_update(
                    source=statement.inserted.source, target=statement.inserted.target
                )
            case other:
                raise NotImplementedError(f"No end-only upsert is spelled out for {other}")

    def key_pages(self, keys: Sequence[Any], parameters_per_row: int = 1) -> Iterator[Sequence[Any]]:
        """Splits keys into runs that fit the dialect's cap on bound parameters per statement."""
        cap = PARAMETER_CAPS.get(self.engine.dialect.name, 32_766) // max(parameters_per_row, 1)
        yield from batched(keys, min(cap, self.PAGE))

    # region Storage Verbs

    def scan_nodes(self) -> Iterator[int]:
        with self.engine.connect() as connection:
            result = connection.execution_options(stream_results=True).execute(
                select(nodes_table.c.node).order_by(nodes_table.c.node)
            )
            for row in result.yield_per(self.PAGE):
                yield row.node

    def has_node(self, node: int) -> bool:
        with self.engine.connect() as connection:
            found = connection.execute(select(nodes_table.c.node).where(nodes_table.c.node == node)).first()
        return found is not None

    def number_of_nodes(self) -> int:
        with self.engine.connect() as connection:
            return connection.execute(select(func.count()).select_from(nodes_table)).scalar_one()

    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        rows = [{"node": int(node)} for node in dict.fromkeys(nodes)]
        with self.engine.begin() as connection:
            if rows:
                connection.execute(self.insert_ignore(nodes_table), rows)

    def drop_nodes(self, nodes: Sequence[int]) -> None:
        keys = list(nodes)
        if not keys:
            return
        with self.engine.begin() as connection:
            for page in self.key_pages(keys, parameters_per_row=2):
                connection.execute(delete(edges_table).where(self._role_clause(page, Role.ANY)))
            for page in self.key_pages(keys):
                connection.execute(delete(nodes_table).where(nodes_table.c.node.in_(page)))

    def scan_edges(self, after: Cursor = None, limit: int | None = None) -> Iterator[Triple]:
        remaining = limit
        cursor = after
        while remaining is None or remaining > 0:
            width = self.PAGE if remaining is None else min(self.PAGE, remaining)
            rows = self._edge_page(None, Role.ANY, cursor, width)
            yield from rows
            if len(rows) < width:
                return
            cursor = rows[-1]
            if remaining is not None:
                remaining -= len(rows)

    def adjacent_edges(self, nodes: Sequence[int], role: Role) -> Iterator[tuple[int, Triple]]:
        keys = list(dict.fromkeys(int(node) for node in nodes))
        for page in self.key_pages(keys, parameters_per_row=2 if role is Role.ANY else 1):
            wanted = set(page)
            cursor: Cursor = None
            while True:
                rows = self._edge_page(page, role, cursor, self.PAGE)
                for source, target, edge in rows:
                    for end in self._ends(source, target, role):
                        if end in wanted:
                            yield end, (source, target, edge)
                if len(rows) < self.PAGE:
                    break
                cursor = rows[-1]

    def find_pairs(self, sources: Sequence[int], targets: Sequence[int]) -> Iterator[tuple[int, Triple]]:
        positions: dict[tuple[int, int], list[int]] = {}
        for position, (source, target) in enumerate(zip(sources, targets, strict=True)):
            positions.setdefault(self.canonical_pair(int(source), int(target)), []).append(position)
        if not positions:
            return
        columns = tuple_(edges_table.c.source, edges_table.c.target)
        key = (edges_table.c.source, edges_table.c.target, edges_table.c.edge)
        with self.engine.connect() as connection:
            for page in batched(positions, self.PAIRS):
                rows = connection.execute(select(*key).where(columns.in_(list(page)))).all()
                for row in rows:
                    for position in positions[(row.source, row.target)]:
                        yield position, (row.source, row.target, row.edge)

    def _edge_page(self, nodes: Sequence[int] | None, role: Role, after: Cursor, width: int) -> list[Triple]:
        """One page of edges in key order, resumed after `after`, read and handed back without a live cursor.

        Keyset order, so a page stays correct while the caller deletes what an earlier page reported.
        """
        key = (edges_table.c.source, edges_table.c.target, edges_table.c.edge)
        statement = select(*key).order_by(*key).limit(width)
        if nodes is not None:
            statement = statement.where(self._role_clause(nodes, role))
        if after is not None:
            statement = statement.where(tuple_(*key) > tuple_(*(literal(part) for part in after)))
        with self.engine.connect() as connection:
            return [(row.source, row.target, row.edge) for row in connection.execute(statement)]

    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        keys = list(nodes)
        if not keys:
            return []
        counts = dict.fromkeys(keys, 0)
        ends = [edges_table.c.source, edges_table.c.target] if role is Role.ANY else [self._role_column(role)]
        with self.engine.connect() as connection:
            for page in self.key_pages(keys):
                for column in ends:
                    counted = (
                        select(column.label("end"), func.count().label("degree"))
                        .where(column.in_(page))
                        .group_by(column)
                    )
                    for row in connection.execute(counted):
                        if row.end in counts:
                            counts[row.end] += row.degree
        return [counts[key] for key in keys]

    def _edge_row(self, source: int, target: int, edge: int) -> dict[str, int]:
        """One edge row with its ends in the orientation this layout stores them in."""
        stored_source, stored_target = self.canonical_pair(int(source), int(target))
        return {"edge": int(edge), "source": stored_source, "target": stored_target}

    def upsert_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        triples = list(zip(sources, targets, edges, strict=True))
        if not triples:
            return
        rows = [self._edge_row(source, target, edge) for source, target, edge in triples]
        with self.engine.begin() as connection:
            for page in self.key_pages(rows, parameters_per_row=3):
                connection.execute(self.upsert_ends(edges_table), list(page))
            ends = [{"node": end} for end in {end for row in rows for end in (row["source"], row["target"])}]
            connection.execute(self.insert_ignore(nodes_table), ends)

    def drop_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        keys = list(edges)
        if not keys:
            return
        with self.engine.begin() as connection:
            for page in self.key_pages(keys):
                connection.execute(delete(edges_table).where(edges_table.c.edge.in_(page)))

    def count_edges(self) -> int:
        with self.engine.connect() as connection:
            return connection.execute(select(func.count()).select_from(edges_table)).scalar_one()

    def biggest_edge_id(self) -> int:
        with self.engine.connect() as connection:
            found = connection.execute(select(func.max(edges_table.c.edge))).scalar()
        return found or 0

    def edge_weights(self, edges: Sequence[int], name: str = WEIGHT) -> list[float | None]:
        """Reads the typed column for `weight`, and falls back to the documents for any other attribute."""
        keys = list(edges)
        if name != WEIGHT or not keys:
            return super().edge_weights(keys, name)
        found: dict[int, float | None] = {}
        with self.engine.connect() as connection:
            for page in self.key_pages(keys):
                statement = select(edges_table.c.edge, edges_table.c.weight).where(edges_table.c.edge.in_(page))
                found.update({row.edge: row.weight for row in connection.execute(statement)})
        return [found.get(key) for key in keys]

    def read_documents(self, store: AttributeStore, keys: Sequence[int]) -> list[Attributes]:
        keys = list(keys)
        if not keys:
            return []
        table = self._attributes_table(store)
        column = self._key_column(store)
        found: dict[int, Attributes] = {}
        with self.engine.connect() as connection:
            for page in self.key_pages(keys):
                rows = connection.execute(select(column, table.c.document).where(column.in_(page)))
                found.update({row[0]: json.loads(row.document) for row in rows})
        if store is AttributeStore.NODES:
            return [found.get(key, {}) for key in keys]
        return [
            found.get(key, {}) if weight is None else {WEIGHT: weight, **found.get(key, {})}
            for key, weight in zip(keys, self.edge_weights(keys), strict=True)
        ]

    def merge_documents(self, store: AttributeStore, keys: Sequence[int], entries: Sequence[Attributes]) -> None:
        keys = list(keys)
        if not keys:
            return
        table = self._attributes_table(store)
        column = self._key_column(store)
        stored = self.read_documents(store, keys)
        merged: dict[int, Attributes] = {}
        for index, (key, held) in enumerate(zip(keys, stored, strict=True)):
            entry = entries[0] if len(entries) == 1 else entries[index]
            merged[int(key)] = {**merged.get(int(key), held), **entry}
        if store is AttributeStore.EDGES:
            merged = self._lift_weights(merged)
        rows = [{column.name: key, "document": json.dumps(entry)} for key, entry in merged.items()]
        with self.engine.begin() as connection:
            for page in self.key_pages(list(merged)):
                connection.execute(delete(table).where(column.in_(page)))
            connection.execute(insert(table), rows)

    def _lift_weights(self, merged: dict[int, Attributes]) -> dict[int, Attributes]:
        """Writes every numeric weight into the edge row's own column and hands back the leftover documents."""
        rows = [{"key": edge, "value": numeric_weight(document.get(WEIGHT))} for edge, document in merged.items()]
        written = update(edges_table).where(edges_table.c.edge == bindparam("key")).values(weight=bindparam("value"))
        with self.engine.begin() as connection:
            for page in self.key_pages(rows, parameters_per_row=2):
                connection.execute(written, list(page))
        return {edge: without_numeric_weight(document) for edge, document in merged.items()}

    def drop_documents(self, store: AttributeStore, keys: Sequence[int]) -> None:
        keys = list(keys)
        if not keys:
            return
        table = self._attributes_table(store)
        cleared = store is AttributeStore.EDGES
        with self.engine.begin() as connection:
            for page in self.key_pages(keys):
                connection.execute(delete(table).where(self._key_column(store).in_(page)))
                if cleared:
                    connection.execute(update(edges_table).where(edges_table.c.edge.in_(page)).values(weight=None))

    def clear(self) -> None:
        with self.engine.begin() as connection:
            for table in (edge_attributes_table, node_attributes_table, edges_table, nodes_table):
                connection.execute(delete(table))
        self.next_edge_id = None

    def close(self) -> None:
        self.engine.dispose()

    # endregion Storage Verbs


class SQLDiGraph(SQLGraph, BaseDiGraph):
    """A directed simple graph stored in a SQL database, as `networkx.DiGraph` is in RAM."""


class SQLMultiGraph(SQLGraph, BaseMultiGraph):
    """An undirected multigraph stored in a SQL database, as `networkx.MultiGraph` is in RAM."""


class SQLMultiDiGraph(SQLGraph, BaseMultiDiGraph):
    """A directed multigraph stored in a SQL database, as `networkx.MultiDiGraph` is in RAM."""
