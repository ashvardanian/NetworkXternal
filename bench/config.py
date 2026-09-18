"""Which stores a run touches and where they live, read from the environment with sane local defaults."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path


@dataclass(frozen=True)
class Target:
    """One store a benchmark runs against, and how to open a graph in it."""

    name: str
    """The name printed in the report."""

    module: str
    """The module holding the graph class."""

    graph: str
    """The class within that module."""

    url: str
    """The connection string, with `{dataset}` standing for the dataset's name."""

    variable: str
    """The environment variable overriding the connection string."""

    enabled: bool = True
    """Whether a run touches this store; a server that is not up is switched off here, not commented out."""

    extras: dict[str, str] = field(default_factory=dict)
    """Extra keyword arguments the graph class takes."""

    def open(self, dataset: str):
        """Opens the graph this target names, for one dataset, creating the directory a local store needs."""
        url = os.getenv(self.variable, self.url).replace("{dataset}", dataset)
        local = url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else url if "://" not in url else None
        if local is not None:
            Path(local).parent.mkdir(parents=True, exist_ok=True)
        return getattr(import_module(self.module), self.graph)(url, **self.extras)


TARGETS = (
    Target("SQLite", "networkxternal.sqlite", "SQLiteGraph", "sqlite:///tmp/{dataset}.db3", "URI_SQLITE"),
    Target("MongoDB", "networkxternal.mongodb", "MongoGraph", "mongodb://localhost:27017/{dataset}", "URI_MONGODB"),
    Target(
        "PostgreSQL",
        "networkxternal.postgres",
        "PostgresGraph",
        "postgresql+psycopg://graph:graph@localhost:5432/{dataset}",
        "URI_POSTGRES",
    ),
    Target(
        "MySQL",
        "networkxternal.mysql",
        "MySQLGraph",
        "mysql://root:graph@127.0.0.1:3306/{dataset}",
        "URI_MYSQL",
    ),
    Target(
        "ClickHouse",
        "networkxternal.clickhouse",
        "ClickHouseGraph",
        "clickhouse://graph:graph@localhost:8123/{dataset}",
        "URI_CLICKHOUSE",
    ),
    Target("Neo4J", "networkxternal.neo4j", "Neo4JGraph", "bolt://localhost:7687/{dataset}", "URI_NEO4J"),
    Target("Memgraph", "networkxternal.memgraph", "MemgraphGraph", "bolt://localhost:7688/{dataset}", "URI_MEMGRAPH"),
    Target("UStore", "networkxternal.ustore", "UStoreGraph", "tmp/ustore/{dataset}", "URI_USTORE"),
)
"""Every store a run can touch; `NETWORKXTERNAL_TARGETS` narrows it to a comma-separated subset."""


def wanted_targets() -> Iterator[Target]:
    """The enabled targets, narrowed by `NETWORKXTERNAL_TARGETS` when it names any."""
    chosen = {name.strip().lower() for name in os.getenv("NETWORKXTERNAL_TARGETS", "").split(",") if name.strip()}
    for target in TARGETS:
        if chosen:
            if target.name.lower() in chosen:
                yield target
        elif target.enabled:
            yield target
