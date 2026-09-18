"""NetworkX-like graphs over external memory: one API, many stores.

Backends are imported on first use, so a missing driver only fails the backend that needs it.
"""

from __future__ import annotations

import importlib
from typing import Any

from networkxternal.base_api import (
    AttributeStore,
    BaseDiGraph,
    BaseGraph,
    BaseMultiDiGraph,
    BaseMultiGraph,
    DegreeView,
    EdgeView,
    NodeView,
    Role,
)

__version__ = "0.4.0"

BACKENDS = {
    "MongoGraph": "networkxternal.mongodb",
    "MongoDiGraph": "networkxternal.mongodb",
    "MongoMultiGraph": "networkxternal.mongodb",
    "MongoMultiDiGraph": "networkxternal.mongodb",
    "SQLGraph": "networkxternal.base_sql",
    "SQLDiGraph": "networkxternal.base_sql",
    "SQLMultiGraph": "networkxternal.base_sql",
    "SQLMultiDiGraph": "networkxternal.base_sql",
    "SQLiteGraph": "networkxternal.sqlite",
    "SQLiteDiGraph": "networkxternal.sqlite",
    "SQLiteMultiGraph": "networkxternal.sqlite",
    "SQLiteMultiDiGraph": "networkxternal.sqlite",
    "PostgresGraph": "networkxternal.postgres",
    "PostgresDiGraph": "networkxternal.postgres",
    "PostgresMultiGraph": "networkxternal.postgres",
    "PostgresMultiDiGraph": "networkxternal.postgres",
    "MySQLGraph": "networkxternal.mysql",
    "MySQLDiGraph": "networkxternal.mysql",
    "MySQLMultiGraph": "networkxternal.mysql",
    "MySQLMultiDiGraph": "networkxternal.mysql",
    "ClickHouseGraph": "networkxternal.clickhouse",
    "ClickHouseDiGraph": "networkxternal.clickhouse",
    "ClickHouseMultiGraph": "networkxternal.clickhouse",
    "ClickHouseMultiDiGraph": "networkxternal.clickhouse",
    "Neo4JGraph": "networkxternal.neo4j",
    "Neo4JDiGraph": "networkxternal.neo4j",
    "Neo4JMultiGraph": "networkxternal.neo4j",
    "Neo4JMultiDiGraph": "networkxternal.neo4j",
    "UStoreGraph": "networkxternal.ustore",
    "UStoreDiGraph": "networkxternal.ustore",
    "UStoreMultiGraph": "networkxternal.ustore",
    "UStoreMultiDiGraph": "networkxternal.ustore",
}
"""Where each exported backend class lives, so importing the package pulls in no driver."""

__all__ = [
    "AttributeStore",
    "BaseDiGraph",
    "BaseGraph",
    "BaseMultiDiGraph",
    "BaseMultiGraph",
    "DegreeView",
    "EdgeView",
    "NodeView",
    "Role",
    *BACKENDS,
]


def __getattr__(name: str) -> Any:
    if name not in BACKENDS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(BACKENDS[name]), name)


def __dir__() -> list[str]:
    return sorted(__all__)
