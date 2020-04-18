import os
from typing import Optional
import importlib

from pystats.file import StatsFile

from pygraphdb.table_sqlite import SQLite, SQLiteMem
from pygraphdb.table_mysql import MySQL
from pygraphdb.table_postgres import PostgreSQL
from pygraphdb.docs_mongo import MongoDB
from pygraphdb.graph_neo4j import Neo4j

try:
    # pylint: disable=undefined-variable
    importlib.reload(unumdb_python)
except NameError:
    import unumdb_python
from unumdb_python import GraphLSM, GraphBPlus
# print('Using UnumDB version: ', unumdb_python.__dict__)


count_nodes = int(os.getenv('COUNT_NODES', '0'))
count_edges = int(os.getenv('COUNT_EDGES', '0'))
count_finds = int(os.getenv('COUNT_FINDS', '10000'))
count_analytics = int(os.getenv('COUNT_ANALYTICS', '1000'))
count_changes = int(os.getenv('COUNT_CHANGES', '10000'))
device_name = os.getenv('DEVICE_NAME', 'Unknown Device')

report_path = 'artifacts/stats.md'
stats_path = 'artifacts/stats.json'
stats = StatsFile(stats_path)

_datasets = [
    # Path, Number of Nodes, Number of Edges
    ('/Users/av/Code/PyGraphDB/artifacts/graph-test/all.csv', 8, 10),
    ('/Users/av/Datasets/graph-communities/all.csv', 0, 52310),
    ('/Users/av/Datasets/graph-eachmovie-ratings/all.csv', 0, 2811716),
    # ('/Users/av/Datasets/graph-patent-citations/all.csv', 0, 16518947),
    # ('/Users/av/Datasets/graph-mouse-gene/all.csv', 0, 14506199),
    # ('/Users/av/Datasets/graph-human-brain/all.csv', 0, 87273967),
]
dataset_test = _datasets[0][0]
datasets = [x[0] for x in _datasets[1:]]

wrapper_types = [
    # SQLiteMem,
    SQLite,
    MySQL,
    # MongoDB,
    # PostgreSQL,
    # Neo4j,
    # GraphLSM,
    # GraphBPlus,
]

_wrappers = [
    # Type, Environment Variable, Default Value
    (GraphLSM, 'URI_UNUMDB_LSM', '/Users/av/DBs/unumdb.GraphLSM/<dataset>'),
    (GraphBPlus, 'URI_UNUMDB_BPLUS', '/Users/av/DBs/unumdb.GraphBPLus/<dataset>.db3'),
    (SQLiteMem, 'URI_SQLITE_MEM', 'sqlite:///:memory:'),
    (SQLite, 'URI_SQLITE', 'sqlite:////Users/av/DBs/sqlite/<dataset>.db3'),
    (MySQL, 'URI_MYSQL', 'mysql://av:temptemp@0.0.0.0:3306/<dataset>'),
    (PostgreSQL, 'URI_PGSQL', 'postgres://0.0.0.0:5432/<dataset>'),
    (Neo4j, 'URI_NEO4J', 'bolt://0.0.0.0:7687/<dataset>'),
    (MongoDB, 'URI_MONGO', 'mongodb://0.0.0.0:27017/<dataset>'),
]


def dataset_number_of_edges(dataset_path: str) -> int:
    for d in _datasets:
        if d[0] != dataset_path:
            continue
        return d[2]
    return 0


def dataset_name(dataset_path: str) -> str:
    parts = dataset_path.split('/')
    if len(parts) > 1:
        return parts[-2]
    return dataset_path


def wrapper_name(cls: type) -> str:
    if isinstance(cls, type):
        return cls.__name__
    else:
        return cls.__class__.__name__


def database_url(cls: type, dataset_path: str) -> Optional[str]:
    name = dataset_name(dataset_path)
    for w in _wrappers:
        if w[0] != cls:
            continue
        url = os.getenv(w[1], w[2])
        url = url.replace('<dataset>', name)
        return url
    return None
