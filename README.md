# NetworkXternal

![NetworkXternal thumbnail](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/NetworkXternal.jpg?raw=true)

NetworkXternal gives [NetworkX](https://github.com/networkx/networkx)-shaped graphs a home outside of RAM.
The same `nodes`, `edges`, `degree` and `neighbors` you already write against, served by UStore, ClickHouse, MongoDB, PostgreSQL, SQLite, MySQL, Neo4J or Memgraph, so a graph grows from Gigabytes to Terabytes without your code noticing.
It is not free — a round-trip costs more than a pointer dereference — but it is a very short way to find out whether your analysis survives the jump.

```sh
pip install "networkxternal[clickhouse]"
```

```python
from networkxternal.clickhouse import ClickHouseGraph
from networkxternal.algorithms import pagerank, triangle_counts

with ClickHouseGraph("clickhouse://graph:graph@localhost:8123/orkut") as graph:
    graph.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 1.5)])
    print(graph.degree[2], list(graph.neighbors(2)))
    print(pagerank(graph, iterations=20))
```

## Supported Stores

Each backend implements a dozen storage verbs — scanning vertices, finding and upserting edges, merging attribute documents — and inherits every view, traversal and attribute map from `BaseGraph`.
All four NetworkX shapes come with each: `Graph`, `DiGraph`, `MultiGraph` and `MultiDiGraph`, named `SQLiteGraph`, `MongoDiGraph`, `ClickHouseMultiGraph` and so on.

- __[UStore][ustore]__, extra `ustore` — speaks graphs natively, so a batch of vertices is one call, and attributes live in two document collections.
- __[ClickHouse][clickhouse]__, extra `clickhouse` — append-only versioned rows merged by identity, where aggregations like degree histograms are home turf.
- __[MongoDB][mongodb]__, extra `mongodb` — compound indexes serve adjacency without touching a document, and attributes live in two collections.
- __[PostgreSQL][postgres]__, extra `postgres` — feature-rich and B-tree indexed, quick to read and slow to update, with attributes as JSON text.
- __[SQLite][sqlite]__, extra `sqlite` — hard to beat below a Gigabyte, at the cost of brutal write amplification.
- __[MySQL][mysql]__, extra `mysql` — ubiquitous, clustered by primary key, with the same JSON attribute tables as the other SQL dialects.
- __[Neo4J][neo4j]__, extra `neo4j` — Cypher over Bolt, keeping attributes as node and relationship properties, unhappy under memory pressure.
- __[Memgraph][memgraph]__, extra `neo4j` — the same Cypher and the same properties, in memory, far faster.

[ustore]: https://unum.cloud/ustore
[clickhouse]: https://clickhouse.com
[mongodb]: https://www.mongodb.com
[postgres]: https://www.postgresql.org
[sqlite]: https://www.sqlite.org
[mysql]: https://www.mysql.com
[neo4j]: https://neo4j.com
[memgraph]: https://memgraph.com

## External-Memory Algorithms

Vanilla NetworkX walks one vertex at a time, which costs one round-trip per step against a store.
The algorithms in `networkxternal.algorithms` expand a whole frontier per call instead, so a traversal costs a round-trip per level rather than per vertex, and they hold vertex state only — never the adjacency of the graph.

| Function                       | What It Holds                            | What It Costs                                      |
| :----------------------------- | :--------------------------------------- | :------------------------------------------------- |
| `breadth_first_layers`         | The frontier and the visited set         | One round-trip per level                           |
| `shortest_path_lengths`        | One depth per reached vertex             | One round-trip per level                           |
| `connected_components`         | One label per vertex                     | One sweep per round-trip page, until labels settle |
| `pagerank`                     | Two floats per vertex                    | One sweep per round-trip page, weights optional    |
| `core_numbers`                 | One degree per vertex                    | The neighbourhood of a peeled vertex only          |
| `triangle_counts`              | The neighbourhood of the wanted vertices | One page of lookups per intersection round         |
| `sample_nodes`, `sample_edges` | The reservoir                            | One pass, nothing else buffered                    |

Everything else NetworkX ships still applies where the graph fits, and `__networkx_backend__` is declared for the dispatch protocol NetworkX 3.x uses.

## Benchmarks

`python -m bench` measures every store through the same API, so the numbers compare storage engines rather than client libraries.
See the [benchmark notes](https://github.com/ashvardanian/NetworkXternal/blob/main/bench/README.md) for the workloads, the container caps and how to point a run at a dataset on disk.

## Development

```sh
uv sync --extra sqlite --group test   # One backend's driver, plus the suite
uv run pytest                         # The conformance suite, against SQLite
uv run ruff format . && uv run ruff check .
```

The same suite runs against any backend by naming it, which is how each one is verified:

```sh
docker compose -f bench/docker-compose.yml --profile clickhouse up -d
NETWORKXTERNAL_TEST_BACKEND=clickhouse uv run pytest -q
```

Python 3.12 is the floor, and the suite is exercised on the free-threaded 3.14t build as well, where a graph instance is safe to share between threads whenever its driver is.
