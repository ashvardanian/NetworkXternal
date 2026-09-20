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
from algorithms import PageRank, triangle_counts

with ClickHouseGraph("clickhouse://graph:graph@localhost:8123/orkut") as graph:
    graph.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 1.5)])
    print(graph.degree[2], list(graph.neighbors(2)))
    print(PageRank.on(graph, iterations=20).run())
```

## Supported Stores

Each backend implements a dozen storage verbs — scanning vertices, finding and upserting edges, merging attribute documents — and inherits every view, traversal and attribute map from `BaseGraph`.
All four NetworkX shapes come with each: `Graph`, `DiGraph`, `MultiGraph` and `MultiDiGraph`, named `SQLiteGraph`, `MongoDiGraph`, `ClickHouseMultiGraph` and so on.

- __[UStore][ustore]__, extra `ustore` — a native graph modality, so a batch of vertices is one call and both ends of an edge are indexed by the engine.
  Attributes live in two document collections; the storage engine is linked at build time, one of `ram`, `nvme`, `rocksdb` or `leveldb`, and persistence is chosen when the graph is opened.
- __[ClickHouse][clickhouse]__, extra `clickhouse` — `ReplacingMergeTree` parts ordered by `(source, target, edge)`, with a mirrored table ordered by target, since a column store has no secondary index.
  Writes are versioned appends, deletes are tombstones collapsed at merge, and degree aggregations run in the engine.
- __[MongoDB][mongodb]__, extra `mongodb` — one document per edge under compound indexes on `(source, target)` and its reverse, so an adjacency read is answered from the index without fetching a document.
  Attributes live in two collections keyed by vertex and by edge.
- __[PostgreSQL][postgres]__, extra `postgres` — B-tree indexes on `(source, target, edge)` and its reverse, JSON attribute tables, and row-constructor `IN` for pair lookups.
  Reads are cheap once the index is in cache; index maintenance dominates the write path.
- __[SQLite][sqlite]__, extra `sqlite` — one file, the same two composite indexes, WAL journaling and a 32,766-parameter cap that bounds every batch.
  A bulk load writes several times the size of the resulting file.
- __[MySQL][mysql]__, extra `mysql` — InnoDB tables clustered by the primary key, with the same composite indexes and JSON attribute tables as the other SQL dialects.
- __[Neo4J][neo4j]__, extra `neo4j` — Cypher over Bolt, relationships reachable from either end, attributes stored as node and relationship properties rather than repacked.
  Range indexes on the vertex and relationship identifiers serve every lookup; the JVM heap is the operative memory limit.
- __[Memgraph][memgraph]__, extra `neo4j` — the same Cypher and the same property storage, held in memory with disk as the overflow, and label indexes declared with the older `CREATE INDEX ON :Label(property)` form.

[ustore]: https://unum.cloud/ustore
[clickhouse]: https://clickhouse.com
[mongodb]: https://www.mongodb.com
[postgres]: https://www.postgresql.org
[sqlite]: https://www.sqlite.org
[mysql]: https://www.mysql.com
[neo4j]: https://neo4j.com
[memgraph]: https://memgraph.com

## External-Memory Algorithms

Vanilla NetworkX walks one vertex at a time, which costs a round trip per step against a store, and pulls a whole neighbourhood to take one of them.
The `algorithms` package drives the store instead: vertex state lives in `array` slots, edges arrive a page at a time, and one hub vertex no longer decides the footprint.

Each algorithm is a class that declares what it holds, and a store says which way it would rather be walked.
An edge-ordered store streams `scan_edges`; a relationship store and UStore index adjacency instead, so a page of vertices is their own unit.

```python
from algorithms import PageRank, Orientation

PageRank.on(graph).run()                                 # the store picks the orientation
PageRank.on(graph, orientation=Orientation.VERTEX).run() # or the caller does
for held in PageRank.on(graph).sweeps():                 # watch it converge
    print(held[42])
pagerank(graph)                                          # the plain function, unchanged
```

An answer is a `VertexMap`, which reads as a `Mapping` over the arrays the walk already held — sixteen bytes a vertex against the eighty-four a `dict` costs, with `.to_dict()` when a caller wants one by name.

| Algorithm                   | Orientation | Held per Vertex | Passes Over the Edges |
| :-------------------------- | :---------- | :-------------- | :-------------------- |
| BreadthFirstLayers          | vertex      | none            | per layer             |
| ShortestPathLengths         | vertex      | 16 B            | per layer             |
| NeighborsOfNeighbors        | vertex      | none            | per layer             |
| DijkstraLengths             | vertex      | 16 B            | per settled vertex    |
| DeltaSteppingLengths        | vertex      | 16 B            | per bucket            |
| ConnectedComponents         | edge        | 16 B            | one                   |
| StronglyConnectedComponents | vertex      | 16 B            | per round             |
| TopologicalOrder            | vertex      | 8 B             | per layer             |
| PageRank                    | either      | 16 B            | per sweep             |
| PersonalizedPageRank        | either      | 16 B            | per sweep             |
| HITS                        | either      | 24 B            | per sweep             |
| BetweennessCentrality       | vertex      | 16 B            | per layer per source  |
| CoreNumbers                 | edge        | 16 B            | per round             |
| TriangleCounts              | edge        | 16 B            | two                   |
| ClusteringCoefficients      | edge        | 16 B            | two                   |
| LabelPropagation            | either      | 16 B            | per sweep             |
| KTruss                      | edge        | 8 B per edge    | per round             |
| DegreeHistogram             | vertex      | none            | one                   |
| SampleNodes, SampleEdges    | either      | none            | one                   |

A bound is declared on the class and measured by the suite, never trusted as prose.
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
