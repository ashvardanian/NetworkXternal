# Benchmarks

Every store answers the same workloads through the same API, so the numbers compare storage engines rather than client libraries.
A run imports one dataset, then replays a fixed sample of its edges through random reads, analytical traversals and writes.

```sh
uv run python -m bench --datasets                          # The catalogue, with sizes
uv run python -m bench --targets sqlite                    # Nothing to start, nothing to clean up
uv run python -m bench --catalogue amazon-1m               # A real graph, downloaded on first use
uv run python -m bench --path ~/datasets/orkut-117m/*.gz   # An edge list already on disk
uv run python -m bench --only "find edge,upsert" --out bench/results/laptop.json
```

## Limits

Two limits apply to every run, because this machine is shared.

- __Memory__: the process caps its own address space at 8 GB, so a runaway workload dies here rather than on the machine.
  `--memory-gb` moves it.
- __Time__: each workload runs for at most 60 seconds and reports what it finished, so a slow store gives partial results instead of stalling the sweep.
  `--budget` moves it, and the import obeys the same clock, which is how a graph larger than the budget is measured on the prefix that fitted.

Servers run in containers capped well below the host, each on its own profile so a laptop never runs five at once.

```sh
docker compose -f bench/docker-compose.yml --profile clickhouse up -d
docker compose -f bench/docker-compose.yml --profile clickhouse down
```

Data directories are `tmpfs`, so a run leaves nothing behind.
Under a memory cap that matters: a server that would otherwise swap is killed instead, which is a clean failure rather than a machine that stops responding.

## Datasets

Nothing ships in the repository, and the benchmark looks in exactly one place: `data/<name>`.
Where those bytes actually live is the machine's decision — point the link at whatever storage has room, then fetch.

```sh
mkdir -p /mnt/data/sources/amazon-1m && ln -s /mnt/data/sources/amazon-1m data/amazon-1m
uv run python -m bench --catalogue amazon-1m
```

Names are lowercase with dashes and the scale last, so the link says what it holds.

| Name               |      Vertices |           Edges | Download | Fetched |
| :----------------- | ------------: | --------------: | -------: | :------ |
| `facebook-88k`     |         4,039 |          88,234 |   0.2 MB | yes     |
| `astroph-198k`     |        18,772 |         198,110 |   1.5 MB | yes     |
| `enron-184k`       |        36,692 |         183,831 |   1.1 MB | yes     |
| `amazon-1m`        |       334,863 |         925,872 |   4.5 MB | yes     |
| `youtube-3m`       |     1,134,890 |       2,987,624 |  10.6 MB | yes     |
| `google-5m`        |       875,713 |       5,105,039 |  21.2 MB | yes     |
| `pokec-31m`        |     1,632,803 |      30,622,564 | 132.5 MB | yes     |
| `livejournal-35m`  |     3,997,962 |      34,681,189 | 124.3 MB | yes     |
| `orkut-117m`       |     3,072,441 |     117,185,083 | 447.3 MB | yes     |
| `friendster-1806m` |    65,608,366 |   1,806,067,135 |   9.4 GB | no      |
| `sinaweibo-261m`   |    58,655,849 |     261,321,071 |   1.7 GB | no      |
| `clueweb09-7940m`  | 1,684,868,322 |   7,939,635,651 |    40 GB | no      |
| `wdc2012-128b`     | 3,563,000,000 | 128,000,000,000 |   2.5 TB | no      |

The last four are listed for a machine with room for them rather than fetched here.
`wdc2012-128b` is the multi-terabyte target: the Common Crawl 2012 hyperlink graph, 3.5 billion pages and 128 billion links, which is the scale this API is meant to survive and the one worth re-measuring against if the client is ever rewritten in a compiled language.

The generated graphs remain for smoke tests, and their skew is nothing like a real graph's — one vertex holds a third of the edges, which stresses hubs and flatters everything else.
`synthetic-10k` also collapses to 5,464 distinct edges once duplicate pairs merge, so the name overstates it.

## What Is Measured

| Phase     | Workloads                                                                                                 |
| :-------- | :-------------------------------------------------------------------------------------------------------- |
| Import    | Bulk edge-list load, in pages sized by the backend                                                        |
| Reads     | Edge lookup by pair, a batch of pair lookups, a full edge scan, adjacency of a vertex, neighbours, degree |
| Analytics | Neighbours of neighbours, breadth-first traversal to depth 3, PageRank sweeps                             |
| Writes    | Single-edge upsert, batched upsert, single-edge removal                                                   |

Workloads are entries in `workloads.py`, selected by `--only` and `--skip` and switched off with the `enabled` flag.
Stores are entries in `config.py`, selected by `--targets` and pointed elsewhere by the environment variable each one names.

## Results

Measured on one machine, 32 cores, servers in capped containers, 200 sampled edges per read workload, a 45-second budget and the 8 GB ceiling.
Operations per second, higher is better.

### amazon-1m, 335k Vertices and 926k Edges

| Workload                            |  SQLite | PostgreSQL |   MySQL | MongoDB | ClickHouse |  Neo4J | Memgraph | UStore in RAM |
| :---------------------------------- | ------: | ---------: | ------: | ------: | ---------: | -----: | -------: | ------------: |
| Import: Edge List                   |  10,638 |      4,493 |   1,588 |  10,941 |      1,692 |  6,817 |    5,073 |        17,990 |
| Random Reads: Find Edge             |   3,202 |        755 |     898 |   1,775 |        212 |    253 |    1,166 |        10,754 |
| Random Reads: Find Edges Batch      |  49,111 |     20,823 |  34,534 |  52,922 |      7,561 | 23,515 |    6,573 |       390,940 |
| Sequential Reads: Scan Edges        | 195,392 |    180,486 | 169,574 |  53,755 |    876,822 | 41,859 |   42,688 |       834,058 |
| Random Reads: Find Edges of Node    |   2,012 |        647 |     840 |   1,632 |        169 |    717 |      950 |        13,431 |
| Random Reads: Find Neighbors        |   1,580 |        383 |     481 |     937 |        118 |    428 |      584 |        13,426 |
| Random Reads: Count Degree          |  61,118 |     21,519 |  41,157 |  97,510 |     16,658 |  9,049 |   61,157 |     1,617,665 |
| Analytics: Neighbors of Neighbors   |     661 |        145 |     218 |      47 |         36 |    220 |       71 |            11 |
| Analytics: Breadth-First to Depth 3 |  40,734 |      6,638 |   5,591 |   5,224 |      4,497 | 10,175 |    3,601 |         2,158 |
| Random Writes: Upsert Edge          |     551 |        156 |     167 |     998 |         89 |    449 |      608 |         9,938 |
| Random Writes: Upsert Edges Batch   |  14,238 |      5,489 |   2,627 |  26,311 |      5,001 | 11,446 |    5,961 |        71,377 |
| Random Writes: Remove Edge          |     762 |        236 |     227 |     763 |         74 |    410 |      595 |        10,619 |

Three things the table says plainly.
A column store wins the scan and loses the point lookup, since `FINAL` resolves a version per read.
A batch of pair lookups beats the same pairs one at a time by an order of magnitude on every store, which is the whole argument for pushing the pair filter into the engine.
UStore's in-memory engine leads everywhere except the two traversals, where it is last by a wide margin — its page of 65,536 vertices is too coarse for a frontier lookup, and that is the next thing to measure rather than to explain away.

The UStore column is the `ram` engine, which is in memory and not comparable to the disk-backed stores.
The `nvme`, `rocksdb` and `leveldb` engines are separate builds; see the note in the main README.

### What the Edge-Centric Rewrite Changed

Same machine, same generated graph, before and after the storage verbs were replaced.

| Measure                        | Vertex-Centric |                 Edge-Centric |
| :----------------------------- | -------------: | ---------------------------: |
| Import 1M edges, no attributes |        83.06 s |                       8.93 s |
| Import 100k edges              |         1.82 s |                       0.71 s |
| `Random Reads: Find Edge`      |          350/s |                      3,136/s |
| Rows fetched per point lookup  |          1,378 | the multiplicity of the pair |

The remaining import cost is attributes: 1M edges load in 8.9 seconds without weights and at roughly a sixth of that rate with them, since every page still writes its documents separately.
That is what the next wave's server-side merge is for.
