# Benchmarks

Every store answers the same workloads through the same API, so the numbers compare storage engines rather than client libraries.
A run imports one dataset, then replays a fixed sample of its edges through random reads, analytical traversals and writes.

```sh
uv run python -m bench --targets sqlite                      # Nothing to start, nothing to clean up
uv run python -m bench --targets clickhouse,mongodb          # Against servers from the compose file
uv run python -m bench --path data/orkut/edges.csv           # Against a dataset on the shared filesystem
uv run python -m bench --only "find edge,upsert" --out bench/results/laptop.json
```

## Servers

Containers are capped far below the host, and each one starts on its own profile so a laptop never runs five servers at once.

```sh
docker compose -f bench/docker-compose.yml --profile clickhouse up -d
docker compose -f bench/docker-compose.yml --profile clickhouse down
```

Data directories are `tmpfs`, so a run leaves nothing behind and never writes a benchmark's throwaway pages to disk.
Under a memory cap that matters: a server that would otherwise swap is killed instead, which is a clean failure rather than a machine that stops responding.

## What Is Measured

| Phase     | Workloads                                                                     |
| :-------- | :---------------------------------------------------------------------------- |
| Import    | Bulk edge-list load, in pages sized by the backend                            |
| Reads     | Edge lookup, adjacency of a vertex, neighbours, degree                        |
| Analytics | Neighbours of neighbours, breadth-first traversal to depth 3, PageRank sweeps |
| Writes    | Single-edge upsert, batched upsert, single-edge removal                       |

Workloads are entries in `workloads.py`, selected by `--only` and `--skip` and switched off with the `enabled` flag.
Stores are entries in `config.py`, selected by `--targets` and pointed elsewhere by the environment variable each one names.

## Datasets

Small graphs are generated from a seeded Pareto draw, so degrees are as uneven as a real social graph's and no fixture file ships in the repository.
Large graphs live on the shared filesystem and reach a run through a `data/<name>` symlink, as the [dataset notes](https://github.com/ashvardanian/NetworkXternal/blob/main/data/README.md) describe.
