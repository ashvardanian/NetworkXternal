"""Runs the workloads against every enabled store and prints the rates.

    python -m bench --targets sqlite,clickhouse --dataset synthetic-10k
    python -m bench --path data/orkut/edges.csv --samples 2000

Import loads the whole dataset; every other workload runs over a sample of it, so a bigger graph
costs more to load but not more to query, and the numbers stay comparable across stores.
"""

from __future__ import annotations

import argparse
import os
import tracemalloc
from pathlib import Path
from time import perf_counter

from bench.config import Target, wanted_targets
from bench.datasets import SYNTHETIC, Dataset, from_path
from bench.report import Measurement, as_markdown, write_json
from bench.workloads import WORKLOADS, Phase, Workload
from networkxternal.base_api import BaseGraph


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    parser.add_argument("--targets", help="Comma-separated stores to measure, overriding the enabled ones")
    parser.add_argument("--dataset", help="One generated dataset by name, instead of every generated one")
    parser.add_argument("--path", help="An edge list on disk to measure instead of a generated graph")
    parser.add_argument("--samples", type=int, default=1_000, help="How many edges the query workloads reuse")
    parser.add_argument("--only", help="Comma-separated workloads to run, by name fragment")
    parser.add_argument("--skip", help="Comma-separated workloads to leave out, by name fragment")
    parser.add_argument("--out", type=Path, help="Where to write the measurements as JSON")
    return parser.parse_args()


def wanted_workloads(only: str | None, skip: str | None) -> list[Workload]:
    """The workloads a run measures, narrowed by name fragments rather than by editing the registry."""
    chosen = [workload for workload in WORKLOADS if workload.enabled or only]
    if only:
        fragments = [fragment.strip().lower() for fragment in only.split(",") if fragment.strip()]
        chosen = [workload for workload in chosen if any(f in workload.name.lower() for f in fragments)]
    if skip:
        fragments = [fragment.strip().lower() for fragment in skip.split(",") if fragment.strip()]
        chosen = [workload for workload in chosen if not any(f in workload.name.lower() for f in fragments)]
    return chosen


def load(graph: BaseGraph, dataset: Dataset) -> tuple[int, float]:
    """Imports the whole dataset, answering how many edges landed and how long it took."""
    started = perf_counter()
    imported = 0
    page: list[tuple[int, int, float | None]] = []
    for edge in dataset.stream():
        page.append(edge)
        if len(page) == graph.PAGE:
            imported += write_page(graph, page)
            page = []
    imported += write_page(graph, page)
    return imported, perf_counter() - started


def write_page(graph: BaseGraph, page: list[tuple[int, int, float | None]]) -> int:
    if not page:
        return 0
    graph.add_edges_from_arrays(
        [source for source, _, _ in page],
        [target for _, target, _ in page],
        columns={"weight": [weight for _, _, weight in page]},
    )
    return len(page)


def sample_of(dataset: Dataset, count: int) -> list[tuple[int, int, float | None]]:
    """The edges every query workload reuses, taken from the head of the stream so each store sees the same ones."""
    sample: list[tuple[int, int, float | None]] = []
    for edge in dataset.stream():
        sample.append(edge)
        if len(sample) == count:
            break
    return sample


def measure(target: Target, dataset: Dataset, workloads: list[Workload], samples: int) -> list[Measurement]:
    """Runs one store through the import and every workload, answering what each one cost."""
    measurements: list[Measurement] = []
    with target.open(dataset.name) as graph:
        graph.clear()
        tracemalloc.start()
        imported, seconds = load(graph, dataset)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        measurements.append(
            Measurement(target.name, dataset.name, "Import: Edge List", Phase.IMPORT, imported, seconds, peak)
        )
        sample = sample_of(dataset, samples)
        for workload in workloads:
            operations, seconds, peak = timed(workload, graph, sample)
            measurements.append(
                Measurement(target.name, dataset.name, workload.name, workload.phase, operations, seconds, peak)
            )
    return measurements


def timed(workload: Workload, graph: BaseGraph, sample: list[tuple[int, int, float | None]]) -> tuple[int, float, int]:
    """Runs one workload, answering how many operations it did, how long it took, and its allocation peak."""
    tracemalloc.start()
    started = perf_counter()
    operations = workload.run(graph, sample)
    seconds = perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return operations, seconds, peak


def main() -> None:
    options = arguments()
    if options.targets:
        os.environ["NETWORKXTERNAL_TARGETS"] = options.targets
    datasets = (
        [from_path(options.path)]
        if options.path
        else [entry for entry in SYNTHETIC if not options.dataset or entry.name == options.dataset]
    )
    workloads = wanted_workloads(options.only, options.skip)

    measurements: list[Measurement] = []
    for dataset in datasets:
        for target in wanted_targets():
            print(f"- {target.name} on {dataset.name}")
            try:
                measurements.extend(measure(target, dataset, workloads, options.samples))
            except Exception as failure:
                print(f"  skipped: {failure}")
    print()
    print(as_markdown(measurements))
    if options.out:
        write_json(measurements, options.out)


if __name__ == "__main__":
    main()
