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
from bench.datasets import CATALOGUE, SYNTHETIC, Dataset, from_catalogue, from_path
from bench.limits import BUDGET_SECONDS, MEMORY_GIGABYTES, Budget, limit_memory
from bench.report import Measurement, as_markdown, write_json
from bench.workloads import WORKLOADS, Phase, Workload
from networkxternal.base_api import BaseGraph


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    parser.add_argument("--targets", help="Comma-separated stores to measure, overriding the enabled ones")
    parser.add_argument("--dataset", help="One generated dataset by name, instead of every generated one")
    parser.add_argument("--path", help="An edge list on disk to measure instead of a generated graph")
    parser.add_argument("--catalogue", help="A real graph by name, downloaded on first use")
    parser.add_argument("--datasets", action="store_true", help="List the catalogue and exit")
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Run each workload a second time under tracemalloc to report its allocation peak",
    )
    parser.add_argument("--samples", type=int, default=1_000, help="How many edges the query workloads reuse")
    parser.add_argument("--only", help="Comma-separated workloads to run, by name fragment")
    parser.add_argument("--skip", help="Comma-separated workloads to leave out, by name fragment")
    parser.add_argument("--out", type=Path, help="Where to write the measurements as JSON")
    parser.add_argument(
        "--budget",
        type=float,
        default=BUDGET_SECONDS,
        help="How many seconds one workload may run before reporting what it finished",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=MEMORY_GIGABYTES,
        help="The address space this run may hold, enforced on the process itself",
    )
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


def load(graph: BaseGraph, dataset: Dataset, budget: Budget) -> tuple[int, float]:
    """Imports the dataset until the budget runs out, answering how many edges landed and how long it took.

    A graph larger than the budget is measured on the prefix that fitted, which the report says outright.
    """
    started = perf_counter()
    imported = 0
    page: list[tuple[int, int, float | None]] = []
    for edge in dataset.stream():
        page.append(edge)
        if len(page) < graph.PAGE:
            continue
        imported += write_page(graph, page)
        page = []
        if budget.expired():
            return imported, perf_counter() - started
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


def measure(
    target: Target, dataset: Dataset, workloads: list[Workload], samples: int, budget: float, memory: bool
) -> list[Measurement]:
    """Runs one store through the import and every workload, answering what each one cost."""
    measurements: list[Measurement] = []
    with target.open(dataset.name) as graph:
        graph.clear()
        imported, seconds = load(graph, dataset, Budget(budget))
        peak = 0
        measurements.append(
            Measurement(target.name, dataset.name, "Import: Edge List", Phase.IMPORT, imported, seconds, peak)
        )
        sample = sample_of(dataset, samples)
        for workload in workloads:
            operations, seconds, peak = timed(workload, graph, sample, budget, memory)
            measurements.append(
                Measurement(target.name, dataset.name, workload.name, workload.phase, operations, seconds, peak)
            )
    return measurements


def timed(
    workload: Workload, graph: BaseGraph, sample: list[tuple[int, int, float | None]], budget: float, memory: bool
) -> tuple[int, float, int]:
    """Runs one workload within its budget, answering what it did, how long it took, and its allocation peak.

    Timing and memory are separate passes: `tracemalloc` charges every allocation, which halves the rate
    of an allocation-heavy scan and would report that slowdown as the store's.
    """
    started = perf_counter()
    operations = workload.run(graph, sample, Budget(budget))
    seconds = perf_counter() - started
    if not memory:
        return operations, seconds, 0
    tracemalloc.start()
    workload.run(graph, sample, Budget(budget))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return operations, seconds, peak


def chosen_datasets(options: argparse.Namespace) -> list[Dataset]:
    """The graphs this run measures: a catalogued one, an edge list on disk, or the generated ones."""
    if options.catalogue:
        return [from_catalogue(options.catalogue)]
    if options.path:
        return [from_path(options.path)]
    return [entry for entry in SYNTHETIC if not options.dataset or entry.name == options.dataset]


def print_catalogue() -> None:
    """Prints every catalogued graph with its shape and download size."""
    print(f"{'Name':<20} {'Vertices':>14} {'Edges':>16} {'Download':>12}  Fetched")
    for entry in CATALOGUE:
        size = f"{entry.download_bytes / 1e6:,.1f} MB"
        print(
            f"{entry.name:<20} {entry.vertices:>14,} {entry.edges:>16,} {size:>12}  {'yes' if entry.fetched else 'no'}"
        )


def main() -> None:
    options = arguments()
    if options.datasets:
        print_catalogue()
        return
    limit_memory(options.memory_gb)
    if options.targets:
        os.environ["NETWORKXTERNAL_TARGETS"] = options.targets
    datasets = chosen_datasets(options)
    workloads = wanted_workloads(options.only, options.skip)

    measurements: list[Measurement] = []
    for dataset in datasets:
        for target in wanted_targets():
            print(f"- {target.name} on {dataset.name}")
            try:
                measurements.extend(
                    measure(target, dataset, workloads, options.samples, options.budget, options.memory)
                )
            except Exception as failure:
                print(f"  skipped: {failure}")
    print()
    print(as_markdown(measurements))
    if options.out:
        write_json(measurements, options.out)


if __name__ == "__main__":
    main()
