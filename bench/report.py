"""Measurements and how they are printed: one row per store, workload and dataset, as Markdown or JSON."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Measurement:
    """What one workload did on one store, over one dataset."""

    target: str
    """The store it ran against."""

    dataset: str
    """The graph it ran over."""

    workload: str
    """The operation measured."""

    phase: str
    """Which part of the run the workload belongs to."""

    operations: int
    """How many logical operations the run performed."""

    seconds: float
    """How long they took, wall-clock."""

    peak_bytes: int = 0
    """The high-water mark of Python allocation during the run, which is what a view holds."""

    @property
    def rate(self) -> float:
        """Operations per second, or zero where nothing was measured."""
        return self.operations / self.seconds if self.seconds > 0 else 0.0


def as_markdown(measurements: list[Measurement]) -> str:
    """One table per dataset of operations per second, and one of the peak bytes each workload held."""
    lines: list[str] = []
    for dataset in dict.fromkeys(entry.dataset for entry in measurements):
        rows = [entry for entry in measurements if entry.dataset == dataset]
        lines.append(f"### {dataset}, operations per second\n")
        lines.extend(table_of(rows, lambda entry: f"{entry.rate:,.0f}"))
        lines.append(f"### {dataset}, peak bytes held\n")
        lines.extend(table_of(rows, lambda entry: f"{entry.peak_bytes:,.0f}"))
    return "\n".join(lines)


def table_of(rows: list[Measurement], cell: Callable[[Measurement], str]) -> list[str]:
    """One table of the given rows, workloads down the side and stores across the top."""
    targets = list(dict.fromkeys(entry.target for entry in rows))
    lines = ["| Workload | " + " | ".join(targets) + " |", "| :--- | " + " | ".join("---:" for _ in targets) + " |"]
    for workload in dict.fromkeys(entry.workload for entry in rows):
        found = {entry.target: entry for entry in rows if entry.workload == workload}
        lines.append(f"| {workload} | " + " | ".join(cell(found[t]) if t in found else "—" for t in targets) + " |")
    lines.append("")
    return lines


def write_json(measurements: list[Measurement], path: Path) -> None:
    """Writes every measurement as one JSON array, for a later run to compare against."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(entry) for entry in measurements], indent=2) + "\n")
