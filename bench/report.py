"""Measurements and how they are printed: one row per store, workload and dataset, as Markdown or JSON."""

from __future__ import annotations

import json
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

    @property
    def rate(self) -> float:
        """Operations per second, or zero where nothing was measured."""
        return self.operations / self.seconds if self.seconds > 0 else 0.0


def as_markdown(measurements: list[Measurement]) -> str:
    """A table per dataset, workloads down the side and stores across the top, in operations per second."""
    lines: list[str] = []
    for dataset in dict.fromkeys(entry.dataset for entry in measurements):
        rows = [entry for entry in measurements if entry.dataset == dataset]
        targets = list(dict.fromkeys(entry.target for entry in rows))
        lines.append(f"### {dataset}\n")
        lines.append("| Workload | " + " | ".join(targets) + " |")
        lines.append("| :--- | " + " | ".join("---:" for _ in targets) + " |")
        for workload in dict.fromkeys(entry.workload for entry in rows):
            cells = []
            for target in targets:
                found = next((e for e in rows if e.workload == workload and e.target == target), None)
                cells.append(f"{found.rate:,.0f}" if found else "—")
            lines.append(f"| {workload} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_json(measurements: list[Measurement], path: Path) -> None:
    """Writes every measurement as one JSON array, for a later run to compare against."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(entry) for entry in measurements], indent=2) + "\n")
