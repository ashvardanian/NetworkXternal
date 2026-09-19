"""What one benchmark run is allowed to take: a memory ceiling it cannot exceed and a clock it answers to.

A run shares the machine, so it caps its own address space rather than trusting the workload to behave,
and every workload stops at its deadline and reports what it finished instead of stalling the sweep.
"""

from __future__ import annotations

import resource
from dataclasses import dataclass, field
from time import perf_counter

MEMORY_GIGABYTES = 8
"""The address space one run may hold, well below what any machine this runs on has."""

BUDGET_SECONDS = 60.0
"""How long one workload may run before it reports what it managed."""


def limit_memory(gigabytes: int = MEMORY_GIGABYTES) -> None:
    """Caps this process's address space, so a runaway workload dies here rather than on the machine."""
    ceiling = gigabytes * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    wanted = ceiling if hard == resource.RLIM_INFINITY else min(ceiling, hard)
    resource.setrlimit(resource.RLIMIT_AS, (wanted, hard))


@dataclass
class Budget:
    """The clock one workload runs against, and what it reports when the clock runs out."""

    seconds: float = BUDGET_SECONDS
    """How long the workload may run."""

    started: float = field(default_factory=perf_counter)
    """When it started, which `expired` measures against."""

    def expired(self) -> bool:
        """Whether the workload has spent its budget and should report what it has."""
        return perf_counter() - self.started >= self.seconds
