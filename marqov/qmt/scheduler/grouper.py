"""Job grouping for QPU multi-tenancy.

Groups a queue of QMTJobs into batches that can co-execute on a single
device, respecting qubit capacity and guard qubit requirements.

Uses greedy first-fit bin-packing — simple and correct for the research phase.
"""

from __future__ import annotations

from marqov.qmt.models import QMTJob


def group_jobs(
    jobs: list[QMTJob],
    device_qubits: int,
    *,
    min_guard_qubits: int = 1,
) -> list[list[QMTJob]]:
    if not jobs:
        return []

    sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)

    groups: list[list[QMTJob]] = []
    group_sizes: list[int] = []

    for job in sorted_jobs:
        placed = False
        for i, group in enumerate(groups):
            space_needed = job.num_qubits + min_guard_qubits
            if group_sizes[i] + space_needed <= device_qubits:
                group.append(job)
                group_sizes[i] += space_needed
                placed = True
                break

        if not placed:
            groups.append([job])
            group_sizes.append(job.num_qubits)

    return groups
