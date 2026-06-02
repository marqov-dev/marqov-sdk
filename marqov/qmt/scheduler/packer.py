"""Qubit packing — assigns jobs to physical qubit subsets on a device.

Uses the NoiseProfile's cross-talk matrix to place jobs on qubit regions
with minimal mutual interference, separated by guard qubits.

Phase 1 strategy: sequential placement with greedy cross-talk minimization.
"""

from __future__ import annotations

import numpy as np

from marqov.qmt.models import (
    NoiseProfile,
    PackingPlan,
    QMTJob,
    QubitMapping,
)


def pack_jobs(
    jobs: list[QMTJob],
    profile: NoiseProfile,
    *,
    min_guard_qubits: int = 1,
) -> PackingPlan:
    """Pack jobs onto a device using noise-aware greedy placement.

    Each job is assigned a contiguous block of physical qubits, chosen to
    minimize cross-talk with already-placed jobs and individual qubit error
    rates.  Guard qubits are inserted between adjacent placements.

    Args:
        jobs: Jobs to pack onto the device.
        profile: Noise characterization of the target device.
        min_guard_qubits: Minimum idle qubits between any two job regions.

    Returns:
        A PackingPlan describing the placement.

    Raises:
        ValueError: If the jobs cannot fit on the device.
    """
    total_qubits_needed = sum(j.num_qubits for j in jobs) + min_guard_qubits * max(
        len(jobs) - 1, 0
    )
    if total_qubits_needed > profile.num_qubits:
        raise ValueError(
            f"Jobs need {total_qubits_needed} qubits but device has {profile.num_qubits}"
        )

    if len(jobs) == 1:
        job = jobs[0]
        start = _best_start_for_first_job(job.num_qubits, profile)
        mapping = _make_mapping(job, start)
        return PackingPlan(
            jobs=jobs,
            mappings=[mapping],
            guard_qubits=set(),
            device_name=profile.device_name,
            total_qubits=profile.num_qubits,
        )

    mappings: list[QubitMapping] = []
    occupied: set[int] = set()

    for job in jobs:
        start = _best_start(job.num_qubits, profile, occupied, min_guard_qubits)
        mapping = _make_mapping(job, start)
        mappings.append(mapping)
        guard_lo = max(0, start - min_guard_qubits)
        guard_hi = min(profile.num_qubits, start + job.num_qubits + min_guard_qubits)
        for q in range(guard_lo, guard_hi):
            occupied.add(q)

    job_qubits: set[int] = set()
    for m in mappings:
        job_qubits |= m.physical_qubits
    guard_qubits = occupied - job_qubits

    return PackingPlan(
        jobs=jobs,
        mappings=mappings,
        guard_qubits=guard_qubits,
        device_name=profile.device_name,
        total_qubits=profile.num_qubits,
    )


def _make_mapping(job: QMTJob, start: int) -> QubitMapping:
    """Create a QubitMapping that places logical qubits starting at *start*."""
    logical_to_physical = {i: start + i for i in range(job.num_qubits)}
    return QubitMapping(job_id=job.job_id, logical_to_physical=logical_to_physical)


def _best_start_for_first_job(num_qubits: int, profile: NoiseProfile) -> int:
    """Find the lowest-error contiguous block for the first job."""
    best_start = 0
    best_error = float("inf")
    for start in range(profile.num_qubits - num_qubits + 1):
        error = float(np.sum(profile.qubit_error_rates[start : start + num_qubits]))
        if error < best_error:
            best_error = error
            best_start = start
    return best_start


def _best_start(
    num_qubits: int,
    profile: NoiseProfile,
    occupied: set[int],
    min_guard: int,
) -> int:
    """Find the best contiguous block that avoids occupied qubits and minimizes cross-talk."""
    best_start = -1
    best_score = float("inf")

    for start in range(profile.num_qubits - num_qubits + 1):
        candidate = set(range(start, start + num_qubits))
        if candidate & occupied:
            continue

        score = float(profile.cross_talk_between(candidate, occupied))
        score += float(np.sum(profile.qubit_error_rates[start : start + num_qubits]))

        if score < best_score:
            best_score = score
            best_start = start

    if best_start == -1:
        raise ValueError("Cannot find valid placement — device is too full")

    return best_start
