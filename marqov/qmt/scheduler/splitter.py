"""Result attribution — decomposes composite measurement results back to per-job results.

After a multi-tenant execution, the QPU returns bitstrings spanning all qubits.
The splitter uses the PackingPlan's qubit mappings to extract each job's bits
and reconstruct per-job measurement counts.
"""

from __future__ import annotations

from marqov.qmt.models import PackingPlan, PackingResult, QubitMapping


def split_results(
    composite_counts: dict[str, int],
    plan: PackingPlan,
    shots: int,
) -> list[PackingResult]:
    job_counts: dict[str, dict[str, int]] = {
        mapping.job_id: {} for mapping in plan.mappings
    }

    for bitstring, count in composite_counts.items():
        # Pad short bitstrings (e.g., from simulators that omit unused qubits)
        if len(bitstring) < plan.total_qubits:
            bitstring = _pad_bitstring(bitstring, plan)
        for mapping in plan.mappings:
            job_bits = _extract_job_bits(bitstring, mapping, plan.total_qubits)
            if job_bits in job_counts[mapping.job_id]:
                job_counts[mapping.job_id][job_bits] += count
            else:
                job_counts[mapping.job_id][job_bits] = count

    return [
        PackingResult(
            job_id=mapping.job_id,
            counts=job_counts[mapping.job_id],
            shots=shots,
        )
        for mapping in plan.mappings
    ]


def _pad_bitstring(bitstring: str, plan: PackingPlan) -> str:
    """Expand a short bitstring to total_qubits length.

    Simulators like QuantumFlow only include qubits that have gates,
    producing bitstrings shorter than total_qubits when there are gaps
    in qubit indices. This function maps the compact bitstring back to
    full-width by inserting '0' at unused qubit positions.

    Args:
        bitstring: Compact bitstring from the simulator.
        plan: The packing plan with qubit mappings.

    Returns:
        Bitstring of length total_qubits with '0' at unused positions.
    """
    # Collect all physical qubits actually used (sorted)
    used_qubits = sorted(plan.physical_qubits_used | plan.guard_qubits)
    if len(bitstring) == len(used_qubits):
        # Map compact positions back to full-width
        full = ["0"] * plan.total_qubits
        for compact_idx, physical_idx in enumerate(used_qubits):
            if compact_idx < len(bitstring) and physical_idx < plan.total_qubits:
                full[physical_idx] = bitstring[compact_idx]
        return "".join(full)
    # Fallback: pad with zeros on the right
    return bitstring.ljust(plan.total_qubits, "0")


def _extract_job_bits(
    bitstring: str,
    mapping: QubitMapping,
    total_qubits: int,
) -> str:
    sorted_logical = sorted(mapping.logical_to_physical.keys())
    bits = []
    for logical in sorted_logical:
        physical = mapping.logical_to_physical[logical]
        bits.append(bitstring[physical])
    return "".join(bits)
