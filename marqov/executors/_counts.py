"""Shared measurement-count helpers.

Vendors report results in two shapes: raw shot counts, or a probability
histogram. Converting a histogram to counts must conserve the shot total —
downstream code (expectation values, fidelity, SPAM correction) divides by
``sum(counts.values())`` and assumes it equals the requested ``shots``.

Naive per-bin rounding does not conserve: three bins at 1/3 of 1000 shots
round to 333 each, losing a shot.
"""

from __future__ import annotations

from typing import Any


def extract_sampler_counts(result: Any) -> dict[str, int]:
    """Extract measurement counts from a Qiskit SamplerV2 result.

    Shared by IBMExecutor and MarqovDevice.run so the two paths cannot
    disagree on bit order or on what to do with multiple classical registers
    (marqov-sdk#161).

    Args:
        result: SamplerV2 PrimitiveResult.

    Returns:
        Mapping of bitstring to count, with qubit 0 leftmost.

    Raises:
        ValueError: If no classical register carrying measurement data can be
            resolved from the result's DataBin.
        NotImplementedError: If the result carries more than one classical
            register.
    """
    pub_result = result[0]
    data_bin = pub_result.data

    # SamplerV2 returns a BitArray per classical register. Find it by
    # capability (it exposes get_counts), NOT by taking dir()[0]: dir() is
    # alphabetical and DataBin also exposes mapping helpers ('items',
    # 'keys', 'ndim', 'shape', 'size', 'values'), so dir()[0] is 'items',
    # a bound method, for any register sorting after it (e.g. the 'meas'
    # register that measure_all() creates).
    keys = getattr(data_bin, "keys", None)
    if callable(keys):
        # Qiskit >= 1.2: DataBin declares its register names.
        names = list(keys())
    else:
        names = [n for n in dir(data_bin) if not n.startswith("_")]

    bit_arrays: list[Any] = [
        candidate
        for candidate in (getattr(data_bin, name, None) for name in names)
        if hasattr(candidate, "get_counts")
    ]

    if not bit_arrays:
        # Returning {} here reads to the caller as "the circuit produced no
        # outcomes", which is indistinguishable from a genuine zero-shot run
        # and hides a result shape we do not understand.
        raise ValueError(
            "SamplerV2 result carries no measurement data: none of its "
            f"DataBin fields {sorted(names)} is a BitArray. Ensure the "
            "circuit has a classical register (e.g. measure_all())."
        )

    if len(bit_arrays) > 1:
        # Each BitArray covers one register. Returning just the first one
        # yields a bitstring narrower than the measurement, silently
        # mis-indexing every downstream consumer (fidelity, SPAM,
        # expectation values). Joining them needs a defined register order
        # and a decision about whether the reversal applies within or
        # across registers, so fail loudly rather than guess.
        raise NotImplementedError(
            f"Result has multiple classical registers ({len(bit_arrays)}); "
            "Marqov cannot yet combine them into a single bitstring. "
            "Use a single classical register (e.g. measure_all())."
        )

    # Qiskit is little-endian (qubit 0 = rightmost); Marqov's convention is
    # qubit 0 = leftmost. Reverse, exactly as AzureQuantumExecutor does for
    # the same framework.
    return {
        bitstring[::-1]: count
        for bitstring, count in bit_arrays[0].get_counts().items()
    }


def allocate_counts(probabilities: dict[str, float], shots: int) -> dict[str, int]:
    """Convert a probability histogram into counts summing exactly to ``shots``.

    Uses the largest-remainder (Hamilton) method: floor every bin, then hand
    the leftover shots to the largest fractional remainders first.

    Keys are passed through untouched, so callers may use whatever key shape
    the vendor gave them (bitstrings, state indices, ...). Bit-order
    normalization is the caller's responsibility.

    Args:
        probabilities: Mapping of outcome key to probability. Probabilities are
            assumed non-negative; they need not sum exactly to 1.
        shots: The number of shots to allocate.

    Returns:
        Mapping of outcome key to integer count, summing to ``shots``. Bins
        allocated zero counts are omitted. Empty when there is nothing to
        allocate.
    """
    if not probabilities or shots <= 0:
        return {}

    counts: dict[str, int] = {}
    remainders: dict[str, float] = {}
    allocated = 0
    for key, probability in probabilities.items():
        exact = float(probability) * shots
        base = int(exact)  # floor (probabilities are non-negative)
        counts[key] = base
        remainders[key] = exact - base
        allocated += base

    leftover = shots - allocated
    if leftover > 0:
        # Hand extra shots to the largest fractional remainders first. Cycles
        # if the histogram is truncated and leftover exceeds the bin count.
        ordered = sorted(remainders, key=lambda k: remainders[k], reverse=True)
        for i in range(leftover):
            counts[ordered[i % len(ordered)]] += 1
    elif leftover < 0:
        # Probabilities summed above 1: reclaim from the smallest remainders.
        # The bound is computed ONCE, before the loop. Recomputing it inline
        # is a live-lock trap: `-leftover` shrinks as shots are reclaimed, so
        # the bound collapses toward `i` and the loop exits still
        # over-allocated — reintroducing the total != shots defect.
        ordered = sorted(remainders, key=lambda k: remainders[k])
        limit = len(ordered) * (-leftover + 1)
        i = 0
        while leftover < 0 and i < limit:
            key = ordered[i % len(ordered)]
            if counts[key] > 0:
                counts[key] -= 1
                leftover += 1
            i += 1

    return {key: count for key, count in counts.items() if count > 0}
