"""Shared measurement-count helpers.

Vendors report results in two shapes: raw shot counts, or a probability
histogram. Converting a histogram to counts must conserve the shot total —
downstream code (expectation values, fidelity, SPAM correction) divides by
``sum(counts.values())`` and assumes it equals the requested ``shots``.

Naive per-bin rounding does not conserve: three bins at 1/3 of 1000 shots
round to 333 each, losing a shot.
"""

from __future__ import annotations


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
