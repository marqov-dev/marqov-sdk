"""Distribution-fidelity metrics for application-level benchmarking.

Implements the *normalized fidelity* of Lubinski et al. ("Application-Oriented
Performance Benchmarks for Quantum Computing", the QED-C suite): an
application-level quality score that compares a backend's output distribution
against an ideal reference distribution and normalizes out the trivial
uniform-noise baseline. Unlike raw classical fidelity, a device that returns
pure noise scores 0 rather than a misleadingly non-zero value.

This complements the SPAM/readout characterization in
:mod:`marqov.benchmarking.spam`: SPAM measures per-qubit measurement error,
whereas normalized fidelity measures how faithfully a full application's output
distribution survives execution on real hardware.

All functions accept either an :class:`~marqov.executors.base.ExecutionResult`
(the object returned by every executor) or a bare ``{bitstring: count}`` /
``{bitstring: probability}`` dict.

Bit-order caveat: these metrics compare bitstrings as keys. They tolerate
differing *widths* (a distribution whose keys have had leading zeros trimmed is
zero-padded to a common width before comparison), but they cannot detect a
differing qubit *ordering* (endianness). The caller is responsible for ensuring
the ideal and backend distributions share the same qubit-order convention — see
``tests/test_azure_bitorder.py`` for why this matters across backends.

Ported from Open QBench (PCSS-Quantum/open-qbench, Apache-2.0).
"""

from __future__ import annotations

import math

from marqov.executors.base import ExecutionResult

# A distribution keyed by bitstring. Values may be raw shot counts or
# probabilities; both are accepted and normalized internally. Callers may also
# pass an ExecutionResult directly.
Distribution = dict[str, float]
DistributionLike = ExecutionResult | Distribution


def _extract(dist: DistributionLike) -> Distribution:
    """Pull a plain ``{bitstring: value}`` dict out of the accepted input types."""
    if isinstance(dist, ExecutionResult):
        return dict(dist.counts)
    return dict(dist)


def _to_probabilities(dist: DistributionLike) -> tuple[Distribution, int]:
    """Validate and normalize a distribution to probabilities.

    Zero-pads bitstrings to the distribution's maximum key width so that keys
    that have had leading zeros trimmed still align.

    Returns:
        A ``(probabilities, width)`` tuple where ``width`` is the (padded)
        bitstring length.

    Raises:
        ValueError: If the distribution is empty, contains a negative value, has
            non-positive total mass, or has keys that collide after zero-padding.
    """
    raw = _extract(dist)
    if not raw:
        raise ValueError("distribution must not be empty")

    width = max(len(k) for k in raw)
    total = 0.0
    for key, value in raw.items():
        if value < 0:
            raise ValueError(f"distribution has a negative value for {key!r}: {value}")
        total += value
    if total <= 0:
        raise ValueError(f"distribution total mass must be positive, got {total}")

    probs: Distribution = {}
    for key, value in raw.items():
        padded = key.zfill(width)
        if padded in probs:
            raise ValueError(
                f"bitstrings collide after zero-padding to width {width}: {padded!r}"
            )
        probs[padded] = value / total
    return probs, width


def _repad(probs: Distribution, width: int) -> Distribution:
    """Widen already-normalized probability keys to ``width`` (leading zeros)."""
    if width == len(next(iter(probs))):
        return probs
    out: Distribution = {}
    for key, value in probs.items():
        padded = key.zfill(width)
        if padded in out:
            raise ValueError(
                f"bitstrings collide after zero-padding to width {width}: {padded!r}"
            )
        out[padded] = value
    return out


def _bhattacharyya(p: Distribution, q: Distribution) -> float:
    r"""Classical fidelity :math:`\left(\sum_i \sqrt{p_i q_i}\right)^2` of two
    already-normalized, width-aligned distributions."""
    overlap = sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in p.keys() | q.keys())
    return overlap**2


def classical_fidelity(dist_a: DistributionLike, dist_b: DistributionLike) -> float:
    r"""Classical (Bhattacharyya) fidelity between two distributions.

    Defined as :math:`F(P, Q) = \left(\sum_i \sqrt{p_i q_i}\right)^2`, which
    lies in ``[0, 1]``: 1 for identical distributions, 0 for distributions with
    disjoint support. Inputs are zero-padded to a common bitstring width before
    comparison.

    Args:
        dist_a: First distribution (ExecutionResult, counts, or probabilities).
        dist_b: Second distribution (ExecutionResult, counts, or probabilities).

    Returns:
        Classical fidelity in ``[0, 1]``.
    """
    p, width_a = _to_probabilities(dist_a)
    q, width_b = _to_probabilities(dist_b)
    width = max(width_a, width_b)
    return _bhattacharyya(_repad(p, width), _repad(q, width))


def fidelity_with_uniform(
    dist: DistributionLike,
    num_states: int | None = None,
) -> float:
    """Classical fidelity of a distribution with the uniform distribution.

    The uniform distribution is taken over ``num_states`` outcomes. When
    ``num_states`` is not given it defaults to ``2 ** width``, inferred from the
    bitstring width — the standard gate-based case. Pass an explicit
    ``num_states`` for non-gate modalities (e.g. boson sampling, where the number
    of valid samples is not a power of two).

    Args:
        dist: Distribution (ExecutionResult, counts, or probabilities).
        num_states: Size of the uniform outcome space. Defaults to
            ``2 ** width``.

    Returns:
        Classical fidelity with the uniform distribution, in ``[0, 1]``.

    Raises:
        ValueError: If ``num_states`` is smaller than the number of populated
            outcomes (which would make the fidelity exceed 1).
    """
    probs, width = _to_probabilities(dist)
    if num_states is None:
        num_states = 2**width
    if num_states < len(probs):
        raise ValueError(
            f"num_states ({num_states}) is smaller than the number of populated "
            f"outcomes ({len(probs)}); the uniform baseline is ill-defined"
        )
    uniform_prob = 1 / num_states
    overlap = sum(math.sqrt(prob * uniform_prob) for prob in probs.values())
    return overlap**2


def normalized_fidelity(
    ideal: DistributionLike,
    backend: DistributionLike,
    num_states: int | None = None,
) -> float:
    r"""Normalized fidelity of Lubinski et al. (QED-C).

    Rescales the classical fidelity between the ideal and backend distributions
    so that a uniform (pure-noise) backend scores 0 and a perfect backend scores
    1:

    .. math::

        F_\text{norm} = \max\!\left(0,
            \frac{F_\text{backend} - F_\text{uniform}}{1 - F_\text{uniform}}\right)

    where :math:`F_\text{backend}` is the classical fidelity between the ideal
    and backend distributions and :math:`F_\text{uniform}` is the classical
    fidelity of the ideal distribution with the uniform distribution.

    Args:
        ideal: Ideal/reference distribution, e.g. from a noiseless simulator
            (ExecutionResult, counts, or probabilities).
        backend: Distribution measured on the device under test. An empty
            distribution (a backend that returned no shots) scores 0.
        num_states: Size of the uniform outcome space used for the baseline.
            Defaults to ``2 ** width`` inferred from the (padded) distributions.

    Returns:
        Normalized fidelity in ``[0, 1]``.

    Raises:
        ValueError: If the ideal distribution is empty, if ``num_states`` is
            smaller than the ideal's support, or if the ideal distribution is
            maximally mixed (a degenerate reference with no signal above the
            uniform baseline — distinct from a noisy backend, so it fails loudly
            rather than returning a misleading 0).
    """
    ideal_probs, ideal_width = _to_probabilities(ideal)

    if not _extract(backend):
        # Backend returned no outcomes (e.g. zero shots) — zero fidelity, not an
        # error. `ideal` was already validated above.
        return 0.0

    backend_probs, backend_width = _to_probabilities(backend)
    width = max(ideal_width, backend_width)
    ideal_probs = _repad(ideal_probs, width)
    backend_probs = _repad(backend_probs, width)

    if num_states is None:
        num_states = 2**width
    if num_states < len(ideal_probs):
        raise ValueError(
            f"num_states ({num_states}) is smaller than the ideal distribution's "
            f"support ({len(ideal_probs)}); the uniform baseline is ill-defined"
        )

    backend_fidelity = _bhattacharyya(ideal_probs, backend_probs)
    uniform_prob = 1 / num_states
    uniform_fidelity = (
        sum(math.sqrt(prob * uniform_prob) for prob in ideal_probs.values()) ** 2
    )

    denominator = 1 - uniform_fidelity
    if denominator <= 0:
        raise ValueError(
            "ideal distribution is maximally mixed; normalized fidelity is "
            "undefined (the reference carries no signal above the uniform "
            "baseline). Check that `ideal` is a correct noiseless reference."
        )

    return max((backend_fidelity - uniform_fidelity) / denominator, 0.0)
