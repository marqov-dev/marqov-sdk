"""Tests for marqov.benchmarking.fidelity module."""

import pytest

from marqov.benchmarking.fidelity import (
    classical_fidelity,
    fidelity_with_uniform,
    normalized_fidelity,
)
from marqov.executors.base import ExecutionResult


def _result(counts: dict[str, int]) -> ExecutionResult:
    """Build an ExecutionResult with the given counts."""
    return ExecutionResult(
        counts=counts,
        backend="test",
        execution_time_ms=0.0,
        shots=sum(counts.values()),
    )


class TestClassicalFidelity:
    """Tests for classical (Bhattacharyya) fidelity."""

    def test_identical_distributions(self) -> None:
        """Identical distributions have fidelity 1."""
        dist = {"00": 0.5, "11": 0.5}
        assert classical_fidelity(dist, dist) == pytest.approx(1.0)

    def test_disjoint_support(self) -> None:
        """Distributions with no shared outcomes have fidelity 0."""
        assert classical_fidelity({"00": 1.0}, {"11": 1.0}) == pytest.approx(0.0)

    def test_partial_overlap_known_value(self) -> None:
        """Non-trivial value: F({00:.5,01:.5}, {00:1}) = (sqrt(.5))^2 = 0.5."""
        assert classical_fidelity({"00": 0.5, "01": 0.5}, {"00": 1.0}) == pytest.approx(
            0.5
        )

    def test_symmetric(self) -> None:
        """Fidelity is symmetric in its arguments."""
        a = {"00": 0.7, "01": 0.3}
        b = {"00": 0.4, "01": 0.6}
        assert classical_fidelity(a, b) == pytest.approx(classical_fidelity(b, a))

    def test_accepts_raw_counts(self) -> None:
        """Raw shot counts give the same result as normalized probabilities."""
        counts = {"00": 300, "01": 100}  # -> {0.75, 0.25}
        probs = {"00": 0.75, "01": 0.25}
        assert classical_fidelity(counts, {"00": 1.0}) == pytest.approx(
            classical_fidelity(probs, {"00": 1.0})
        )

    def test_accepts_execution_result(self) -> None:
        """Accepts ExecutionResult directly, using its counts."""
        ideal = _result({"00": 500, "11": 500})
        measured = _result({"00": 480, "11": 470, "01": 30, "10": 20})
        assert 0.0 < classical_fidelity(ideal, measured) < 1.0

    def test_width_mismatch_is_aligned(self) -> None:
        """Trimmed-width keys are zero-padded to align, not treated as disjoint."""
        # "0" should pad to "00" and match the padded distribution.
        assert classical_fidelity({"0": 1.0}, {"00": 1.0}) == pytest.approx(1.0)


class TestFidelityWithUniform:
    """Tests for fidelity_with_uniform."""

    def test_peaked_two_qubit(self) -> None:
        """A delta distribution over 4 states has uniform-fidelity 1/4."""
        assert fidelity_with_uniform({"00": 1.0}) == pytest.approx(0.25)

    def test_uniform_is_one(self) -> None:
        """The uniform distribution has fidelity 1 with itself."""
        dist = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
        assert fidelity_with_uniform(dist) == pytest.approx(1.0)

    def test_num_states_override(self) -> None:
        """Explicit num_states overrides the 2**width default (e.g. boson sampling)."""
        assert fidelity_with_uniform({"20": 1.0}, num_states=3) == pytest.approx(1 / 3)

    def test_infers_width(self) -> None:
        """Default outcome space is 2**(bitstring width)."""
        assert fidelity_with_uniform({"000": 1.0}) == pytest.approx(1 / 8)

    def test_num_states_below_support_raises(self) -> None:
        """num_states smaller than the populated support is rejected."""
        dist = {"00": 0.5, "01": 0.5}
        with pytest.raises(ValueError, match="smaller than the number"):
            fidelity_with_uniform(dist, num_states=1)


class TestNormalizedFidelity:
    """Tests for the Lubinski/QED-C normalized fidelity."""

    def test_perfect_backend(self) -> None:
        """Backend matching the ideal distribution scores 1."""
        ideal = {"00": 0.5, "11": 0.5}
        assert normalized_fidelity(ideal, ideal) == pytest.approx(1.0)

    def test_uniform_backend_scores_zero(self) -> None:
        """A pure-noise (uniform) backend scores 0, not the raw fidelity."""
        ideal = {"00": 1.0}
        uniform = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
        assert normalized_fidelity(ideal, uniform) == pytest.approx(0.0)

    def test_worse_than_uniform_clamps_to_zero(self) -> None:
        """Backend anti-correlated with the ideal clamps to 0 (never negative)."""
        ideal = {"00": 1.0}
        backend = {"11": 1.0}
        assert normalized_fidelity(ideal, backend) == 0.0

    def test_between_zero_and_one(self) -> None:
        """A partially-degraded backend lands strictly between 0 and 1."""
        ideal = {"00": 1.0}
        backend = {"00": 0.7, "01": 0.1, "10": 0.1, "11": 0.1}
        assert 0.0 < normalized_fidelity(ideal, backend) < 1.0

    def test_accepts_execution_results(self) -> None:
        """Works directly on the ExecutionResult objects executors return."""
        ideal = _result({"00": 512, "11": 512})
        backend = _result({"00": 480, "11": 470, "01": 40, "10": 34})
        assert 0.0 < normalized_fidelity(ideal, backend) <= 1.0

    def test_trimmed_backend_keys_still_score_high(self) -> None:
        """Regression: a correct backend with leading-zeros trimmed must NOT
        be flagged as broken (was the headline review bug)."""
        ideal = _result({"00": 500, "11": 500})
        # Same distribution but the "00" key arrived trimmed to "0".
        trimmed = {"0": 500, "11": 500}
        assert normalized_fidelity(ideal, trimmed) == pytest.approx(1.0)

    def test_zero_shot_backend_scores_zero(self) -> None:
        """A backend that returned no shots scores 0, it does not crash."""
        ideal = _result({"00": 500, "11": 500})
        empty = _result({})  # ExecutionResult with no counts
        assert normalized_fidelity(ideal, empty) == 0.0

    def test_empty_ideal_raises(self) -> None:
        """An empty reference is a caller error, not a zero score."""
        with pytest.raises(ValueError, match="must not be empty"):
            normalized_fidelity({}, {"00": 1.0})

    def test_maximally_mixed_ideal_raises(self) -> None:
        """A degenerate (uniform) reference fails loudly rather than returning 0."""
        ideal = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
        backend = {"00": 0.5, "11": 0.5}
        with pytest.raises(ValueError, match="maximally mixed"):
            normalized_fidelity(ideal, backend)

    def test_num_states_below_support_raises(self) -> None:
        """num_states below the ideal's support is rejected, not silently zeroed."""
        ideal = {"00": 0.5, "11": 0.5}
        backend = {"00": 0.5, "11": 0.5}
        with pytest.raises(ValueError, match="smaller than the number of populated"):
            normalized_fidelity(ideal, backend, num_states=1)

    def test_maximally_mixed_ideal_raises_even_with_empty_backend(self) -> None:
        """Regression: an invalid `ideal` must raise regardless of whether
        `backend` happens to be empty -- the empty-backend shortcut must not
        bypass ideal-side validation."""
        ideal = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}  # maximally mixed
        with pytest.raises(ValueError, match="maximally mixed"):
            normalized_fidelity(ideal, {})

    def test_num_states_below_support_raises_even_with_empty_backend(self) -> None:
        """Same regression as above, for the num_states-vs-support check."""
        ideal = {"00": 0.5, "11": 0.5}
        with pytest.raises(ValueError, match="smaller than the number of populated"):
            normalized_fidelity(ideal, {}, num_states=1)

    def test_num_states_baseline_independent_of_backend_width(self) -> None:
        """Regression: the uniform-noise baseline depends only on `ideal`'s own
        width, not on an incidental extra bit the backend happens to report
        (e.g. an ancilla) -- the same relative signal must score the same
        either way."""
        ideal = {"00": 0.5, "11": 0.5}
        backend_2bit = {"00": 0.45, "01": 0.05, "10": 0.05, "11": 0.45}
        # Same relative signal, reported with an extra leading always-0 bit.
        backend_3bit = {"000": 0.45, "001": 0.05, "010": 0.05, "011": 0.45}
        assert normalized_fidelity(ideal, backend_2bit) == pytest.approx(
            normalized_fidelity(ideal, backend_3bit)
        )

    def test_all_zero_backend_scores_zero_like_empty(self) -> None:
        """Regression: a nonempty backend dict whose counts are all zero (e.g.
        every SPAM-corrected bin rounded to 0) scores 0 like a truly empty
        backend, rather than crashing on 'total mass must be positive'."""
        ideal = {"00": 0.5, "11": 0.5}
        assert normalized_fidelity(ideal, {"00": 0, "11": 0}) == 0.0

    def test_generator_backend_not_spuriously_empty(self) -> None:
        """Regression: a single-use iterable `backend` (e.g. a generator of
        (key, value) pairs) must not be drained by a redundant double
        extraction and then misread as empty."""
        ideal = {"00": 0.5, "11": 0.5}
        backend_gen = iter({"00": 0.5, "11": 0.5}.items())
        assert normalized_fidelity(ideal, backend_gen) == pytest.approx(1.0)

    def test_matches_reference_formula(self) -> None:
        """Result equals the explicit (F_b - F_u)/(1 - F_u) computation."""
        ideal = {"00": 0.5, "11": 0.5}
        backend = {"00": 0.45, "11": 0.4, "01": 0.1, "10": 0.05}
        f_backend = classical_fidelity(ideal, backend)
        f_uniform = fidelity_with_uniform(ideal)
        expected = max((f_backend - f_uniform) / (1 - f_uniform), 0.0)
        assert normalized_fidelity(ideal, backend) == pytest.approx(expected)


class TestValidation:
    """Input validation."""

    def test_empty_distribution_raises(self) -> None:
        """An empty distribution is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            classical_fidelity({}, {"0": 1.0})

    def test_zero_mass_raises(self) -> None:
        """A distribution with no positive mass is rejected."""
        with pytest.raises(ValueError, match="positive"):
            fidelity_with_uniform({"0": 0.0, "1": 0.0})

    def test_negative_value_raises(self) -> None:
        """A negative count/probability is rejected before it reaches sqrt."""
        with pytest.raises(ValueError, match="negative"):
            classical_fidelity({"0": 2.0, "1": -1.0}, {"0": 1.0})

    def test_padding_collision_raises(self) -> None:
        """Keys that collide after zero-padding are a real ambiguity -> error."""
        with pytest.raises(ValueError, match="collide"):
            fidelity_with_uniform({"1": 0.5, "01": 0.5})

    def test_nan_value_raises(self) -> None:
        """A NaN count/probability is rejected. NaN comparisons are always
        False in Python, so a plain `< 0` check alone would silently let it
        through and propagate to a NaN final result."""
        with pytest.raises(ValueError, match="non-finite"):
            classical_fidelity({"0": float("nan"), "1": 0.5}, {"0": 1.0})

    def test_infinite_value_raises(self) -> None:
        """An infinite count/probability is rejected. It would otherwise
        divide out to inf/inf = NaN downstream."""
        with pytest.raises(ValueError, match="non-finite"):
            classical_fidelity({"0": float("inf"), "1": 0.5}, {"0": 1.0})

    def test_invalid_type_raises_value_error_not_type_error(self) -> None:
        """An unsupported distribution type raises the documented ValueError,
        not a bare TypeError -- e.g. a caller forwarding None as a 'no data
        yet' sentinel gets a catchable, informative error."""
        with pytest.raises(ValueError, match="expected an ExecutionResult"):
            classical_fidelity(None, {"0": 1.0})

    def test_int_keyed_distribution_raises(self) -> None:
        """Integer-keyed distributions (a real pattern for statevector-index
        -based executors) raise a clear ValueError instead of crashing on
        `len()` of an int before any documented validation runs."""
        with pytest.raises(ValueError, match="must be bitstrings"):
            classical_fidelity({0: 512, 3: 488}, {0: 1.0})

    def test_classical_fidelity_does_not_exceed_one(self) -> None:
        """Regression: identical distributions must not exceed the documented
        [0, 1] range due to ordinary floating-point rounding in sqrt/sum.
        This exact distribution (found by search, not adversarially crafted)
        overshoots to 1.0000000000000004 without the clamp."""
        dist = {
            "00000": 0.21894291090539744,
            "00001": 0.16969750829963992,
            "00010": 0.3976074585155386,
            "00011": 0.2137521222794242,
        }
        assert classical_fidelity(dist, dist) <= 1.0
