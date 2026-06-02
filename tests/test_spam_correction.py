"""Tests for SPAM correction functions."""

import numpy as np
import pytest

from marqov.benchmarking.spam import (
    ConfusionMatrix,
    SPAMResult,
    apply_spam_correction,
    build_correction_matrix,
)


class TestBuildCorrectionMatrix:
    """Tests for build_correction_matrix()."""

    def test_identity_confusion_gives_identity_correction(self) -> None:
        """Perfect readout: correction matrix is identity."""
        spam = SPAMResult(
            matrices={
                0: ConfusionMatrix(p00=1.0, p01=0.0, p10=0.0, p11=1.0),
            },
            shots=1000,
        )
        correction = build_correction_matrix(spam)
        expected = np.eye(2)
        np.testing.assert_array_almost_equal(correction, expected)

    def test_two_qubit_asymmetric_errors(self) -> None:
        """2-qubit correction with asymmetric error rates."""
        spam = SPAMResult(
            matrices={
                0: ConfusionMatrix(p00=0.98, p01=0.02, p10=0.05, p11=0.95),
                1: ConfusionMatrix(p00=0.97, p01=0.03, p10=0.04, p11=0.96),
            },
            shots=1000,
        )
        correction = build_correction_matrix(spam)
        assert correction.shape == (4, 4)
        q0 = np.array([[0.98, 0.05], [0.02, 0.95]])
        q1 = np.array([[0.97, 0.04], [0.03, 0.96]])
        joint = np.kron(q0, q1)
        expected_correction = np.linalg.inv(joint)
        np.testing.assert_array_almost_equal(correction, expected_correction)

    def test_empty_spam_result_raises(self) -> None:
        """Empty SPAMResult raises ValueError."""
        spam = SPAMResult(matrices={}, shots=1000)
        with pytest.raises(ValueError, match="no confusion matrices"):
            build_correction_matrix(spam)

    def test_singular_matrix_raises(self) -> None:
        """Degenerate confusion matrix raises LinAlgError."""
        spam = SPAMResult(
            matrices={
                0: ConfusionMatrix(p00=1.0, p01=0.0, p10=1.0, p11=0.0),
            },
            shots=1000,
        )
        with pytest.raises(np.linalg.LinAlgError):
            build_correction_matrix(spam)

    def test_qubit_ordering_is_sorted(self) -> None:
        """Qubits are ordered by index, not insertion order."""
        cm_a = ConfusionMatrix(p00=0.9, p01=0.1, p10=0.2, p11=0.8)
        cm_b = ConfusionMatrix(p00=0.95, p01=0.05, p10=0.03, p11=0.97)
        spam = SPAMResult(matrices={1: cm_b, 0: cm_a}, shots=1000)
        correction = build_correction_matrix(spam)
        q0 = np.array([[0.9, 0.2], [0.1, 0.8]])
        q1 = np.array([[0.95, 0.03], [0.05, 0.97]])
        expected = np.linalg.inv(np.kron(q0, q1))
        np.testing.assert_array_almost_equal(correction, expected)


class TestApplySpamCorrection:
    """Tests for apply_spam_correction()."""

    def test_identity_correction_preserves_counts(self) -> None:
        """Identity correction matrix returns counts unchanged."""
        counts = {"00": 500, "11": 500}
        correction = np.eye(4)
        result = apply_spam_correction(counts, correction)
        assert result == {"00": 500, "11": 500}

    def test_round_trip_with_known_errors(self) -> None:
        """Apply known noise then correct — should recover original."""
        spam = SPAMResult(
            matrices={
                0: ConfusionMatrix(p00=0.98, p01=0.02, p10=0.05, p11=0.95),
                1: ConfusionMatrix(p00=0.97, p01=0.03, p10=0.04, p11=0.96),
            },
            shots=1000,
        )
        q0 = np.array([[0.98, 0.05], [0.02, 0.95]])
        q1 = np.array([[0.97, 0.04], [0.03, 0.96]])
        joint_confusion = np.kron(q0, q1)
        ideal_vec = np.array([1000, 0, 0, 0], dtype=float)
        noisy_vec = joint_confusion @ ideal_vec
        noisy_counts = {}
        for i, c in enumerate(noisy_vec):
            bs = format(i, "02b")
            if c > 0:
                noisy_counts[bs] = int(round(c))
        correction = build_correction_matrix(spam)
        corrected = apply_spam_correction(noisy_counts, correction)
        assert corrected.get("00", 0) >= 950
        assert sum(corrected.values()) == sum(noisy_counts.values())

    def test_negative_clamping(self) -> None:
        """Correction that produces negatives clamps to zero."""
        correction = np.array([[2.0, -1.0], [-1.0, 2.0]])
        counts = {"0": 100, "1": 900}
        result = apply_spam_correction(counts, correction)
        for v in result.values():
            assert v >= 0
        assert sum(result.values()) == 1000

    def test_empty_counts_returns_empty(self) -> None:
        """Empty input returns empty dict."""
        result = apply_spam_correction({}, np.eye(4))
        assert result == {}

    def test_dimension_mismatch_raises(self) -> None:
        """Correction matrix size must match qubit count."""
        counts = {"000": 500, "111": 500}
        correction = np.eye(4)
        with pytest.raises(ValueError, match="does not match"):
            apply_spam_correction(counts, correction)

    def test_zero_count_entries_filtered(self) -> None:
        """Entries with zero counts are not in output."""
        correction = np.eye(2)
        counts = {"0": 1000}
        result = apply_spam_correction(counts, correction)
        assert "1" not in result
        assert result == {"0": 1000}


class TestSpamCorrectionExports:
    """Tests for package-level exports."""

    def test_import_from_package(self) -> None:
        """New functions importable from marqov.benchmarking."""
        from marqov.benchmarking import apply_spam_correction, build_correction_matrix

        assert callable(build_correction_matrix)
        assert callable(apply_spam_correction)
