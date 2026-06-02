"""Tests for single-qubit Clifford group and RB sequence generation."""

from __future__ import annotations

import numpy as np
import pytest

from marqov.qmt.characterization.clifford import (
    CLIFFORD_GROUP,
    clifford_to_circuit,
    clifford_to_matrix,
    compose_clifford_indices,
    generate_rb_sequence,
    inverse_clifford_index,
)


class TestCliffordGroup:
    def test_has_24_elements(self) -> None:
        assert len(CLIFFORD_GROUP) == 24

    def test_identity_is_first(self) -> None:
        assert CLIFFORD_GROUP[0] == []

    def test_all_elements_are_unitary(self) -> None:
        for i, gates in enumerate(CLIFFORD_GROUP):
            mat = clifford_to_matrix(i)
            product = mat @ mat.conj().T
            np.testing.assert_array_almost_equal(
                product,
                np.eye(2),
                decimal=10,
                err_msg=f"Clifford {i} ({gates}) is not unitary",
            )

    def test_closure(self) -> None:
        for i in range(24):
            for j in range(24):
                k = compose_clifford_indices(i, j)
                assert 0 <= k < 24, f"Compose({i}, {j}) = {k} out of range"

    def test_inverse(self) -> None:
        for i in range(24):
            inv = inverse_clifford_index(i)
            composed = compose_clifford_indices(i, inv)
            assert composed == 0, (
                f"Clifford {i} inverse is {inv}, but compose = {composed}"
            )


class TestRBSequence:
    def test_sequence_length(self) -> None:
        seq = generate_rb_sequence(length=10, seed=42)
        assert len(seq) == 11

    def test_sequence_composes_to_identity(self) -> None:
        seq = generate_rb_sequence(length=20, seed=42)
        result = 0
        for idx in seq:
            result = compose_clifford_indices(result, idx)
        assert result == 0

    def test_seed_reproducibility(self) -> None:
        s1 = generate_rb_sequence(length=10, seed=99)
        s2 = generate_rb_sequence(length=10, seed=99)
        assert s1 == s2

    def test_different_seeds_differ(self) -> None:
        s1 = generate_rb_sequence(length=10, seed=1)
        s2 = generate_rb_sequence(length=10, seed=2)
        assert s1 != s2

    def test_excludes_identity_from_random(self) -> None:
        """Random Cliffords should never be identity (element 0)."""
        for seed in range(100):
            seq = generate_rb_sequence(length=5, seed=seed)
            # The random portion (all but last) should never be 0
            for idx in seq[:-1]:
                assert idx != 0, f"Identity sampled in random portion (seed={seed})"

    def test_always_produces_gates(self) -> None:
        """Every sequence should have at least one physical gate."""
        for seed in range(100):
            seq = generate_rb_sequence(length=1, seed=seed)
            total_gates = sum(len(CLIFFORD_GROUP[idx]) for idx in seq)
            assert total_gates > 0, f"Zero gates for seed={seed}: {seq}"


class TestCliffordToCircuit:
    def test_produces_circuit(self) -> None:
        from marqov.circuits import Circuit

        seq = generate_rb_sequence(length=5, seed=42)
        circuit = clifford_to_circuit(seq, qubit=3)
        assert isinstance(circuit, Circuit)

    def test_targets_correct_qubit(self) -> None:
        seq = [1]
        circuit = clifford_to_circuit(seq, qubit=7)
        assert circuit is not None
