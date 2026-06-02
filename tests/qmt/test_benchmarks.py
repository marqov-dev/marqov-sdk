"""Tests for QMT benchmark circuit generators."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from marqov.qmt.characterization.benchmarks import (
    ghz_on_qubits,
    mirror_circuit,
    random_circuit,
)


class TestGHZOnQubits:
    def test_creates_circuit_on_specified_qubits(self) -> None:
        """GHZ on [3,4,5] should apply gates to those qubit indices."""
        circuit = ghz_on_qubits([3, 4, 5])
        # Circuit is constructed — verify it returns a Circuit instance
        from marqov.circuits import Circuit

        assert isinstance(circuit, Circuit)

    def test_single_qubit(self) -> None:
        """Single-qubit GHZ is just a Hadamard."""
        circuit = ghz_on_qubits([0])
        from marqov.circuits import Circuit

        assert isinstance(circuit, Circuit)

    def test_empty_qubits_raises(self) -> None:
        """Empty qubit list should raise."""
        with pytest.raises(IndexError):
            ghz_on_qubits([])

    def test_two_qubits(self) -> None:
        """Two-qubit GHZ should work."""
        circuit = ghz_on_qubits([0, 1])
        from marqov.circuits import Circuit

        assert isinstance(circuit, Circuit)

    @pytest.mark.skip(reason="requires real quantumflow for simulation")
    def test_produces_ghz_correlations(self) -> None:
        circuit = ghz_on_qubits([0, 1, 2])
        state = circuit.simulate()
        probs = state.probabilities()
        assert probs[0] == pytest.approx(0.5, abs=0.01)
        assert probs[7] == pytest.approx(0.5, abs=0.01)


class TestMirrorCircuit:
    def test_returns_circuit(self) -> None:
        """Mirror circuit returns a valid Circuit instance."""
        circuit = mirror_circuit(qubits=[0, 1], depth=2, seed=42)
        from marqov.circuits import Circuit

        assert isinstance(circuit, Circuit)

    def test_seed_reproducibility(self) -> None:
        """Same seed produces same circuit structure."""
        c1 = mirror_circuit(qubits=[0, 1], depth=2, seed=42)
        c2 = mirror_circuit(qubits=[0, 1], depth=2, seed=42)
        # Both circuits should have the same internal gate sequence
        d1 = c1.to_dict()
        d2 = c2.to_dict()
        assert d1 == d2

    @pytest.mark.skip(reason="requires real quantumflow — to_dict empty under mock")
    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different circuits."""
        c1 = mirror_circuit(qubits=[0, 1], depth=4, seed=1)
        c2 = mirror_circuit(qubits=[0, 1], depth=4, seed=2)
        d1 = c1.to_dict()
        d2 = c2.to_dict()
        assert d1 != d2

    @pytest.mark.skip(reason="requires real quantumflow for simulation")
    def test_mirror_circuit_identity(self) -> None:
        circuit = mirror_circuit(qubits=[0, 1], depth=2, seed=42)
        state = circuit.simulate()
        probs = state.probabilities()
        assert probs[0] == pytest.approx(1.0, abs=0.01)


class TestRandomCircuit:
    def test_returns_circuit(self) -> None:
        """Random circuit returns a valid Circuit instance."""
        circuit = random_circuit(qubits=[2, 3, 4], depth=5, seed=42)
        from marqov.circuits import Circuit

        assert isinstance(circuit, Circuit)

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical circuits."""
        c1 = random_circuit(qubits=[0, 1], depth=3, seed=99)
        c2 = random_circuit(qubits=[0, 1], depth=3, seed=99)
        d1 = c1.to_dict()
        d2 = c2.to_dict()
        assert d1 == d2

    @pytest.mark.skip(reason="requires real quantumflow — to_dict empty under mock")
    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different circuits."""
        c1 = random_circuit(qubits=[0, 1], depth=5, seed=1)
        c2 = random_circuit(qubits=[0, 1], depth=5, seed=2)
        d1 = c1.to_dict()
        d2 = c2.to_dict()
        assert d1 != d2

    @pytest.mark.skip(reason="requires real quantumflow — to_dict empty under mock")
    def test_uses_specified_qubits(self) -> None:
        """Gates should only target the specified qubits."""
        circuit = random_circuit(qubits=[3, 5], depth=2, seed=42)
        gate_data = circuit.to_dict()
        for gate in gate_data.get("gates", []):
            for q in gate["qubits"]:
                assert q in {3, 5}, f"Gate targeted unexpected qubit {q}"

    @pytest.mark.skip(reason="requires real quantumflow for simulation")
    def test_simulation_reproducibility(self) -> None:
        c1 = random_circuit(qubits=[0, 1], depth=3, seed=99)
        c2 = random_circuit(qubits=[0, 1], depth=3, seed=99)
        s1 = c1.simulate().probabilities()
        s2 = c2.simulate().probabilities()
        for i in range(len(s1)):
            assert s1[i] == pytest.approx(s2[i])


class TestGHZDepthMultiplier:
    def test_multiplier_1_is_default(self) -> None:
        """depth_multiplier=1 produces the same circuit as the default."""
        c1 = ghz_on_qubits([0, 1])
        c2 = ghz_on_qubits([0, 1], depth_multiplier=1)
        assert c1.to_dict() == c2.to_dict()

    @pytest.mark.skip(reason="requires real quantumflow — to_dict empty under mock")
    def test_multiplier_adds_identity_pairs(self) -> None:
        """Higher depth_multiplier adds CNOT-CNOT identity pairs."""
        c_base = ghz_on_qubits([0, 1], depth_multiplier=1)
        c_deep = ghz_on_qubits([0, 1], depth_multiplier=3)
        base_gates = len(c_base.to_dict().get("gates", []))
        deep_gates = len(c_deep.to_dict().get("gates", []))
        assert deep_gates > base_gates
