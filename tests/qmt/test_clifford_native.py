"""Tests for native-gate Clifford decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from marqov.circuits import Circuit
from marqov.qmt.characterization.clifford import (
    CLIFFORD_GROUP,
    clifford_to_circuit,
    clifford_to_circuit_native,
    clifford_to_matrix,
    generate_rb_sequence,
)


class TestCliffordNativeDecomposition:
    @pytest.mark.skip(reason="requires real braket LocalSimulator — braket is mocked in unit tests")
    def test_all_24_cliffords_match(self) -> None:
        """Every native-gate circuit must produce the same unitary as the standard decomposition."""
        for idx in range(24):
            standard = clifford_to_circuit([idx], qubit=0)
            native = clifford_to_circuit_native([idx], qubit=0)

            std_braket = standard.to_braket()
            nat_braket = native.to_braket()

            from braket.devices import LocalSimulator
            from braket.circuits import Circuit as BraketCircuit

            sim = LocalSimulator("braket_sv")

            for input_state in ["0", "1"]:
                std_test = BraketCircuit()
                nat_test = BraketCircuit()
                if input_state == "1":
                    std_test.x(0)
                    nat_test.x(0)
                for instr in std_braket.instructions:
                    std_test.add_instruction(instr)
                for instr in nat_braket.instructions:
                    nat_test.add_instruction(instr)

                std_result = sim.run(std_test, shots=1000).result()
                nat_result = sim.run(nat_test, shots=1000).result()

                std_counts = std_result.measurement_counts
                nat_counts = nat_result.measurement_counts

                std_dominant = max(std_counts, key=std_counts.get)
                nat_dominant = max(nat_counts, key=nat_counts.get)
                assert std_dominant == nat_dominant, (
                    f"Clifford {idx}: standard gives {std_counts}, "
                    f"native gives {nat_counts} for |{input_state}>"
                )

    @pytest.mark.skip(reason="requires real braket LocalSimulator — braket is mocked in unit tests")
    def test_rb_sequence_native_composes_to_identity(self) -> None:
        """A native-gate RB sequence should compose to identity (all shots return |0>)."""
        from braket.devices import LocalSimulator

        seq = generate_rb_sequence(length=8, seed=42)
        circuit = clifford_to_circuit_native(seq, qubit=0)
        braket_circ = circuit.to_braket()

        sim = LocalSimulator()
        result = sim.run(braket_circ, shots=1000).result()
        counts = dict(result.measurement_counts)

        total = sum(counts.values())
        zeros = counts.get("0", 0)
        assert zeros / total > 0.99, f"Expected |0>, got {counts}"

    def test_native_circuit_uses_only_rx_rz(self) -> None:
        """Native circuits should only contain Rx and Rz gates."""
        seq = generate_rb_sequence(length=4, seed=123)
        circuit = clifford_to_circuit_native(seq, qubit=0)
        braket_circ = circuit.to_braket()
        for instr in braket_circ.instructions:
            gate_name = instr.operator.name.lower()
            assert gate_name in ("rx", "rz"), (
                f"Unexpected gate {gate_name} in native circuit"
            )
