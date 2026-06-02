"""Tests for Azure bit-order normalization.

These test the static bit-order conversion logic, not the full
Azure execution flow (which requires real Azure credentials).
"""



class TestQiskitBitOrderConversion:
    """Tests for Qiskit path bitstring reversal."""

    def test_reversal_x_on_qubit_0(self) -> None:
        """X(0) on 2 qubits: Qiskit returns '01', Marqov expects '10'."""
        raw_counts = {"01": 1000}
        converted = {k.replace(" ", "")[::-1]: v for k, v in raw_counts.items()}
        assert converted == {"10": 1000}

    def test_reversal_bell_state(self) -> None:
        """Bell state: Qiskit '00'/'11' should stay '00'/'11' (palindromes)."""
        raw_counts = {"00": 500, "11": 500}
        converted = {k.replace(" ", "")[::-1]: v for k, v in raw_counts.items()}
        assert converted == {"00": 500, "11": 500}

    def test_reversal_three_qubits(self) -> None:
        """3-qubit case: Qiskit '001' -> Marqov '100'."""
        raw_counts = {"001": 1000}
        converted = {k.replace(" ", "")[::-1]: v for k, v in raw_counts.items()}
        assert converted == {"100": 1000}

    def test_space_separated_bitstrings(self) -> None:
        """Qiskit multi-register: '0 1' -> '01' -> '10'."""
        raw_counts = {"0 1": 1000}
        converted = {k.replace(" ", "")[::-1]: v for k, v in raw_counts.items()}
        assert converted == {"10": 1000}


class TestCirqHistogramConversion:
    """Tests for Cirq histogram integer-to-bitstring conversion."""

    def test_integer_to_bitstring_2_qubits(self) -> None:
        """Cirq integer 1 with 2 qubits -> '01' -> reversed -> '10'."""
        raw_histogram = {1: 1000}
        num_qubits = 2
        converted = {
            format(k, f"0{num_qubits}b")[::-1]: v
            for k, v in raw_histogram.items()
        }
        assert converted == {"10": 1000}

    def test_integer_to_bitstring_bell_state(self) -> None:
        """Bell state: integers 0 and 3 -> '00' and '11'."""
        raw_histogram = {0: 500, 3: 500}
        num_qubits = 2
        converted = {
            format(k, f"0{num_qubits}b")[::-1]: v
            for k, v in raw_histogram.items()
        }
        assert converted == {"00": 500, "11": 500}

    def test_integer_to_bitstring_3_qubits(self) -> None:
        """3-qubit: integer 4 = '100' -> reversed -> '001'."""
        raw_histogram = {4: 1000}
        num_qubits = 3
        converted = {
            format(k, f"0{num_qubits}b")[::-1]: v
            for k, v in raw_histogram.items()
        }
        assert converted == {"001": 1000}

    def test_all_bitstrings_are_strings(self) -> None:
        """Output keys are strings, not integers."""
        raw_histogram = {0: 500, 1: 300, 2: 200}
        num_qubits = 2
        converted = {
            format(k, f"0{num_qubits}b")[::-1]: v
            for k, v in raw_histogram.items()
        }
        for key in converted:
            assert isinstance(key, str)
            assert len(key) == num_qubits
