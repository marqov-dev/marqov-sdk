"""Cross-executor conformance tests for measurement-result normalization.

Every executor must return results in ONE shape, regardless of vendor:

1. **Bit order** — qubit 0 is the LEFTMOST character of the bitstring.
   This convention is stated in ``local.py``, ``rigetti.py``, ``ionq.py`` and
   ``azure.py``; these tests pin it for every executor.

2. **Shot conservation** — ``sum(counts.values()) == shots``. Downstream code
   (expectation values, fidelity, SPAM correction) divides by the total and
   assumes it equals the requested shot count.

These tests call the executors' real normalization code. They deliberately do
NOT re-implement the conversion inside the test — a test that reproduces the
implementation passes even when the implementation is deleted.

The canonical probe is an asymmetric 2-qubit outcome: X applied to qubit 0
only, which must normalize to "10". Symmetric states (Bell, GHZ) are
palindromes and cannot detect a reversal.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# Canonical probe: X on qubit 0 of a 2-qubit register.
QUBIT0_EXCITED = "10"


class TestAllocateCounts:
    """The shared largest-remainder allocator."""

    def test_conserves_shots_when_probabilities_sum_above_one(self) -> None:
        """An unnormalized histogram must still allocate exactly `shots`.

        Vendors do not always hand back a normalized histogram. Over-allocation
        must be reclaimed fully, or the allocator reintroduces the very
        "total != shots" defect it exists to prevent.
        """
        from marqov.executors._counts import allocate_counts

        counts = allocate_counts({"00": 0.5, "01": 0.5, "10": 0.5, "11": 0.5}, shots=1000)

        assert sum(counts.values()) == 1000

    def test_conserves_shots_for_single_bin_above_one(self) -> None:
        """A single bin with probability > 1 must not over-allocate."""
        from marqov.executors._counts import allocate_counts

        counts = allocate_counts({"a": 2.0}, shots=10)

        assert sum(counts.values()) == 10


class TestIBMBitOrder:
    """IBMExecutor._extract_counts against a real Qiskit SamplerV2 result."""

    @staticmethod
    def _real_sampler_result() -> Any:
        """Produce a genuine Qiskit PrimitiveResult for X(0) on 2 qubits.

        Uses Qiskit's own local StatevectorSampler so the DataBin/BitArray
        shapes are real, not hand-mocked.
        """
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.measure_all()
        return StatevectorSampler().run([qc], shots=100).result()

    def test_extract_counts_places_qubit0_leftmost(self) -> None:
        """X(0) must normalize to '10' (qubit 0 leftmost).

        Qiskit reports this as '01' (qubit 0 rightmost). The executor is
        responsible for converting to the Marqov convention, exactly as
        AzureQuantumExecutor does for the same framework.
        """
        from marqov.executors.ibm import IBMExecutor

        result = self._real_sampler_result()
        counts = IBMExecutor._extract_counts(result)

        assert counts == {QUBIT0_EXCITED: 100}

    def test_multi_register_result_does_not_silently_drop_a_register(self) -> None:
        """Two classical registers must not yield a 1-bit string for 2 qubits.

        Selecting the first BitArray and ignoring the rest returns counts
        narrower than the measurement, which mis-indexes every downstream
        consumer (fidelity, SPAM, expectation values). Failing loudly is the
        safe behaviour; silently returning half the data is not.
        """
        from qiskit import ClassicalRegister, QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        from marqov.executors.ibm import IBMExecutor

        reg_a, reg_b = ClassicalRegister(1, "a"), ClassicalRegister(1, "b")
        qc = QuantumCircuit(2)
        qc.add_register(reg_a)
        qc.add_register(reg_b)
        qc.x(0)
        qc.measure(0, reg_a[0])
        qc.measure(1, reg_b[0])
        result = StatevectorSampler().run([qc], shots=100).result()

        with pytest.raises(NotImplementedError, match="multiple classical registers"):
            IBMExecutor._extract_counts(result)


class TestRigettiBitOrder:
    """RigettiExecutor._result_to_counts."""

    def test_result_to_counts_places_qubit0_leftmost(self) -> None:
        """Readout row [1, 0] (qubit 0 measured 1) must normalize to '10'."""
        from marqov.executors.rigetti import RigettiExecutor

        class _FakeResult:
            def get_register_map(self) -> dict[str, list[list[int]]]:
                return {"ro": [[1, 0]] * 100}

        counts = RigettiExecutor._result_to_counts(_FakeResult(), num_qubits=2)

        assert counts == {QUBIT0_EXCITED: 100}


class TestIonQBitOrder:
    """IonQExecutor._histogram_to_counts."""

    def test_histogram_to_counts_places_qubit0_leftmost(self) -> None:
        """State index 2 (binary '10') must normalize to '10'."""
        from marqov.executors.ionq import IonQExecutor

        counts = IonQExecutor._histogram_to_counts({"2": 1.0}, shots=100, num_qubits=2)

        assert counts == {QUBIT0_EXCITED: 100}

    def test_counts_sum_to_shots_for_non_terminating_probabilities(self) -> None:
        """Thirds must still allocate exactly `shots` counts."""
        from marqov.executors.ionq import IonQExecutor

        third = 1.0 / 3.0
        counts = IonQExecutor._histogram_to_counts(
            {"0": third, "1": third, "2": third}, shots=1000, num_qubits=2
        )

        assert sum(counts.values()) == 1000

    def test_counts_sum_to_shots_when_probabilities_sum_above_one(self) -> None:
        """IonQ must conserve shots on an unnormalized histogram too."""
        from marqov.executors.ionq import IonQExecutor

        counts = IonQExecutor._histogram_to_counts(
            {"0": 0.5, "1": 0.5, "2": 0.5, "3": 0.5}, shots=1000, num_qubits=2
        )

        assert sum(counts.values()) == 1000


class TestBraketShotConservation:
    """BraketExecutor's probability fallback, used when a QPU (e.g. IonQ
    Forte-1) returns measurementProbabilities instead of raw counts."""

    def test_probability_fallback_counts_sum_to_shots(self) -> None:
        """Thirds must still allocate exactly `shots` counts."""
        from marqov.circuits import Circuit
        from marqov.executors.braket import BraketExecutor, BraketExecutorConfig

        third = 1.0 / 3.0

        class _FakeResult:
            measurement_counts: dict[str, int] = {}
            measurement_probabilities = {"00": third, "01": third, "10": third}

        class _FakeTask:
            id = "arn:aws:braket:us-east-1:000000000000:quantum-task/fake"

            def result(self) -> Any:
                return _FakeResult()

            def metadata(self) -> dict[str, Any]:
                return {"executionDuration": 1}

        class _FakeDevice:
            name = "FakeDevice"

            def run(self, *args: Any, **kwargs: Any) -> Any:
                return _FakeTask()

        executor = BraketExecutor(
            BraketExecutorConfig(device_arn="arn:aws:braket:us-east-1::device/qpu/fake/Fake",
                                 s3_bucket="bucket")
        )
        # Bypass the AWS session/device lookup; we are testing normalization only.
        executor._get_device = lambda: asyncio.sleep(0, result=_FakeDevice())  # type: ignore[method-assign]

        circuit = Circuit()
        circuit.h(0)
        result = asyncio.run(executor.execute(circuit, shots=1000))

        assert sum(result.counts.values()) == 1000

    def test_probability_fallback_preserves_braket_bit_order(self) -> None:
        """Braket already reports qubit 0 leftmost; the fallback must not reorder."""
        from marqov.circuits import Circuit
        from marqov.executors.braket import BraketExecutor, BraketExecutorConfig

        class _FakeResult:
            measurement_counts: dict[str, int] = {}
            measurement_probabilities = {QUBIT0_EXCITED: 1.0}

        class _FakeTask:
            id = "arn:aws:braket:us-east-1:000000000000:quantum-task/fake"

            def result(self) -> Any:
                return _FakeResult()

            def metadata(self) -> dict[str, Any]:
                return {"executionDuration": 1}

        class _FakeDevice:
            name = "FakeDevice"

            def run(self, *args: Any, **kwargs: Any) -> Any:
                return _FakeTask()

        executor = BraketExecutor(
            BraketExecutorConfig(device_arn="arn:aws:braket:us-east-1::device/qpu/fake/Fake",
                                 s3_bucket="bucket")
        )
        executor._get_device = lambda: asyncio.sleep(0, result=_FakeDevice())  # type: ignore[method-assign]

        circuit = Circuit()
        circuit.h(0)
        result = asyncio.run(executor.execute(circuit, shots=100))

        assert result.counts == {QUBIT0_EXCITED: 100}
