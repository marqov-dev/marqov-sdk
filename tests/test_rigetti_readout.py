"""Actual pyQuil PyQVM sparse-wire execution and strict readout validation, offline."""

from types import SimpleNamespace

import numpy as np
import pytest
from pyquil.pyqvm import PyQVM

from marqov import Circuit
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig


def result(readout):
    return SimpleNamespace(get_register_map=lambda: {"ro": readout})


@pytest.mark.asyncio
@pytest.mark.parametrize("sparse", [False, True])
async def test_actual_pyqvm_measures_the_excited_physical_wire(sparse):
    q = 2 if sparse else 0
    qvm = PyQVM(n_qubits=3, seed=17)
    compiled = []

    def compile(program):
        compiled.append(program)
        return program

    qc = SimpleNamespace(qam=qvm, compile=compile, run=qvm.run)
    ex = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="3q-qvm"), qc=qc)
    out = await ex.execute(Circuit().x(q), shots=16)
    assert out.counts == {"1": 16}
    assert out.metadata["measured_qubits"] == [q]
    assert f"MEASURE {q} ro[0]" in compiled[0].out()
    assert out.metadata["num_qubits"] == 1  # Active-wire width remains unchanged


@pytest.mark.asyncio
async def test_actual_pyqvm_sparse_entanglement_preserves_asymmetric_counts():
    qvm = PyQVM(n_qubits=4, seed=17)
    qc = SimpleNamespace(qam=qvm, compile=lambda p: p, run=qvm.run)
    out = await RigettiExecutor(
        RigettiExecutorConfig(quantum_processor_id="4q-qvm"), qc=qc
    ).execute(Circuit().h(3).cnot(3, 1).x(0), shots=32)
    assert out.metadata["measured_qubits"] == [0, 1, 3]
    assert set(out.counts) == {"100", "111"}
    assert sum(out.counts.values()) == 32
    assert out.raw_result.get_register_map()["ro"].shape == (32, 3)


@pytest.mark.parametrize(
    "readout",
    [
        None,
        [],
        [[1, 0]],
        [[1, 0]] * 3,
        [[1], [0]],
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0], [0]],
        [[2, 0], [0, 1]],
        [[-1, 0], [0, 1]],
        [[1.5, 0], [0, 1]],
        [[1.0, 0.0], [0.0, 1.0]],
        [["1", "0"], ["0", "1"]],
        [[float("nan"), 0], [0, 1]],
    ],
)
def test_partial_malformed_or_coerced_readout_rejected(readout):
    with pytest.raises(ValueError):
        RigettiExecutor._result_to_counts(result(readout), 2, 2)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, np.bool_])
def test_vendor_integer_and_boolean_arrays_accepted(dtype):
    assert RigettiExecutor._result_to_counts(
        result(np.array([[1, 0], [0, 1]], dtype=dtype)), 2, 2
    ) == {"10": 1, "01": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("shots", [0, -1, True, np.bool_(True), 1.5, float("nan")])
async def test_invalid_shots_refused_before_backend(shots):
    def unexpected(program):
        raise AssertionError("backend contacted")

    qc = SimpleNamespace(compile=unexpected, run=unexpected)
    with pytest.raises(ValueError, match="positive integer"):
        await RigettiExecutor(RigettiExecutorConfig(), qc=qc).execute(Circuit().x(0), shots=shots)


@pytest.mark.asyncio
async def test_partial_execution_is_error_with_recovery_context():
    qc = SimpleNamespace(compile=lambda p: p, run=lambda p: result([[1, 0], [1, 0]]))
    ex = RigettiExecutor(RigettiExecutorConfig(), qc=qc)
    with pytest.raises(RuntimeError, match="requested") as caught:
        await ex.execute(Circuit().x(0).z(1), shots=4)
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.remote_job["provider"] == "rigetti"


@pytest.mark.asyncio
async def test_numpy_integer_shots_preserve_exact_readout_count():
    qvm = PyQVM(n_qubits=1, seed=17)
    qc = SimpleNamespace(qam=qvm, compile=lambda p: p, run=qvm.run)
    out = await RigettiExecutor(RigettiExecutorConfig(), qc=qc).execute(
        Circuit().x(0), shots=np.int64(4)
    )
    assert out.shots == 4 and type(out.shots) is int
    assert out.counts == {"1": 4}
