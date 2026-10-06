"""Offline native-label export and submission parity using real Braket objects."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import quantumflow as qf
from braket.circuits import Circuit as BraketCircuit

from marqov.circuits import Circuit
from marqov.device import MarqovDevice
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig

ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
PHYSICAL = dict(preserve_qubit_labels=True, verbatim=True, disable_qubit_rewiring=True)


def native():
    return Circuit.from_braket(BraketCircuit().rx(7, 0.3).rz(1, -0.7).cz(7, 1).xy(1, 7, 0.4))


def records(circuit):
    return [
        (i.operator.name, list(map(int, i.target)), getattr(i.operator, "angle", None))
        for i in circuit.instructions
    ]


def providers(arn=ARN, backend="rigetti"):
    task = SimpleNamespace(
        id="offline-task",
        metadata=lambda: {},
        result=lambda: SimpleNamespace(
            measurement_counts={"10": 4},
            measured_qubits=[7, 1],
            measurement_counts_copied_from_device=True,
        ),
    )
    provider = SimpleNamespace(name="offline", run=Mock(return_value=task))
    executor = BraketExecutor(
        BraketExecutorConfig(device_arn=arn, s3_bucket="unused", s3_prefix="test")
    )
    executor._get_device = AsyncMock(return_value=provider)
    device = MarqovDevice(backend, {"device_arn": arn, "s3_destination_folder": ("unused", "test")})
    device._get_provider_device = Mock(return_value=provider)
    return executor, device, provider


def test_native_export_preserves_sparse_order_and_angles():
    source = native()
    before = source.to_dict()
    assert records(source.to_braket_native()) == [
        ("Rx", [7], 0.3),
        ("Rz", [1], -0.7),
        ("CZ", [7, 1], None),
        ("XY", [1, 7], pytest.approx(0.4, abs=1e-15)),
    ]
    assert source.to_dict() == before
    assert [r[1] for r in records(source.to_braket())] == [[1], [0], [1, 0], [0, 1]]


def test_native_export_matches_complex_operators():
    source = native()
    exported = source.to_braket_native()
    # Independently compare each original native operator to Braket's matrix,
    # preserving the order of its targets rather than comparing probabilities.
    for qf_gate, instruction in zip(source._qf, exported.instructions, strict=True):
        np.testing.assert_allclose(
            qf_gate.asoperator(), instruction.operator.to_matrix(), atol=1e-14
        )


@pytest.mark.parametrize(
    "gate",
    [
        qf.H(7),
        qf.X(7),
        qf.CNot(7, 1),
        qf.Rx(math.inf, 7),
        qf.Rz(float("nan"), 1),
        qf.XY(1e308, 1, 7),
        qf.Rx(10**1000, 7),
        qf.Rx(0.2, -1),
        qf.Rx(0.2, "7"),
        qf.Rx(0.2, True),
    ],
)
def test_native_export_refuses_invalid_gates(gate):
    source = Circuit()
    source._qf += gate
    with pytest.raises(ValueError):
        source.to_braket_native()


def test_native_export_refuses_empty_and_symbolic():
    import sympy

    with pytest.raises(ValueError, match="nonempty"):
        Circuit().to_braket_native()
    source = Circuit()
    source._qf += qf.Rx(sympy.Symbol("theta"), 7)
    with pytest.raises(ValueError, match="finite real"):
        source.to_braket_native()


@pytest.mark.asyncio
async def test_physical_submission_parity_and_provenance_stays_unqualified():
    executor, device, provider = providers()
    outcome = await executor.execute(native(), shots=4, **PHYSICAL)
    executor_call = provider.run.call_args
    provider.run.reset_mock()
    assert device.run(native(), shots=4, **PHYSICAL) == {"10": 4}
    device_call = provider.run.call_args
    assert (
        records(executor_call.args[0])
        == records(device_call.args[0])
        == [
            ("StartVerbatimBox", [], None),
            ("Rx", [7], 0.3),
            ("Rz", [1], -0.7),
            ("CZ", [7, 1], None),
            ("XY", [1, 7], pytest.approx(0.4, abs=1e-15)),
            ("EndVerbatimBox", [], None),
        ]
    )
    assert executor_call.kwargs["disable_qubit_rewiring"] is True
    assert device_call.kwargs["disable_qubit_rewiring"] is True
    assert executor_call.kwargs["shots"] == device_call.kwargs["shots"] == 4
    assert executor_call.kwargs["s3_destination_folder"] == device_call.args[1]
    assert outcome.metadata["measurement_provenance"]["physical_mapping_status"] == "unqualified"


@pytest.mark.asyncio
@pytest.mark.parametrize("rewiring", [None, False, True])
async def test_default_logical_submission_and_explicit_rewiring(rewiring):
    executor, device, provider = providers()
    options = {} if rewiring is None else {"disable_qubit_rewiring": rewiring}
    await executor.execute(native(), shots=4, **options)
    executor_call = provider.run.call_args
    provider.run.reset_mock()
    device.run(native(), shots=4, **options)
    device_call = provider.run.call_args
    assert records(executor_call.args[0]) == records(device_call.args[0])
    assert [r[1] for r in records(executor_call.args[0])] == [[1], [0], [1, 0], [0, 1]]
    if rewiring is None:
        assert "disable_qubit_rewiring" not in executor_call.kwargs
        assert device_call.kwargs["disable_qubit_rewiring"] is False
    else:
        assert executor_call.kwargs["disable_qubit_rewiring"] is rewiring
        assert device_call.kwargs["disable_qubit_rewiring"] is rewiring


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"preserve_qubit_labels": True},
        {"preserve_qubit_labels": True, "verbatim": True},
        {"preserve_qubit_labels": True, "disable_qubit_rewiring": True},
        {**PHYSICAL, "verbatim": False},
        {**PHYSICAL, "disable_qubit_rewiring": False},
        {**PHYSICAL, "preserve_qubit_labels": 1},
        {**PHYSICAL, "verbatim": "true"},
        {**PHYSICAL, "disable_qubit_rewiring": np.bool_(True)},
    ],
)
async def test_conflicting_options_refused_before_provider_initialization(options):
    executor, device, provider = providers()
    with pytest.raises(ValueError):
        await executor.execute(native(), **options)
    with pytest.raises(ValueError):
        device.run(native(), **options)
    executor._get_device.assert_not_called()
    device._get_provider_device.assert_not_called()
    provider.run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arn",
    [
        "offline",
        "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
        "arn:aws:braket:us-west-1::device/qpu/rigetti/",
        "arn:aws:braket:us-west-1::device/qpu/rigetti/   ",
        "arn:aws:braket:invalid:123::device/qpu/rigetti/Fake",
    ],
)
async def test_non_rigetti_targets_refused(arn):
    executor, device, provider = providers(arn)
    with pytest.raises(ValueError, match="Rigetti QPU"):
        await executor.execute(native(), **PHYSICAL)
    with pytest.raises(ValueError, match="Rigetti QPU"):
        device.run(native(), **PHYSICAL)
    provider.run.assert_not_called()
    executor._get_device.assert_not_called()
    device._get_provider_device.assert_not_called()


@pytest.mark.parametrize(
    "backend,params",
    [
        ("local", {}),
        ("marqov-sim", {"device_arn": ARN}),
        ("ibm", {"ibm_channel": "ibm_cloud", "device_arn": ARN}),
        ("azure", {"azure_subscription_id": "offline", "device_arn": ARN}),
    ],
)
def test_non_braket_paths_cannot_ignore_physical_intent(backend, params):
    device = MarqovDevice(backend, params)
    device._get_provider_device = Mock()
    with pytest.raises(ValueError, match="Rigetti QPU"):
        device.run(native(), **PHYSICAL)
    device._get_provider_device.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [Circuit().h(7), Circuit().rx(10**1000, 7)])
async def test_unsupported_native_gate_refused_before_submit(source):
    executor, device, provider = providers()
    with pytest.raises(ValueError):
        await executor.execute(source, **PHYSICAL)
    with pytest.raises(ValueError):
        device.run(source, **PHYSICAL)
    provider.run.assert_not_called()
    executor._get_device.assert_not_called()
    device._get_provider_device.assert_not_called()
