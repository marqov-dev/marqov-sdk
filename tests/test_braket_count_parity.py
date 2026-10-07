"""Offline count normalization through both public Braket entry points."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from marqov.circuits import Circuit
from marqov.device import MarqovDevice
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig


async def run_both(monkeypatch, result, shots):
    # Replacing both provider factories makes a real AWS initialization fail
    # immediately if either entry point stops using the injected offline device.
    no_cloud = Mock(side_effect=AssertionError("Unexpected AWS initialization"))
    monkeypatch.setattr("braket.aws.AwsDevice", no_cloud)
    monkeypatch.setattr("marqov.executors.braket.AwsDevice", no_cloud)
    task = SimpleNamespace(
        id="offline-task", result=lambda: result,
        metadata=lambda **kwargs: {},
    )
    provider = SimpleNamespace(name="offline", run=Mock(return_value=task))
    arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
    executor = BraketExecutor(
        BraketExecutorConfig(device_arn=arn, s3_bucket="unused", s3_prefix="test")
    )
    executor._get_device = AsyncMock(return_value=provider)
    device = MarqovDevice(
        "ionq", {"device_arn": arn, "s3_destination_folder": ("unused", "test")}
    )
    device._get_provider_device = Mock(return_value=provider)
    circuit = Circuit().h(0).cnot(0, 1)
    outcome = await executor.execute(circuit, shots=shots)
    counts = device.run(circuit, shots=shots)
    assert provider.run.call_count == 2
    assert all(call.kwargs["shots"] == shots for call in provider.run.call_args_list)
    no_cloud.assert_not_called()
    return counts, outcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shots, expected",
    [
        (1000, {"10": 334, "01": 333, "11": 333}),
        (7, {"10": 3, "01": 2, "11": 2}),
        (2, {"10": 1, "01": 1}),
        (1, {"10": 1}),
        (0, {}),
    ],
)
async def test_probability_fallback_conserves_shots(monkeypatch, shots, expected):
    probabilities = {"10": 1 / 3, "01": 1 / 3, "11": 1 / 3}
    result = SimpleNamespace(
        measurement_counts={}, measurement_probabilities=probabilities,
        measured_qubits=[0, 1],
    )
    counts, outcome = await run_both(monkeypatch, result, shots)
    assert counts == outcome.counts == expected
    assert sum(counts.values()) == shots
    assert all(count > 0 for count in counts.values())
    assert probabilities == {"10": 1 / 3, "01": 1 / 3, "11": 1 / 3}
    assert result.measurement_counts == {}
    assert outcome.metadata["measurement_provenance"]["count_origin"] == "probability_derived"


@pytest.mark.asyncio
async def test_provider_counts_take_precedence_without_rescaling(monkeypatch):
    original = {"10": 3, "01": 1}
    result = SimpleNamespace(
        measurement_counts=original.copy(),
        measurement_probabilities={"10": 0.5, "01": 0.5},
        measured_qubits=[0, 1], measurement_counts_copied_from_device=True,
    )
    counts, outcome = await run_both(monkeypatch, result, shots=1000)
    assert counts == outcome.counts == original
    assert result.measurement_counts == original
    assert counts is not result.measurement_counts
    assert outcome.counts is not result.measurement_counts


@pytest.mark.asyncio
async def test_empty_result_stays_empty(monkeypatch):
    result = SimpleNamespace(measurement_counts={}, measurement_probabilities={})
    counts, outcome = await run_both(monkeypatch, result, shots=1000)
    assert counts == outcome.counts == {}
