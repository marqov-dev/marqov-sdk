"""Offline provider-shaped observations through the real Braket executor."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from braket.tasks import GateModelQuantumTaskResult

from marqov.circuits import Circuit
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig


def provider_result(*, probabilities=False, reported_shots=4):
    payload = {
        "braketSchemaHeader": {"name": "braket.task_result.gate_model_task_result", "version": "1"},
        "taskMetadata": {
            "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
            "id": "arn:aws:braket:us-east-1:123456789012:quantum-task/offline",
            "deviceId": "offline", "shots": reported_shots,
        },
        "additionalMetadata": {"action": {
            "braketSchemaHeader": {"name": "braket.ir.jaqcd.program", "version": "1"},
            "instructions": [],
        }},
        "measuredQubits": [7, 1],
    }
    if probabilities:
        payload["measurementProbabilities"] = {"10": 0.5, "01": 0.5}
    else:
        payload["measurements"] = [[1, 0], [1, 0], [0, 1], [1, 1]]
    return GateModelQuantumTaskResult.from_string(json.dumps(payload))


async def execute_result(result, shots=4):
    task = SimpleNamespace(
        id="arn:aws:braket:us-east-1:123456789012:quantum-task/offline",
        result=lambda: result, metadata=lambda: {"shots": shots},
    )
    submissions = []

    def run(circuit, **options):
        submissions.append(options)
        return task

    device = SimpleNamespace(name="offline", run=run)
    executor = BraketExecutor(BraketExecutorConfig(device_arn="offline", s3_bucket="unused"))
    executor._get_device = AsyncMock(return_value=device)
    outcome = await executor.execute(Circuit().h(0).cnot(0, 1), shots=shots)
    assert submissions[0]["shots"] == shots
    assert outcome.raw_result is result
    assert outcome.shots == shots  # Legacy field remains requested shots.
    evidence = outcome.metadata["measurement_provenance"]
    json.dumps(evidence, allow_nan=False)
    return outcome, evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("requested, reported", [(4, 4), (10, 10)])
async def test_actual_measurement_rows_and_partial_returns(requested, reported):
    provider = provider_result(reported_shots=reported)
    original = provider.measurements.copy()
    outcome, evidence = await execute_result(provider, requested)
    assert outcome.counts == {"10": 2, "01": 1, "11": 1}
    assert evidence == {
        "protocol_version": "marqov.braket-measurement-provenance/v1",
        "count_origin": "provider_measurements",
        "requested_shots": requested, "observed_shots": 4,
        "provider_reported_shots": reported,
        "measured_qubits": [7, 1],
        "bitstring_order": "measured_qubits_left_to_right",
        "raw_shot_eligible": True, "ineligibility_reason": None,
        "physical_mapping_status": "unqualified",
    }
    np.testing.assert_array_equal(provider.measurements, original)
    assert provider.measurements_copied_from_device is True
    assert provider.measurement_counts_copied_from_device is False


@pytest.mark.asyncio
async def test_braket_synthesized_nonempty_counts_are_not_raw_shots():
    provider = provider_result(probabilities=True)
    assert provider.measurement_counts  # Already synthesized inside Braket.
    outcome, evidence = await execute_result(provider)
    assert outcome.counts == dict(provider.measurement_counts)
    assert evidence["count_origin"] == "probability_derived"
    assert evidence["observed_shots"] is None
    assert evidence["raw_shot_eligible"] is False
    assert evidence["ineligibility_reason"] == "probability_derived"


@pytest.mark.asyncio
async def test_sdk_probability_fallback_is_labeled():
    provider = SimpleNamespace(measurement_counts={}, measurement_probabilities={"10": 0.5, "01": 0.5})
    outcome, evidence = await execute_result(provider)
    assert outcome.counts == {"10": 2, "01": 2}
    assert evidence["count_origin"] == "probability_derived"
    assert evidence["observed_shots"] is None
    assert evidence["raw_shot_eligible"] is False


@pytest.mark.asyncio
async def test_unknown_origin_is_explicit_even_with_counts_and_order():
    provider = SimpleNamespace(measurement_counts={"10": 4}, measured_qubits=[7, 1])
    outcome, evidence = await execute_result(provider)
    assert outcome.counts == {"10": 4}
    assert evidence["count_origin"] == "unknown"
    assert evidence["observed_shots"] is None
    assert evidence["ineligibility_reason"] == "unknown_origin"
    assert evidence["raw_shot_eligible"] is False


@pytest.mark.asyncio
async def test_explicit_provider_counts_preserve_order_and_total():
    provider = SimpleNamespace(
        measurement_counts={"10": 3}, measured_qubits=[7, 1],
        measurement_counts_copied_from_device=True,
    )
    _, evidence = await execute_result(provider)
    assert evidence["count_origin"] == "provider_counts"
    assert evidence["observed_shots"] == 3
    assert evidence["measured_qubits"] == [7, 1]
    assert evidence["raw_shot_eligible"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("attribute, value, reason", [
    ("measured_qubits", None, "missing_measured_qubits"),
    ("measured_qubits", [7, 7], "invalid_measured_qubits"),
    ("measured_qubits", [7], "invalid_measured_qubits"),
    ("measurement_counts", {"10": 4}, "counts_measurements_mismatch"),
    ("measurement_counts", {"10": -1}, "invalid_counts"),
    ("measurement_counts", {"10": True}, "invalid_counts"),
    ("measurements", np.array([[2, 0]]), "invalid_measurements"),
    ("measurements", np.array([[1.0, 0.0]]), "invalid_measurements"),
])
async def test_malformed_evidence_is_ineligible_without_losing_result(attribute, value, reason):
    provider = provider_result()
    setattr(provider, attribute, value)
    outcome, evidence = await execute_result(provider)
    assert outcome.raw_result is provider
    assert evidence["raw_shot_eligible"] is False
    assert evidence["ineligibility_reason"] == reason


@pytest.mark.asyncio
async def test_excess_observed_shots_is_not_eligible():
    _, evidence = await execute_result(provider_result(), shots=2)
    assert evidence["observed_shots"] == 4
    assert evidence["ineligibility_reason"] == "excess_observed_shots"
    assert evidence["raw_shot_eligible"] is False


@pytest.mark.asyncio
async def test_zero_observations_are_explicit_but_not_raw_shot_eligible():
    provider = provider_result()
    provider.measurements = np.empty((0, 2), dtype=int)
    provider.measurement_counts = {}
    provider.measurement_probabilities = {}
    _, evidence = await execute_result(provider)
    assert evidence["observed_shots"] == 0
    assert evidence["raw_shot_eligible"] is False
    assert evidence["ineligibility_reason"] == "no_observed_shots"


@pytest.mark.asyncio
async def test_optional_evidence_failure_preserves_completed_counts():
    class BrokenEvidence:
        measurement_counts = {"10": 4}

        @property
        def measured_qubits(self):
            raise RuntimeError("optional evidence failed")

    provider = BrokenEvidence()
    outcome, evidence = await execute_result(provider)
    assert outcome.counts == {"10": 4}
    assert evidence["raw_shot_eligible"] is False
    assert evidence["ineligibility_reason"] == "invalid_evidence"
