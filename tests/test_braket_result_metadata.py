"""Completed Braket results survive optional metadata failures; timing is explicit."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from botocore.exceptions import ClientError
from braket.ir.jaqcd import Program
from braket.task_result import AdditionalMetadata, SimulatorMetadata

from marqov.circuits import Circuit
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig


async def execute(*, additional_metadata=None, metadata_error=None):
    result = SimpleNamespace(measurement_counts={"0": 4}, additional_metadata=additional_metadata)
    metadata = Mock(return_value={"status": "COMPLETED", "shots": 4})
    if metadata_error is not None:
        metadata.side_effect = metadata_error
    task = SimpleNamespace(id="offline-task", result=lambda: result, metadata=metadata)
    device = SimpleNamespace(name="offline", run=Mock(return_value=task))
    executor = BraketExecutor(BraketExecutorConfig(device_arn="offline", s3_bucket="unused"))
    executor._get_device = AsyncMock(return_value=device)
    with patch("marqov.executors.braket.time.perf_counter", side_effect=[10.0, 10.25]):
        outcome = await executor.execute(Circuit().x(0), shots=4)
    assert outcome.counts == {"0": 4}
    assert outcome.raw_result is result
    metadata.assert_called_once_with(use_cached_value=True)
    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["ThrottlingException", "ExpiredTokenException", "AccessDeniedException"]
)
async def test_completed_counts_survive_metadata_client_error(code, caplog):
    error = ClientError(
        {"Error": {"Code": code, "Message": "offline optional failure"}}, "GetQuantumTask"
    )
    with caplog.at_level(logging.DEBUG, logger="marqov.executors.braket"):
        result = await execute(metadata_error=error)
    assert result.metadata["execution_duration_ms"] is None
    assert result.execution_time_ms == 250.0
    assert "metadata" in caplog.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [0, 125])
async def test_real_simulator_metadata_duration_including_zero(duration):
    additional = AdditionalMetadata(
        action=Program(instructions=[]),
        simulatorMetadata=SimulatorMetadata(executionDuration=duration),
    )
    result = await execute(additional_metadata=additional)
    assert result.metadata["execution_duration_ms"] == duration
    assert result.execution_time_ms == duration
    assert result.metadata["wall_time_ms"] == 250.0
    # Local wall time includes conversion, polling and download: it is not queue time.
    assert result.metadata["queue_time_ms"] is None


@pytest.mark.asyncio
async def test_unknown_execution_duration_falls_back_to_wall_time():
    result = await execute()
    assert result.metadata["execution_duration_ms"] is None
    assert result.metadata["queue_time_ms"] is None
    assert result.execution_time_ms == result.metadata["wall_time_ms"] == 250.0


@pytest.mark.asyncio
async def test_real_duration_survives_metadata_lookup_failure():
    additional = AdditionalMetadata(
        action=Program(instructions=[]), simulatorMetadata=SimulatorMetadata(executionDuration=125)
    )
    result = await execute(
        additional_metadata=additional, metadata_error=RuntimeError("offline lookup")
    )
    assert result.metadata["execution_duration_ms"] == 125
    assert result.execution_time_ms == 125


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [-1, True, 1.5, float("nan"), float("inf"), "125"])
async def test_malformed_optional_duration_is_unknown(duration):
    additional = SimpleNamespace(simulatorMetadata=SimpleNamespace(executionDuration=duration))
    result = await execute(additional_metadata=additional)
    assert result.metadata["execution_duration_ms"] is None
    assert result.execution_time_ms == 250.0


@pytest.mark.asyncio
async def test_rigetti_program_duration_is_not_assumed_simulator_execution_ms():
    additional = SimpleNamespace(
        rigettiMetadata=SimpleNamespace(
            nativeQuilMetadata=SimpleNamespace(programDuration=300.1, qpuRuntimeEstimation=191.21)
        )
    )
    result = await execute(additional_metadata=additional)
    assert result.metadata["execution_duration_ms"] is None
    assert result.metadata["queue_time_ms"] is None


@pytest.mark.asyncio
async def test_broken_optional_result_property_preserves_counts():
    class Broken:
        @property
        def simulatorMetadata(self):
            raise RuntimeError("offline malformed optional metadata")

    result = await execute(additional_metadata=Broken())
    assert result.metadata["execution_duration_ms"] is None
