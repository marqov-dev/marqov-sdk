"""Input validation and provider-shape checks found by the 2026-10 review."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import timedelta
from fractions import Fraction
from unittest.mock import MagicMock

import numpy as np
import pytest
import temporalio.api.common.v1
from google.protobuf.duration_pb2 import Duration
from temporalio.common import RetryPolicy

from marqov.circuits import Circuit
from marqov.executors.ionq import IonQExecutorConfig
from marqov.executors.quantinuum import QuantinuumExecutorConfig
from marqov.workflows.decorators import task
from marqov.workflows.graph import TaskConfig

BAD_SECONDS = [0, -1, math.nan, math.inf, True, "60"]


@pytest.mark.parametrize(
    "build",
    [
        lambda c: c.cnot(1, 1),
        lambda c: c.cx(1, 1),
        lambda c: c.cz(0, 0),
        lambda c: c.swap(2, 2),
        lambda c: c.ccx(0, 0, 1),
        lambda c: c.ccx(0, 1, 1),
        lambda c: c.toffoli(1, 0, 1),
        lambda c: c.cswap(0, 1, 1),
        lambda c: c.fredkin(0, 0, 1),
    ],
)
def test_repeated_qubit_is_rejected_by_the_builder(build):
    with pytest.raises(ValueError, match="distinct qubits"):
        build(Circuit())


def test_repeated_qubit_is_rejected_by_from_dict():
    data = Circuit().cnot(0, 1).to_dict()
    data["gates"][0]["qubits"] = [1, 1]
    with pytest.raises(ValueError, match="distinct qubits"):
        Circuit.from_dict(data)


def test_distinct_qubits_are_unaffected():
    circuit = Circuit().x(0).cnot(0, 1).swap(1, 2).ccx(0, 1, 2).cswap(2, 0, 1)
    assert circuit.num_qubits == 3


@pytest.mark.parametrize("timeout", BAD_SECONDS)
def test_ionq_rejects_an_unusable_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        IonQExecutorConfig(api_key="k", timeout_seconds=timeout)


@pytest.mark.parametrize("timeout", BAD_SECONDS)
def test_quantinuum_rejects_an_unusable_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        QuantinuumExecutorConfig(device_name="H2-1", timeout_seconds=timeout)


@pytest.mark.parametrize("timeout", [None, 0.5, 60])
def test_usable_timeouts_are_kept(timeout):
    assert IonQExecutorConfig(timeout_seconds=timeout).timeout_seconds == timeout
    quantinuum = QuantinuumExecutorConfig(device_name="H2-1", timeout_seconds=timeout)
    assert quantinuum.timeout_seconds == timeout


@pytest.mark.parametrize("retries", [-1, -2, 1.5, True, "1"])
def test_task_rejects_unusable_retries(retries):
    with pytest.raises(ValueError, match="retries"):
        task(retries=retries)(lambda: 1)


@pytest.mark.parametrize("timeout", BAD_SECONDS)
def test_task_rejects_an_unusable_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        task(timeout=timeout)(lambda: 1)


@pytest.mark.parametrize("timeout", [1e300, 10**400, 1e-9])
def test_task_rejects_a_timeout_temporal_cannot_represent(timeout):
    with pytest.raises(ValueError, match="timeout"):
        task(timeout=timeout)(lambda: 1)


def test_task_rejects_retries_temporal_cannot_represent():
    with pytest.raises(ValueError, match="retries"):
        task(retries=2**31 - 1)(lambda: 1)


def test_task_config_accepts_the_largest_representable_settings():
    config = TaskConfig(name="t", retries=2**31 - 2, timeout_seconds=315_576_000_000)
    retry_policy = RetryPolicy(maximum_attempts=config.retries + 1)
    retry_policy.apply_to_proto(temporalio.api.common.v1.RetryPolicy())
    Duration().FromTimedelta(timedelta(seconds=config.timeout_seconds))


@pytest.mark.parametrize("timeout", [np.int64(60), np.float32(1.5), Fraction(1, 2)])
def test_task_timeout_is_stored_as_a_json_ready_float(timeout):
    config = TaskConfig(name="t", timeout_seconds=timeout)
    assert type(config.timeout_seconds) is float
    assert json.loads(json.dumps(config.timeout_seconds)) == float(timeout)


def test_task_keeps_usable_settings():
    assert task(retries=2, timeout=1.5)(lambda: 1)() == 1


@pytest.fixture
def queue_type():
    return pytest.importorskip("braket.aws.queue_information").QueueType


def _braket_with_queue(quantum_tasks):
    from braket.aws.queue_information import QueueDepthInfo

    from marqov.executors.braket import BraketExecutor, BraketExecutorConfig

    executor = BraketExecutor(
        BraketExecutorConfig(device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                             s3_bucket="bucket")
    )
    device = MagicMock()
    device.status = "ONLINE"
    device.queue_depth.return_value = QueueDepthInfo(quantum_tasks=quantum_tasks, jobs="0")
    executor._device = device
    return executor


def test_braket_status_sums_the_string_queue_counts(queue_type):
    executor = _braket_with_queue({queue_type.NORMAL: "7", queue_type.PRIORITY: "3"})
    status = asyncio.run(executor.get_status())
    assert status.queue_depth == 10
    assert status.queue_time_seconds == 300


def test_braket_status_reads_a_lower_bound_count(queue_type):
    executor = _braket_with_queue({queue_type.NORMAL: ">4000", queue_type.PRIORITY: "0"})
    assert asyncio.run(executor.get_status()).queue_depth == 4000


def test_braket_status_leaves_an_unreadable_count_unknown(queue_type):
    executor = _braket_with_queue({queue_type.NORMAL: "many"})
    assert asyncio.run(executor.get_status()).queue_depth is None


def test_braket_queue_depth_has_the_declared_shape(queue_type):
    executor = _braket_with_queue({queue_type.NORMAL: "7", queue_type.PRIORITY: ">4000"})
    assert asyncio.run(executor.get_queue_depth()) == {"Normal": 7, "Priority": 4000}
