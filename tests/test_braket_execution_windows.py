"""Tests for Braket execution-window serialization — the JSON shape consumed by downstream availability tooling.

The serialized shape is a single contract with that consumer; a drift here (wrong token, wrong time
format, or dropping the []/None distinction) would silently break the advisory downstream.
"""

from datetime import time
from types import SimpleNamespace

import pytest
from braket.device_schema.device_execution_window import DeviceExecutionWindow, ExecutionDay

from marqov.executors.base import DeviceStatus
from marqov.executors.braket import (
    BraketExecutor,
    BraketExecutorConfig,
    _serialize_execution_windows,
)

THE_TEN = {
    "Everyday", "Weekdays", "Weekend",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}


def _device(windows: object) -> SimpleNamespace:
    return SimpleNamespace(properties=SimpleNamespace(service=SimpleNamespace(executionWindows=windows)))


def test_serializes_value_and_hhmmss() -> None:
    # NB: the enum MEMBER is WEEKENDS but its .value is "Weekend" — we must serialize .value, not .name.
    windows = [
        DeviceExecutionWindow(executionDay=ExecutionDay.WEEKENDS, windowStartHour=time(0, 0, 0), windowEndHour=time(6, 59, 0)),
    ]
    assert _serialize_execution_windows(_device(windows)) == [
        {"executionDay": "Weekend", "windowStartHour": "00:00:00", "windowEndHour": "06:59:00"},
    ]


def test_truncates_subsecond_to_hhmmss() -> None:
    windows = [
        DeviceExecutionWindow(executionDay=ExecutionDay.EVERYDAY, windowStartHour=time(0, 0, 0), windowEndHour=time(23, 59, 59, 999999)),
    ]
    out = _serialize_execution_windows(_device(windows))
    assert out is not None and out[0]["windowEndHour"] == "23:59:59"  # not "23:59:59.999999"


def test_empty_windows_preserved_as_empty_list() -> None:
    # [] (device reports none) must stay distinct from None (unknown) — the platform's state model relies on it.
    assert _serialize_execution_windows(_device([])) == []


def test_missing_property_fails_safe_to_none() -> None:
    class NoProps:
        @property
        def properties(self) -> object:
            raise RuntimeError("device has no properties")

    assert _serialize_execution_windows(NoProps()) is None


def test_non_iterable_windows_fails_safe_to_none() -> None:
    assert _serialize_execution_windows(_device(object())) is None


def test_execution_day_enum_has_exactly_the_ten_values() -> None:
    # A future braket bump that ADDS a member fails this — forcing a re-look before windows silently drift.
    assert {e.value for e in ExecutionDay} == THE_TEN


def test_device_status_defaults_execution_windows_to_none() -> None:
    assert DeviceStatus(status="online", queue_depth=0, queue_time_seconds=0).execution_windows is None
    assert DeviceStatus.always_online().execution_windows is None


@pytest.mark.asyncio
async def test_get_status_includes_serialized_windows() -> None:
    from unittest.mock import patch

    windows = [
        DeviceExecutionWindow(executionDay=ExecutionDay.EVERYDAY, windowStartHour=time(9, 0, 0), windowEndHour=time(17, 0, 0)),
    ]
    device = SimpleNamespace(
        status="ONLINE",
        properties=SimpleNamespace(service=SimpleNamespace(executionWindows=windows)),
        queue_depth=lambda: SimpleNamespace(quantum_tasks={}),
    )
    config = BraketExecutorConfig(
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_bucket="b"
    )
    with patch("marqov.executors.braket.AwsDevice", return_value=device), \
        patch("marqov.executors.braket.boto3.Session"), \
        patch("marqov.executors.braket.AwsSession"):
        status = await BraketExecutor(config).get_status()

    assert status.execution_windows == [
        {"executionDay": "Everyday", "windowStartHour": "09:00:00", "windowEndHour": "17:00:00"},
    ]
