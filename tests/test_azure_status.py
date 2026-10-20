"""Azure device-status accessors driven through a provider-shaped workspace.

The fake mirrors ``azure.quantum.Workspace.get_targets()``: objects carrying
``name`` and ``current_availability``, whose vocabulary is exactly
``Available``, ``Degraded`` and ``Unavailable``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from marqov.executors import AzureQuantumExecutor
from marqov.executors.azure import AzureQuantumExecutorConfig


def _executor(workspace) -> AzureQuantumExecutor:
    config = AzureQuantumExecutorConfig(
        subscription_id="sub", resource_group="rg",
        workspace_name="ws", location="eastus", target="ionq.simulator",
    )
    executor = AzureQuantumExecutor(config)

    async def _get_workspace():
        return workspace

    patcher = patch.object(executor, "_get_workspace", _get_workspace)
    patcher.start()
    return executor


def _workspace(availability):
    targets = [
        SimpleNamespace(name="other.target", current_availability="Available"),
        SimpleNamespace(name="ionq.simulator", current_availability=availability),
    ]
    return SimpleNamespace(get_targets=lambda: targets)


class _BrokenWorkspace:
    def get_targets(self):
        raise RuntimeError("credential or network failure")


@pytest.mark.parametrize(
    ("availability", "available", "status"),
    [
        ("Available", True, "online"),
        ("Degraded", False, "maintenance"),
        ("Unavailable", False, "offline"),
    ],
)
class TestAzureStatusVocabulary:
    @pytest.mark.asyncio
    async def test_device_status_is_raw_value(self, availability, available, status) -> None:
        assert await _executor(_workspace(availability)).get_device_status() == availability

    @pytest.mark.asyncio
    async def test_is_device_available(self, availability, available, status) -> None:
        assert await _executor(_workspace(availability)).is_device_available() is available

    @pytest.mark.asyncio
    async def test_get_status(self, availability, available, status) -> None:
        assert (await _executor(_workspace(availability)).get_status()).status == status


class TestAzureStatusCannotTell:
    @pytest.mark.asyncio
    async def test_workspace_error_is_maintenance_not_offline(self) -> None:
        executor = _executor(_BrokenWorkspace())
        assert (await executor.get_status()).status == "maintenance"

    @pytest.mark.asyncio
    async def test_workspace_error_keeps_public_contract(self) -> None:
        executor = _executor(_BrokenWorkspace())
        assert await executor.get_device_status() == "Unknown"
        assert await executor.is_device_available() is False

    @pytest.mark.asyncio
    async def test_missing_target_is_maintenance(self) -> None:
        executor = _executor(SimpleNamespace(get_targets=lambda: []))
        assert await executor.get_device_status() == "Unknown"
        assert (await executor.get_status()).status == "maintenance"
        assert await executor.is_device_available() is False

    @pytest.mark.asyncio
    async def test_missing_target_logs_warning_naming_configured_target(self, caplog) -> None:
        workspace = SimpleNamespace(
            get_targets=lambda: [SimpleNamespace(name="ionq.qpu", current_availability="Available")]
        )
        executor = _executor(workspace)
        with caplog.at_level(logging.WARNING, logger="marqov.executors.azure"):
            assert (await executor.get_status()).status == "maintenance"
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("ionq.simulator" in m and "ionq.qpu" in m for m in messages)

    @pytest.mark.asyncio
    async def test_availability_is_case_and_whitespace_insensitive(self) -> None:
        assert await _executor(_workspace(" available ")).is_device_available() is True
