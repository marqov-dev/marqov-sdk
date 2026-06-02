"""Tests for BackendCache status polling."""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add platform/src to path for backend_cache import
sys.path.insert(0, "platform/src")

from marqov.executors.base import DeviceStatus


def _make_backend(**kwargs):
    """Create a Backend instance with defaults."""
    from backend_cache import Backend
    defaults = {
        "slug": "test", "name": "Test", "provider": "AWS Braket",
        "device_type": "qpu", "provider_target_id": "arn:test",
        "region": "us-east-1", "qubit_count": 25, "pricing": {},
        "is_available": True, "status": "online",
    }
    defaults.update(kwargs)
    return Backend(**defaults)


def _make_cache():
    """Create a BackendCache with mocked Supabase."""
    from backend_cache import BackendCache
    mock_supabase = MagicMock()
    mock_response = MagicMock()
    mock_response.data = []
    mock_supabase.from_.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
    cache = BackendCache(mock_supabase)
    return cache, mock_supabase


class TestGetOrCreateStatusExecutor:

    def test_returns_braket_executor_for_aws(self):
        cache, _ = _make_cache()
        backend = _make_backend(slug="ionq-aria-1", provider="AWS Braket")
        executor = cache._get_or_create_status_executor(backend)
        assert executor is not None

    def test_returns_none_for_azure(self):
        cache, _ = _make_cache()
        backend = _make_backend(slug="quantinuum-h1", provider="Azure Quantum")
        executor = cache._get_or_create_status_executor(backend)
        assert executor is None

    def test_caches_executor_instances(self):
        cache, _ = _make_cache()
        backend = _make_backend(slug="ionq-aria-1", provider="AWS Braket")
        first = cache._get_or_create_status_executor(backend)
        second = cache._get_or_create_status_executor(backend)
        assert first is second


class TestRefreshDeviceStatus:

    def test_skips_simulators(self):
        cache, mock_supabase = _make_cache()
        cache._cache["sv1"] = _make_backend(slug="sv1", device_type="simulator")
        cache._refresh_device_status()
        mock_supabase.table.assert_not_called()

    def test_writes_status_to_db(self):
        cache, mock_supabase = _make_cache()
        cache._cache["ionq-aria-1"] = _make_backend(slug="ionq-aria-1")

        mock_executor = MagicMock()
        mock_status = DeviceStatus(status="offline", queue_depth=None, queue_time_seconds=None)

        async def fake_get_status():
            return mock_status

        mock_executor.get_status = fake_get_status
        cache._status_executors["ionq-aria-1"] = mock_executor

        cache._refresh_device_status()

        mock_supabase.table.assert_called_with("backends")
        update_call = mock_supabase.table.return_value.update
        update_call.assert_called_once()
        update_data = update_call.call_args[0][0]
        assert update_data["status"] == "offline"

    def test_error_isolation(self):
        cache, mock_supabase = _make_cache()
        cache._cache["failing"] = _make_backend(slug="failing")
        cache._cache["working"] = _make_backend(slug="working")

        # Failing executor
        failing_exec = MagicMock()
        async def fail_status():
            raise Exception("API down")
        failing_exec.get_status = fail_status
        cache._status_executors["failing"] = failing_exec

        # Working executor
        working_exec = MagicMock()
        async def ok_status():
            return DeviceStatus(status="online", queue_depth=3, queue_time_seconds=90)
        working_exec.get_status = ok_status
        cache._status_executors["working"] = working_exec

        cache._refresh_device_status()

        # Working backend should still have been updated despite failing one
        assert mock_supabase.table.called
