"""Unsupported options must fail before any simulator or input processing."""

from unittest.mock import Mock

import pytest

from marqov.executors.qilisdk import QiliSDKExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["execute", "execute_analog"])
@pytest.mark.parametrize(
    "options", [{"seed": 42}, {"num_threads": 2}, {"z_option": "secret-value", "a_option": True}]
)
async def test_unsupported_options_fail_before_execution(method, options):
    executor = object.__new__(QiliSDKExecutor)
    executor._backend = Mock()
    executor._validate_circuit = Mock()
    with pytest.raises(TypeError) as caught:
        await getattr(executor, method)(object(), **options)
    assert str(caught.value) == (
        f"QiliSDKExecutor.{method}() got unsupported options: " + ", ".join(sorted(options))
    )
    assert "secret-value" not in str(caught.value)
    executor._validate_circuit.assert_not_called()
    executor._backend.execute.assert_not_called()
