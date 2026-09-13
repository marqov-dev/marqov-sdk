"""Unsupported options must fail before any simulator or input processing."""

from unittest.mock import Mock

import pytest

from marqov.executors.qilisdk import QiliSDKExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["execute", "execute_analog"])
@pytest.mark.parametrize(
    "options",
    [{"unknown_option": 42}, {"num_threads": 2}, {"z_option": "secret-value", "a_option": True}],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["execute", "execute_analog"])
@pytest.mark.parametrize(
    "seed,error",
    [(True, TypeError), (1.5, TypeError), ("42", TypeError), (-1, ValueError), (2**31, ValueError)],
)
async def test_invalid_seed_fails_before_execution(method, seed, error):
    executor = object.__new__(QiliSDKExecutor)
    executor._backend = Mock()
    executor._validate_circuit = Mock()
    with pytest.raises(error):
        await getattr(executor, method)(object(), seed=seed)
    executor._backend.execute.assert_not_called()
    executor._validate_circuit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["execute", "execute_analog"])
async def test_qutip_seed_is_explicitly_rejected(method):
    from marqov.executors.qilisdk import QiliSDKExecutorConfig

    executor = object.__new__(QiliSDKExecutor)
    executor.config = QiliSDKExecutorConfig(simulator="qutip")
    executor._backend = Mock()
    with pytest.raises(ValueError, match="only by the qilisim"):
        await getattr(executor, method)(object(), seed=42)
    executor._backend.execute.assert_not_called()
