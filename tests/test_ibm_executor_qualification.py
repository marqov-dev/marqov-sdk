"""Real IBM Runtime SamplerV2 local execution and submitted-job identity."""

from unittest.mock import Mock

import pytest

pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("qiskit_aer")
from qiskit_aer import AerSimulator
from marqov.circuits import Circuit
from marqov.executors import IBMExecutor, IBMExecutorConfig
from marqov.executors.ibm import IBMExecutionError


@pytest.mark.asyncio
@pytest.mark.parametrize("qubit,key", [(0, "10"), (1, "01")])
async def test_real_sampler_local_counts_and_provenance(qubit, key):
    executor = IBMExecutor(IBMExecutorConfig(backend_name="aer_simulator"))
    # Explicit offline qualification backend, never a remote failure fallback.
    executor._backend = AerSimulator(seed_simulator=7)
    result = await executor.execute(Circuit().x(qubit).cz(0, 1), shots=32)
    assert result.counts == {key: 32}
    assert result.metadata["access_path"] == "local"
    assert result.metadata["engine"] == "SamplerV2"
    assert result.metadata["job_id"]
    assert len(result.metadata["reproducibility"]["counts_sha256"]) == 64


@pytest.mark.asyncio
async def test_unresolved_result_retains_submitted_job(monkeypatch):
    executor = IBMExecutor(IBMExecutorConfig(backend_name="configured-provider-device"))
    executor._backend = Mock()
    monkeypatch.setattr(executor, "_transpile_sync", lambda circuit, backend: circuit)
    job = Mock()
    job.job_id.return_value = "existing-provider-job"
    job.result.side_effect = TimeoutError()
    submit = Mock(return_value=job)
    monkeypatch.setattr(executor, "_submit_sampler_sync", submit)
    with pytest.raises(IBMExecutionError) as error:
        await executor.execute(Circuit().h(0), shots=32)
    assert error.value.job_id == "existing-provider-job"
    assert submit.call_count == 1


def test_token_not_in_config_repr():
    assert "private-test" not in repr(IBMExecutorConfig(backend_name="test", token="private-test"))
