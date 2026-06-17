"""End-to-end tests for the Rigetti executor against a real local QVM.

These submit real programs through ``quilc`` and ``qvm``, so they are skipped
unless ``MARQOV_QVM_AVAILABLE=1`` is set. Start the containers first
(see ``CONTRIBUTING.md`` §4):

    docker run -d -p 5555:5555 rigetti/quilc -R
    docker run -d -p 5000:5000 rigetti/qvm -S

then run:

    MARQOV_QVM_AVAILABLE=1 pytest tests/integration/test_rigetti_qvm.py
"""

import os

import pytest

from marqov.circuits import bell_state, ghz_state
from marqov.executors import RigettiExecutor
from marqov.executors.rigetti import RigettiExecutorConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MARQOV_QVM_AVAILABLE") != "1",
        reason="Requires a local QVM + quilc (set MARQOV_QVM_AVAILABLE=1, see CONTRIBUTING.md §4)",
    ),
]


@pytest.fixture
def qvm_executor() -> RigettiExecutor:
    """A RigettiExecutor wired to the local 2-qubit QVM."""
    return RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm"))


@pytest.mark.asyncio
async def test_bell_state_only_correlated_outcomes(qvm_executor: RigettiExecutor) -> None:
    """A Bell state on the QVM produces only the correlated 00 and 11 outcomes."""
    result = await qvm_executor.execute(bell_state(), shots=1000)

    assert sum(result.counts.values()) == 1000
    # An ideal Bell state never yields 01 or 10.
    assert set(result.counts) <= {"00", "11"}
    # Both correlated outcomes should appear with ~50/50 weight (loose bound).
    assert result.counts.get("00", 0) > 250
    assert result.counts.get("11", 0) > 250
    assert result.backend == "2q-qvm"
    assert result.metadata["provider"] == "rigetti"
    assert result.metadata["num_qubits"] == 2


@pytest.mark.asyncio
async def test_ghz_state_only_correlated_outcomes() -> None:
    """A 3-qubit GHZ state on the QVM produces only 000 and 111."""
    executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="3q-qvm"))
    result = await executor.execute(ghz_state(3), shots=1000)

    assert sum(result.counts.values()) == 1000
    assert set(result.counts) <= {"000", "111"}
    assert result.counts.get("000", 0) > 250
    assert result.counts.get("111", 0) > 250


@pytest.mark.asyncio
async def test_qvm_status_is_online(qvm_executor: RigettiExecutor) -> None:
    """The QVM reports online with no queue."""
    status = await qvm_executor.get_status()
    assert status.status == "online"
    assert status.queue_depth == 0
