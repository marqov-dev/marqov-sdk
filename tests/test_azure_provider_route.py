"""The program both Azure paths hand to azure-quantum's Qiskit provider.

``AzureQuantumExecutor`` and the Azure branch of ``MarqovDevice`` both go
through ``AzureQuantumProvider(workspace).get_backend(target).run(...)``.
These tests build the real provider over an offline workspace, so
``get_backend`` picks the backend class azure-quantum would pick, and the
backend's own translation (Qiskit to QIR through qdk) runs. Only the final
upload, ``AzureBackendBase._run``, is replaced: it records the program and
returns a real ``AzureQuantumJob`` over an offline vendor job that serves a
``microsoft.quantum-results.v2`` histogram.

What is checked is the program itself: the circuit passed to ``backend.run``
and the QIR it becomes. Neither may contain a barrier (Rigetti's adaptor
refuses ``__quantum__qis__barrier__body``), and measurements must be added
without one.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from azure.quantum._client.models import TargetStatus
from azure.quantum.qiskit import AzureQuantumProvider
from azure.quantum.qiskit.backends.backend import AzureBackendBase
from azure.quantum.qiskit.job import AzureQuantumJob

from marqov.circuits import Circuit
from marqov.device import MarqovDevice
from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig

# (target, provider id): one target per gate-model provider the Qiskit
# provider serves, simulators and a QPU name. Each resolves to a QIR backend.
TARGETS = [
    ("rigetti.sim.qvm", "rigetti"),
    ("ionq.simulator", "ionq"),
    ("ionq.qpu.forte-1", "ionq"),
    ("quantinuum.sim.h2-1e", "quantinuum"),
]
TARGET_IDS = [name for name, _ in TARGETS]

_QUBIT = r"(?:null|inttoptr \(i64 (\d+) to %Qubit\*\))"
_RESULT = r"(?:null|inttoptr \(i64 (\d+) to %Result\*\))"
_MEASURE = re.compile(rf"call void @__quantum__qis__m__body\(%Qubit\* {_QUBIT}, %Result\* {_RESULT}\)")
_RECORD = re.compile(rf"call void @__quantum__rt__result_record_output\(%Result\* {_RESULT},")


def _offline_workspace(target: str, provider_id: str) -> Mock:
    """A workspace that lists one available target and is never reached."""
    workspace = Mock()
    status = TargetStatus({"id": target, "currentAvailability": "Available", "averageQueueTime": 0})
    workspace._get_target_status.return_value = [(provider_id, status)]
    return workspace


class _NativeJob:
    """Offline vendor job serving a microsoft.quantum-results.v2 histogram."""

    def __init__(self, target: str, input_params: dict, metadata: dict, outcomes: dict) -> None:
        self.id = "offline-job"
        self._outcomes = outcomes  # tuple(outcome bits) -> count
        self.details = SimpleNamespace(
            id=self.id, name=self.id, target=target, status="Succeeded",
            output_data_format="microsoft.quantum-results.v2",
            input_params=input_params, metadata=dict(metadata), error_data=None,
        )

    def wait_until_completed(self, **kwargs: Any) -> None:
        return None

    def refresh(self) -> None:
        return None

    def get_results_histogram(self, **kwargs: Any) -> dict:
        return {
            str(list(outcome)): {"outcome": list(outcome), "count": count}
            for outcome, count in self._outcomes.items()
        }

    def get_results_shots(self, **kwargs: Any) -> list:
        return [list(outcome) for outcome, count in self._outcomes.items() for _ in range(count)]


class _Submission:
    """Replaces only the upload step and simulates a basis-state result.

    The simulated outcome follows the submitted QIR: each recorded result
    reports the bit of the qubit measured into it, in the order the QIR
    records them. The test supplies which qubits are 1 (the circuit is a
    basis state), so the counts keys depend on the real translation.
    """

    def __init__(self, ones: set[int], shots: int) -> None:
        self.ones = ones
        self.shots = shots
        self.circuits: list[Any] = []
        self.qir: list[str] = []
        self.targets: list[str] = []

    def outcome(self, qir: str) -> tuple[int, ...]:
        result_to_qubit = {
            int(result or 0): int(qubit or 0)
            for qubit, result in _MEASURE.findall(qir)
        }
        return tuple(
            1 if result_to_qubit[int(result or 0)] in self.ones else 0
            for result in _RECORD.findall(qir)
        )

    def __enter__(self) -> _Submission:
        submission = self

        def fake_upload(backend, job_name, input_data, input_params, metadata, **options):
            qir = input_data.decode("utf-8")
            submission.qir.append(qir)
            submission.targets.append(backend.name)
            native = _NativeJob(
                backend.name, input_params, metadata,
                {submission.outcome(qir): submission.shots},
            )
            return AzureQuantumJob(backend, azure_job=native)

        self._patches = [patch.object(AzureBackendBase, "_run", fake_upload)]
        for patcher in self._patches:
            patcher.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for patcher in self._patches:
            patcher.stop()

    def record_circuits(self, backend: Any) -> None:
        """Wrap backend.run so the circuit it is handed is kept."""
        run = backend.run

        def recording_run(circuit, *args, **kwargs):
            self.circuits.append(circuit)
            return run(circuit, *args, **kwargs)

        backend.run = recording_run


def _real_backend(target: str, provider_id: str) -> Any:
    provider = AzureQuantumProvider(workspace=_offline_workspace(target, provider_id))
    return provider.get_backend(target)


def _executor(target: str, backend: Any = None) -> AzureQuantumExecutor:
    executor = AzureQuantumExecutor(AzureQuantumExecutorConfig(
        subscription_id="sub", resource_group="rg", workspace_name="ws",
        location="eastus", target=target,
    ))
    executor._backend = backend
    return executor


def _device(target: str) -> MarqovDevice:
    return MarqovDevice(target, {
        "backend": target, "subscription_id": "sub",
        "resource_group": "rg", "workspace_name": "ws",
    })


def _assert_measured_without_barrier(circuit: Any, width: int) -> None:
    """Every qubit measured once into 'meas', in order, and no barrier anywhere."""
    names = [instruction.operation.name for instruction in circuit.data]
    assert "barrier" not in names
    assert [register.name for register in circuit.cregs] == ["meas"]
    assert circuit.cregs[0].size == width
    measures = [inst for inst in circuit.data if inst.operation.name == "measure"]
    assert [
        (circuit.find_bit(inst.qubits[0]).index, circuit.find_bit(inst.clbits[0]).index)
        for inst in measures
    ] == [(qubit, qubit) for qubit in range(width)]
    # The measurements close the circuit: nothing follows them.
    assert names[-width:] == ["measure"] * width


# A basis-state probe that is asymmetric (qubit 0 leftmost reads "100..."),
# with an identity wire (qubit 2) that must still be measured and reported.
PROBE_WIDTH = 4
PROBE_ONES = {0, 3}
PROBE_KEY = "1001"


def _probe() -> Circuit:
    return Circuit(num_qubits=PROBE_WIDTH).x(0).z(1).x(3)


class TestExecutorSubmission:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("target", "provider_id"), TARGETS, ids=TARGET_IDS)
    async def test_prepared_circuit_and_qir_carry_no_barrier(self, target, provider_id) -> None:
        backend = _real_backend(target, provider_id)
        with _Submission(PROBE_ONES, shots=20) as submission:
            submission.record_circuits(backend)
            result = await _executor(target, backend).execute(_probe(), shots=20)

        [circuit] = submission.circuits
        _assert_measured_without_barrier(circuit, PROBE_WIDTH)
        [qir] = submission.qir
        assert "barrier" not in qir
        assert submission.targets == [target]
        assert result.counts == {PROBE_KEY: 20}

    @pytest.mark.asyncio
    async def test_a_barrier_in_the_exported_circuit_is_removed(self) -> None:
        """Barriers never change results, so none reaches any target."""
        from qiskit import QuantumCircuit

        def with_barriers(self) -> QuantumCircuit:
            exported = QuantumCircuit(2)
            exported.x(0)
            exported.barrier()
            exported.z(1)
            exported.barrier(0)
            return exported

        backend = _real_backend("rigetti.sim.qvm", "rigetti")
        with patch.object(Circuit, "to_qiskit", with_barriers), \
                _Submission({0}, shots=5) as submission:
            submission.record_circuits(backend)
            result = await _executor("rigetti.sim.qvm", backend).execute(
                Circuit().x(0).z(1), shots=5,
            )

        [circuit] = submission.circuits
        _assert_measured_without_barrier(circuit, 2)
        assert "barrier" not in submission.qir[0]
        assert result.counts == {"10": 5}

    @pytest.mark.asyncio
    async def test_a_circuit_without_qubits_is_refused_before_submission(self) -> None:
        backend = _real_backend("ionq.simulator", "ionq")
        with _Submission(set(), shots=5) as submission:
            submission.record_circuits(backend)
            with pytest.raises(ValueError, match=r"'ionq\.simulator'.*no qubits"):
                await _executor("ionq.simulator", backend).execute(Circuit(), shots=5)
        assert submission.circuits == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("framework", ["qiskit", "cirq"])
    async def test_pasqal_target_is_refused_before_azure_is_contacted(self, framework) -> None:
        executor = _executor("pasqal.sim.emu-tn")
        executor.config.framework = framework
        with patch.object(
            AzureQuantumExecutor, "_create_workspace_sync",
            side_effect=AssertionError("Azure must not be contacted"),
        ):
            with pytest.raises(ValueError, match=r"'pasqal\.sim\.emu-tn'.*pulse"):
                await executor.execute(Circuit().x(0), shots=5)


class TestDeviceSubmission:
    """MarqovDevice's Azure branch uses the same provider route.

    The workspace is the only thing replaced: the device builds the real
    ``AzureQuantumProvider`` over it and asks it for the backend.
    """

    @staticmethod
    def _run(target: str, provider_id: str, circuit: Circuit, ones: set[int], shots: int):
        workspace = _offline_workspace(target, provider_id)
        recorded: list[Any] = []
        real_get_backend = AzureQuantumProvider.get_backend

        def get_backend(provider, name=None, **kwargs):
            backend = real_get_backend(provider, name, **kwargs)
            recorded.append(backend)
            return backend

        with patch("azure.quantum.Workspace", return_value=workspace) as workspace_cls, \
                patch.object(AzureQuantumProvider, "get_backend", get_backend), \
                _Submission(ones, shots=shots) as submission:
            device = _device(target)
            # Record circuits on the backend once the device has asked for it.
            original = MarqovDevice._get_provider_device

            def provider_device(self):
                backend = original(self)
                if not hasattr(backend, "_recording"):
                    submission.record_circuits(backend)
                    backend._recording = True
                return backend

            with patch.object(MarqovDevice, "_get_provider_device", provider_device):
                counts = device.run(circuit, shots=shots)

        workspace_cls.assert_called_once_with(
            subscription_id="sub", resource_group="rg", name="ws", location="eastus",
        )
        workspace.get_targets.assert_not_called()
        return counts, submission, recorded

    @pytest.mark.parametrize(("target", "provider_id"), TARGETS, ids=TARGET_IDS)
    def test_device_submits_through_the_provider_without_barrier(self, target, provider_id) -> None:
        counts, submission, recorded = self._run(target, provider_id, _probe(), PROBE_ONES, 30)

        assert [backend.name for backend in recorded] == [target]
        [circuit] = submission.circuits
        _assert_measured_without_barrier(circuit, PROBE_WIDTH)
        assert "barrier" not in submission.qir[0]
        assert counts == {PROBE_KEY: 30}
        assert all(type(value) is int for value in counts.values())

    @pytest.mark.asyncio
    async def test_device_and_executor_report_the_same_keys(self) -> None:
        target, provider_id = TARGETS[0]
        device_counts, _, _ = self._run(target, provider_id, _probe(), PROBE_ONES, 8)

        backend = _real_backend(target, provider_id)
        with _Submission(PROBE_ONES, shots=8):
            result = await _executor(target, backend).execute(_probe(), shots=8)

        assert device_counts == result.counts == {PROBE_KEY: 8}

    def test_device_refuses_pasqal_before_azure_is_contacted(self) -> None:
        with patch("azure.quantum.Workspace", side_effect=AssertionError("Azure must not be contacted")):
            with pytest.raises(ValueError, match=r"'pasqal\.sim\.emu-tn'.*pulse"):
                _device("pasqal.sim.emu-tn").run(Circuit().x(0), shots=5)

    def test_device_refuses_a_circuit_without_qubits(self) -> None:
        with patch("azure.quantum.Workspace", side_effect=AssertionError("Azure must not be contacted")):
            with pytest.raises(ValueError, match=r"'ionq\.simulator'.*no qubits"):
                _device("ionq.simulator").run(Circuit(), shots=5)

    def test_device_raises_naming_the_target_when_the_job_fails(self) -> None:
        target, provider_id = "rigetti.sim.qvm", "rigetti"
        workspace = _offline_workspace(target, provider_id)

        def failed_upload(backend, job_name, input_data, input_params, metadata, **options):
            native = _NativeJob(target, input_params, metadata, {})
            native.details.status = "Failed"
            return AzureQuantumJob(backend, azure_job=native)

        with patch("azure.quantum.Workspace", return_value=workspace), \
                patch.object(AzureBackendBase, "_run", failed_upload):
            with pytest.raises(RuntimeError, match=r"Job failed.*'rigetti\.sim\.qvm'"):
                _device(target).run(Circuit().x(0), shots=5)
