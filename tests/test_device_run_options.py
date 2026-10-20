"""Unrecognised run options must refuse loudly, never vanish (marqov-sdk#77).

A dropped option is indistinguishable from an applied one: ``verbatim=True``
silently ignored once submitted circuits unboxed and the compiler folded the
benchmark away. Each entry point names the option it refuses and itself, and
refuses before provider initialization or submission.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from braket.circuits import Circuit as BraketCircuit

from marqov.circuits import Circuit
from marqov.device import MarqovDevice
from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig
from marqov.executors.cudaq import CudaqExecutor, CudaqExecutorConfig
from marqov.executors.cunqa import CUNQAExecutor, CUNQAExecutorConfig
from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig
from marqov.executors.local import LocalExecutor
from marqov.executors.quantinuum import QuantinuumExecutor, QuantinuumExecutorConfig
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig

ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"


def native():
    return Circuit.from_braket(BraketCircuit().rx(0, 0.3).rz(1, -0.7).cz(0, 1))


def providers():
    task = SimpleNamespace(
        id="offline-task",
        metadata=lambda: {},
        result=lambda: SimpleNamespace(
            measurement_counts={"10": 4},
            measured_qubits=[0, 1],
            measurement_counts_copied_from_device=True,
        ),
    )
    provider = SimpleNamespace(name="offline", run=Mock(return_value=task))
    executor = BraketExecutor(
        BraketExecutorConfig(device_arn=ARN, s3_bucket="unused", s3_prefix="test")
    )
    executor._get_device = AsyncMock(return_value=provider)
    device = MarqovDevice("rigetti", {"device_arn": ARN, "s3_destination_folder": ("unused", "test")})
    device._get_provider_device = Mock(return_value=provider)
    return executor, device, provider


def instruction_names(circuit):
    return [instruction.operator.name for instruction in circuit.instructions]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [{"verbatum": True}, {"verbatim": True, "shot_count": 5}, {"s3_bucket": "other"}],
)
async def test_braket_executor_refuses_unknown_options_before_initialization(options):
    executor, _, provider = providers()
    unknown = next(key for key in options if key != "verbatim")
    with pytest.raises(TypeError, match=rf"BraketExecutor\.execute\(\).*{unknown}"):
        await executor.execute(native(), shots=4, **options)
    executor._get_device.assert_not_called()
    provider.run.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [{"verbatum": True}, {"verbatim": True, "shot_count": 5}, {"s3_bucket": "other"}],
)
def test_device_cloud_braket_refuses_unknown_options_before_initialization(options):
    _, device, provider = providers()
    unknown = next(key for key in options if key != "verbatim")
    with pytest.raises(TypeError, match=rf"MarqovDevice\.run\(\).*'rigetti'.*{unknown}"):
        device.run(native(), shots=4, **options)
    device._get_provider_device.assert_not_called()
    provider.run.assert_not_called()


@pytest.mark.parametrize(
    "backend,params",
    [
        ("local", {}),
        ("marqov-sim", {"device_arn": ARN}),
        ("ibm", {"ibm_channel": "ibm_cloud"}),
        ("azure", {"azure_subscription_id": "offline"}),
    ],
)
@pytest.mark.parametrize(
    "options",
    [{"verbatim": True}, {"disable_qubit_rewiring": True}, {"verbatum": True}],
)
def test_non_cloud_braket_paths_refuse_every_option(backend, params, options):
    """These paths read no run options, so any option would otherwise be dropped."""
    device = MarqovDevice(backend, params)
    device._get_provider_device = Mock()
    (name,) = options
    with pytest.raises(TypeError, match=rf"MarqovDevice\.run\(\).*'{backend}'.*{name}"):
        device.run(native(), shots=4, **options)
    device._get_provider_device.assert_not_called()


@pytest.mark.asyncio
async def test_verbatim_reaches_the_braket_submission_on_both_entry_points():
    executor, device, provider = providers()
    await executor.execute(native(), shots=4, verbatim=True)
    executor_submitted = provider.run.call_args.args[0]
    provider.run.reset_mock()
    device.run(native(), shots=4, verbatim=True)
    device_submitted = provider.run.call_args.args[0]
    for submitted in (executor_submitted, device_submitted):
        names = instruction_names(submitted)
        assert names[0] == "StartVerbatimBox"
        assert names[-1] == "EndVerbatimBox"


def azure(framework):
    return AzureQuantumExecutor(
        AzureQuantumExecutorConfig(
            subscription_id="sub", resource_group="rg", workspace_name="ws",
            location="eastus", target="ionq.simulator", framework=framework,
        )
    )


def quantinuum():
    return QuantinuumExecutor(
        QuantinuumExecutorConfig(device_name="H2-1", simulator="state-vector")
    )


def guard(monkeypatch, target, name):
    """Replace a provider touchpoint with a recorder that fails if reached."""
    stub = Mock(side_effect=AssertionError(f"provider reached: {name}"))
    monkeypatch.setattr(target, name, stub)
    return stub


def local_setup(monkeypatch):
    executor = LocalExecutor()
    return executor, [
        guard(monkeypatch, Circuit, "simulate"),
        guard(monkeypatch, executor, "_sample_measurements"),
    ]


def cudaq_setup(monkeypatch):
    import marqov.executors.cudaq as cudaq_module

    return CudaqExecutor(CudaqExecutorConfig(target="qpp-cpu")), [
        guard(monkeypatch, cudaq_module, "_import_cudaq"),
    ]


def rigetti_setup(monkeypatch):
    executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="9q-square-qvm"))
    return executor, [guard(monkeypatch, executor, "_get_qc_sync")]


def ionq_setup(monkeypatch):
    session = Mock()
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=session)
    return executor, [
        guard(monkeypatch, session, "request"),
        guard(monkeypatch, executor, "_request"),
        guard(monkeypatch, executor, "_request_sync"),
    ]


def azure_setup(framework):
    def setup(monkeypatch):
        executor = azure(framework)
        return executor, [
            guard(monkeypatch, executor, "_get_workspace"),
            guard(monkeypatch, executor, "_create_workspace_sync"),
        ]

    return setup


def cunqa_setup(monkeypatch):
    client = Mock()
    executor = CUNQAExecutor(CUNQAExecutorConfig(n_qpus=1, walltime="00:05:00"), client=client)
    return executor, [
        guard(monkeypatch, client, name) for name in ("qraise", "get_QPUs", "run", "gather", "qdrop")
    ]


def quantinuum_setup(monkeypatch):
    executor = quantinuum()
    return executor, [guard(monkeypatch, executor, "_get_backend_sync")]


EXECUTORS = {
    "LocalExecutor": local_setup,
    "CudaqExecutor": cudaq_setup,
    "RigettiExecutor": rigetti_setup,
    "IonQExecutor": ionq_setup,
    "AzureQuantumExecutor-qiskit": azure_setup("qiskit"),
    "AzureQuantumExecutor-cirq": azure_setup("cirq"),
    "CUNQAExecutor": cunqa_setup,
    "QuantinuumExecutor": quantinuum_setup,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("key", sorted(EXECUTORS))
@pytest.mark.parametrize("option", ["verbatum", "verbatim", "max_costs"])
async def test_every_executor_refuses_options_it_does_not_apply(key, option, monkeypatch):
    """Executors that used to accept and drop options now name and refuse them.

    Each executor's first provider touchpoint is replaced with a recorder that
    raises, so a missing guard fails here deterministically and never reaches
    a simulator, QVM, socket or vendor SDK.
    """
    executor, guards = EXECUTORS[key](monkeypatch)
    owner = key.split("-")[0]
    with pytest.raises(TypeError, match=rf"{owner}\.execute\(\).*{option}"):
        await executor.execute(Circuit().x(0).z(1), shots=4, **{option: True})
    for stub in guards:
        stub.assert_not_called()


@pytest.mark.asyncio
async def test_quantinuum_forwards_documented_pytket_options():
    backend = Mock()
    backend.get_compiled_circuit.side_effect = lambda circuit, **_: circuit
    backend.process_circuit.return_value = ("job",)
    backend.get_jobid.return_value = "job"
    backend.get_result.return_value.get_counts.return_value = {(1, 0): 4}
    executor = quantinuum()
    executor._get_backend = AsyncMock(return_value=backend)
    await executor.execute(Circuit().x(0).z(1), shots=4, postprocess=True, seed=7)
    assert backend.process_circuit.call_args.kwargs == {
        "n_shots": 4, "postprocess": True, "seed": 7,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "option,value",
    [
        ("max_cost", 5),
        ("group", "usage"),
        ("pytketpass", object()),
        ("request_options", {}),
        ("options", {"tket-opt-level": 0}),
        ("api_handler", object()),
    ],
)
async def test_quantinuum_refuses_options_pytket_process_circuit_never_reads(
    option, value, monkeypatch
):
    """pytket-quantinuum 0.59.1 documents these but drops them in process_circuits."""
    executor, guards = quantinuum_setup(monkeypatch)
    with pytest.raises(TypeError, match=rf"QuantinuumExecutor\.execute\(\).*{option}"):
        await executor.execute(Circuit().x(0).z(1), shots=4, **{option: value})
    for stub in guards:
        stub.assert_not_called()
