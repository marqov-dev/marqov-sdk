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
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig

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
