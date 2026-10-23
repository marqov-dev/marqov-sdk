"""MarqovDevice and ExecutorFactory read one backend config the same way (marqov-sdk#162).

Both paths consume ``marqov.executors._params.normalize_backend_params``, so a
key name, an alias, and a default are each decided in one place. The
unprefixed spellings are canonical; the ``ibm_``/``azure_`` spellings still
work and warn. These tests drive the real ``device.py`` branches with only the
provider SDK boundary stubbed, so a path that stops reading the canonical key
fails here rather than silently using a default.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from marqov.circuits import Circuit
from marqov.device import MarqovDevice
from marqov.executors.factory import ExecutorFactory
from tests.test_platform_package import _ROOT, _package_parts, _platform_references

SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"


def _normalize(params: dict) -> dict:
    # Imported per call so the rest of this module still collects and runs
    # against a tree that does not have the normaliser yet.
    return importlib.import_module("marqov.executors._params").normalize_backend_params(params)


# ---------------------------------------------------------------------------
# Criterion 1: every renamed row of the issue's table
# ---------------------------------------------------------------------------

# (deprecated spelling, canonical spelling, sample value)
RENAMED_KEYS = [
    ("ibm_channel", "channel", "ibm_cloud"),
    ("ibm_instance", "instance", "crn:v1:bluemix:public:quantum-computing:us-east:a/abc::"),
    ("ibm_token", "token", "tok"),
    ("ibm_optimization_level", "optimization_level", 3),
    ("azure_subscription_id", "subscription_id", "sub-123"),
    ("azure_resource_group", "resource_group", "rg"),
    ("azure_workspace_name", "workspace_name", "ws"),
    ("azure_location", "location", "westus"),
]


@pytest.mark.parametrize(("deprecated", "canonical", "value"), RENAMED_KEYS)
def test_deprecated_and_canonical_spellings_normalise_to_the_same_value(
    deprecated, canonical, value
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_canonical = _normalize({canonical: value})

    with pytest.warns(DeprecationWarning, match=rf"'{deprecated}'.*'{canonical}'"):
        from_deprecated = _normalize({deprecated: value})

    assert from_deprecated == from_canonical
    assert from_canonical[canonical] == value
    assert deprecated not in from_deprecated


@pytest.mark.parametrize(("deprecated", "canonical", "value"), RENAMED_KEYS)
def test_matching_values_under_both_spellings_do_not_warn(deprecated, canonical, value):
    """The canonical spelling supplied the value, so nothing is deprecated in use."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _normalize({deprecated: value, canonical: value})[canonical] == value


@pytest.mark.parametrize(
    "params",
    [{"ibm_token": "t", "token": ""}, {"ibm_token": "", "token": "t"},
     {"ibm_token": None, "token": "t"}, {"ibm_token": "t", "token": None}],
)
def test_blank_spelling_is_absent_not_a_conflict(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _normalize(params)["token"] == "t"


def test_unrelated_keys_pass_through_untouched():
    params = {"device_arn": SV1_ARN, "backend_name": "ibm_kingston", "target": "ionq.simulator",
              "s3_bucket": "b", "provider": "AWS Braket", "shots": 10}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _normalize(params) == params


def test_normaliser_does_not_mutate_its_input():
    params = {"ibm_token": "tok"}
    with pytest.warns(DeprecationWarning):
        _normalize(params)
    assert params == {"ibm_token": "tok"}


# ---------------------------------------------------------------------------
# MarqovDevice provider routing recognises the canonical spellings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"channel": "ibm_cloud"}, "ibm"),
        ({"instance": "crn:v1:a/b/c"}, "ibm"),
        ({"channel": "ibm_cloud", "token": "tok"}, "ibm"),
        # A bare canonical token is ambiguous (IonQ, QB and others take one),
        # so it never selects IBM: it must not reach QiskitRuntimeService.
        ({"token": "tok"}, None),
        ({"subscription_id": "sub"}, "azure"),
        # Dicts that routed before canonical keys existed keep their route:
        # a stray canonical key never pulls them onto another provider.
        ({"device_arn": SV1_ARN, "token": "tok"}, "braket"),
        ({"device_arn": SV1_ARN, "channel": "x", "subscription_id": "s"}, "braket"),
        ({"azure_subscription_id": "sub", "token": "tok"}, "azure"),
        ({"ibm_token": "tok", "device_arn": SV1_ARN}, "ibm"),
    ],
)
def test_provider_routing_accepts_canonical_keys_without_rerouting_old_dicts(params, expected):
    from marqov.backends import is_azure, is_braket, is_ibm

    routed = "ibm" if is_ibm(params) else "azure" if is_azure(params) else (
        "braket" if is_braket(params) else None
    )
    assert routed == expected


# ---------------------------------------------------------------------------
# Criterion 2: the Braket S3 prefix defaults the same way on both paths
# ---------------------------------------------------------------------------


def _offline_braket_provider():
    task = SimpleNamespace(
        id="offline-task",
        metadata=lambda: {},
        result=lambda: SimpleNamespace(
            measurement_counts={"0": 4},
            measured_qubits=[0],
            measurement_counts_copied_from_device=True,
        ),
    )
    return SimpleNamespace(name="offline", run=Mock(return_value=task))


@pytest.mark.asyncio
async def test_bucket_only_braket_config_is_accepted_by_both_paths_with_marqov_prefix():
    """Before marqov-sdk#162, the MarqovDevice side raised ValueError: it required s3_prefix."""
    config = {"device_arn": SV1_ARN, "s3_bucket": "b"}

    device_provider = _offline_braket_provider()
    device = MarqovDevice("sv1", {"backend": "sv1", **config})
    with patch.object(MarqovDevice, "_get_provider_device", return_value=device_provider):
        device.run(Circuit().x(0), shots=4)
    device_folder = device_provider.run.call_args.args[1]

    factory_provider = _offline_braket_provider()
    executor = ExecutorFactory.create_executor("sv1", {"provider": "AWS Braket", **config})
    executor._get_device = AsyncMock(return_value=factory_provider)
    await executor.execute(Circuit().x(0), shots=4)
    factory_folder = factory_provider.run.call_args.kwargs["s3_destination_folder"]

    assert device_folder == factory_folder == ("b", "marqov")


def test_bare_token_dict_is_not_routed_to_ibm():
    """An IonQ-style key must not be handed to QiskitRuntimeService."""
    device = MarqovDevice("ionq-aria", {"backend": "ionq-aria", "token": "IONQ_KEY"})
    with pytest.raises(ValueError, match="Cannot determine provider"):
        device._get_provider_device()


@pytest.mark.parametrize("prefix", [None, ""])
@pytest.mark.asyncio
async def test_blank_s3_prefix_defaults_to_marqov_on_both_paths(prefix):
    config = {"device_arn": SV1_ARN, "s3_bucket": "b", "s3_prefix": prefix}

    device_provider = _offline_braket_provider()
    device = MarqovDevice("sv1", {"backend": "sv1", **config})
    with patch.object(MarqovDevice, "_get_provider_device", return_value=device_provider):
        device.run(Circuit().x(0), shots=4)

    factory_provider = _offline_braket_provider()
    executor = ExecutorFactory.create_executor("sv1", {"provider": "AWS Braket", **config})
    executor._get_device = AsyncMock(return_value=factory_provider)
    await executor.execute(Circuit().x(0), shots=4)

    assert device_provider.run.call_args.args[1] == ("b", "marqov")
    assert factory_provider.run.call_args.kwargs["s3_destination_folder"] == ("b", "marqov")


def test_device_braket_still_requires_a_bucket():
    device = MarqovDevice("sv1", {"backend": "sv1", "device_arn": SV1_ARN})
    with patch.object(MarqovDevice, "_get_provider_device", return_value=_offline_braket_provider()):
        with pytest.raises(ValueError, match="s3_bucket"):
            device.run(Circuit().x(0), shots=4)


# ---------------------------------------------------------------------------
# Azure: canonical keys reach the workspace, location defaults on both paths
# ---------------------------------------------------------------------------

AZURE_CANONICAL = {"subscription_id": "sub", "resource_group": "rg", "workspace_name": "ws"}


@pytest.fixture
def fake_workspace(monkeypatch):
    """Stub azure.quantum.Workspace, the provider SDK boundary of the Azure branch."""
    import sys
    import types

    captured: dict = {}

    class FakeWorkspace:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def get_targets(self, name=None):
            captured["target"] = name
            return object()

    module = types.ModuleType("azure.quantum")
    module.Workspace = FakeWorkspace
    monkeypatch.setitem(sys.modules, "azure.quantum", module)
    return captured


def test_device_azure_branch_reads_canonical_keys_and_defaults_location(fake_workspace):
    device = MarqovDevice("ionq.simulator", {"backend": "ionq.simulator", **AZURE_CANONICAL})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        device._get_provider_device()

    assert fake_workspace["kwargs"] == {
        "subscription_id": "sub", "resource_group": "rg", "name": "ws", "location": "eastus",
    }


def test_factory_azure_defaults_location_to_eastus():
    executor = ExecutorFactory.create_executor(
        "azure-ionq-simulator",
        {"provider": "Azure Quantum", "target": "ionq.simulator", **AZURE_CANONICAL},
    )
    assert executor.config.location == "eastus"


def test_factory_azure_accepts_deprecated_spellings_with_warning():
    with pytest.warns(DeprecationWarning, match="subscription_id"):
        executor = ExecutorFactory.create_executor(
            "azure-ionq-simulator",
            {"provider": "Azure Quantum", "target": "ionq.simulator",
             "azure_subscription_id": "sub", "azure_resource_group": "rg",
             "azure_workspace_name": "ws", "azure_location": "westus"},
        )
    assert (executor.config.subscription_id, executor.config.location) == ("sub", "westus")


@pytest.mark.parametrize("missing", ["subscription_id", "resource_group", "workspace_name"])
def test_factory_azure_still_requires_workspace_identity(missing):
    config = {"provider": "Azure Quantum", "target": "ionq.simulator", **AZURE_CANONICAL}
    del config[missing]
    with pytest.raises(ValueError, match=missing):
        ExecutorFactory.create_executor("azure-ionq-simulator", config)


# ---------------------------------------------------------------------------
# Criterion 3: optimization_level reaches the IBM preset pass manager
# ---------------------------------------------------------------------------


def _stub_sampler_v2(result):
    class _Job:
        def result(self):
            return result

    class _StubSamplerV2:
        def __init__(self, mode=None) -> None:
            pass

        def run(self, pubs, shots=None):
            return _Job()

    return _StubSamplerV2


def _pass_manager_level(params: dict) -> int:
    """Run the real IBM branch of MarqovDevice.run; return the level the PM was built with."""
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import preset_passmanagers

    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    result = StatevectorSampler().run([qc], shots=4).result()

    real = preset_passmanagers.generate_preset_pass_manager
    levels: list = []

    def recording_pm(*args, **kwargs):
        levels.append(kwargs.get("optimization_level", args[0] if args else None))
        return real(*args, **kwargs)

    device = MarqovDevice("ibm-fake-backend", {"backend": "ibm-fake-backend", **params})
    with (
        patch.object(MarqovDevice, "_get_provider_device",
                     return_value=GenericBackendV2(num_qubits=2, seed=7)),
        patch("qiskit_ibm_runtime.SamplerV2", _stub_sampler_v2(result)),
        patch.object(preset_passmanagers, "generate_preset_pass_manager", recording_pm),
    ):
        device.run(Circuit().x(0), shots=4)

    assert len(levels) == 1
    return levels[0]


@pytest.mark.parametrize(
    "params",
    [
        # All canonical spellings: before marqov-sdk#162 MarqovDevice could not even
        # route this dict to IBM.
        {"channel": "ibm_cloud", "token": "tok", "optimization_level": 3},
        # The issue's exact case: the level was silently dropped and the
        # circuit transpiled at level 1.
        {"ibm_token": "tok", "optimization_level": 3},
    ],
    ids=["canonical", "mixed"],
)
def test_optimization_level_reaches_ibm_pass_manager_through_device(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _pass_manager_level(params) == 3


def test_deprecated_ibm_optimization_level_still_reaches_pass_manager():
    with pytest.warns(DeprecationWarning, match="'ibm_optimization_level'.*'optimization_level'"):
        assert _pass_manager_level({"channel": "ibm_cloud", "ibm_optimization_level": 2}) == 2


def test_device_ibm_default_optimization_level_is_unchanged():
    assert _pass_manager_level({"channel": "ibm_cloud"}) == 1


@pytest.fixture
def fake_runtime(monkeypatch):
    import sys
    import types

    captured: dict = {}

    class FakeQiskitRuntimeService:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def backend(self, name):
            return object()

    module = types.ModuleType("qiskit_ibm_runtime")
    module.QiskitRuntimeService = FakeQiskitRuntimeService
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", module)
    return captured


def test_device_ibm_connection_reads_canonical_keys(fake_runtime):
    crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/abc::"
    device = MarqovDevice(
        "ibm_kingston",
        {"backend": "ibm_kingston", "channel": "ibm_cloud", "instance": crn, "token": "tok"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        device._get_provider_device()
    assert fake_runtime["kwargs"] == {"channel": "ibm_cloud", "instance": crn, "token": "tok"}


def test_factory_ibm_accepts_deprecated_spellings_with_warning():
    with pytest.warns(DeprecationWarning, match="'ibm_optimization_level'"):
        executor = ExecutorFactory.create_executor(
            "ibm-kingston",
            {"provider": "IBM Quantum", "backend_name": "ibm_kingston",
             "ibm_channel": "ibm_cloud", "ibm_token": "tok", "ibm_optimization_level": 3},
        )
    assert (executor.config.channel, executor.config.token, executor.config.optimization_level) == (
        "ibm_cloud", "tok", 3,
    )


# ---------------------------------------------------------------------------
# Criterion 4: conflicting spellings refuse instead of picking one
# ---------------------------------------------------------------------------


def test_conflicting_spellings_raise_naming_both_keys():
    with pytest.raises(ValueError, match=r"'ibm_channel'.*'channel'|'channel'.*'ibm_channel'"):
        _normalize({"ibm_channel": "ibm_cloud", "channel": "ibm_quantum_platform"})


def test_device_refuses_conflicting_spellings():
    with pytest.raises(ValueError, match=r"'ibm_channel'.*'channel'|'channel'.*'ibm_channel'"):
        MarqovDevice("ibm_kingston", {"backend": "ibm_kingston", "ibm_channel": "ibm_cloud",
                                      "channel": "ibm_quantum_platform"})


def test_factory_refuses_conflicting_spellings():
    with pytest.raises(ValueError, match=r"'azure_location'.*'location'|'location'.*'azure_location'"):
        ExecutorFactory.create_executor(
            "azure-ionq-simulator",
            {"provider": "Azure Quantum", "target": "ionq.simulator", **AZURE_CANONICAL,
             "location": "eastus", "azure_location": "westus"},
        )


# ---------------------------------------------------------------------------
# Deprecation warnings are attributed to the user's call, not to marqov
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["MarqovDevice", "get_device", "create_executor"])
def test_deprecation_warning_points_at_the_callers_line(entry):
    from marqov.device import get_device

    calls = {
        "MarqovDevice": lambda: MarqovDevice("ibm_k", {"backend": "ibm_k", "ibm_token": "t"}),
        "get_device": lambda: get_device({"backend": "ibm_k", "ibm_token": "t"}),
        "create_executor": lambda: ExecutorFactory.create_executor(
            "ibm-k", {"provider": "IBM Quantum", "ibm_token": "t"}),
    }
    with pytest.warns(DeprecationWarning, match="'ibm_token'") as record:
        calls[entry]()
    assert [Path(w.filename).name for w in record] == [Path(__file__).name]


# ---------------------------------------------------------------------------
# Criterion 5: the shared normaliser adds no edge toward marqov.platform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", ["marqov/device.py", "marqov/executors/_params.py"])
def test_device_and_params_never_import_platform(rel):
    path: Path = _ROOT / rel
    assert path.is_file(), f"{rel} is missing; the guard would pass vacuously"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = list(_platform_references(tree, _package_parts(path)))
    assert not offenders, f"{rel} references marqov.platform: {offenders}"
