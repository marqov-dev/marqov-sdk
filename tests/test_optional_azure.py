"""azure-quantum is an extra: core installs lack it and name the extra to install."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from marqov import _optional

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
AZURE_REQUIREMENT = "azure-quantum[qiskit,cirq]>=3.12.0,<3.13.0"

MISSING_AZURE = '''
import asyncio
import importlib.abc
import sys

class MissingAzure(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {blocked!r} or fullname.startswith({blocked!r} + "."):
            raise ModuleNotFoundError("deliberately absent", name={blocked!r})

sys.meta_path.insert(0, MissingAzure())
from marqov import Circuit, MarqovDevice
from marqov.executors import AzureQuantumExecutor, AzureQuantumExecutorConfig

config = AzureQuantumExecutorConfig(
    subscription_id="unused", resource_group="unused", workspace_name="unused",
    location="eastus", target="ionq.simulator",
)
device = MarqovDevice("ionq.simulator", {{
    "azure_subscription_id": "unused",
    "azure_resource_group": "unused",
    "azure_workspace_name": "unused",
}})
for action in (
    lambda: device._get_provider_device(),
    lambda: asyncio.run(AzureQuantumExecutor(config).execute(Circuit().h(0), shots=1)),
):
    try:
        action()
    except ImportError as error:
        assert str(error) == 'Azure Quantum requires pip install "marqov[azure]"', str(error)
    else:
        raise AssertionError("Missing provider dependency was accepted")
'''


# "azure" is a namespace package: azure-core or azure-identity can provide it
# while azure-quantum is absent, so both missing names must reach the message.
@pytest.mark.parametrize("blocked", ["azure", "azure.quantum"])
def test_azure_paths_without_the_package_name_the_extra(blocked):
    result = subprocess.run(
        [sys.executable, "-c", MISSING_AZURE.format(blocked=blocked)], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_broken_azure_installation_is_not_reported_as_an_absent_extra(monkeypatch):
    def broken(module):
        raise ModuleNotFoundError("vendor dependency absent", name="azure_vendor_dep")

    monkeypatch.setattr(_optional, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as error:
        _optional.require_azure()
    assert error.value.name == "azure_vendor_dep"


def test_core_dependencies_exclude_azure_quantum():
    names = [re.split(r"[\[<>=!~ ;]", dep)[0] for dep in PROJECT["dependencies"]]
    assert "azure-quantum" not in names


@pytest.mark.parametrize("extra", ["azure", "all"])
def test_extras_stay_below_the_cryptography_capped_release(extra):
    assert AZURE_REQUIREMENT in PROJECT["optional-dependencies"][extra]
