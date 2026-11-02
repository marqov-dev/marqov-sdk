"""Circuit interop without the framework installed names the marqov extra to install."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module, action, message",
    [
        ("qiskit", "Circuit().h(0).to_qiskit()", 'Qiskit interop requires pip install "marqov[qiskit]"'),
        ("cirq", "Circuit().h(0).to_cirq()", 'Cirq interop requires pip install "marqov[cirq]"'),
    ],
)
def test_export_without_framework_names_the_extra(module, action, message):
    code = f'''
import importlib.abc
import sys

class Missing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {module!r} or fullname.startswith({module + "."!r}):
            raise ModuleNotFoundError("deliberately absent", name={module!r})

sys.meta_path.insert(0, Missing())
from marqov import Circuit

try:
    {action}
except ImportError as error:
    assert str(error) == {message!r}, str(error)
else:
    raise AssertionError("Missing framework was accepted")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
