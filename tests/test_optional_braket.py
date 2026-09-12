"""Core workflow imports must not depend on the optional AWS provider SDK."""

import subprocess
import sys
from pathlib import Path


def test_core_capture_and_provider_refusal_without_braket():
    code = '''
import importlib.abc
import sys

class MissingBraket(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "braket" or fullname.startswith("braket."):
            raise ModuleNotFoundError("Braket deliberately absent", name="braket")

sys.meta_path.insert(0, MissingBraket())
from marqov import task, workflow, Circuit, LocalExecutor, MarqovDevice
from marqov.executors import BraketExecutor, BraketExecutorConfig

@task
def twice(value):
    return value * 2

@workflow
def example(value):
    return {"answer": twice(value), "labels": ["managed", "native"]}

captured = example(21).capture()
assert len(captured.graph.nodes) == 1
node_id = next(iter(captured.graph.nodes))
assert captured.resolve_output({node_id: 42}) == {
    "answer": 42, "labels": ["managed", "native"]}
assert LocalExecutor is not None and Circuit is not None
for action in (
    lambda: BraketExecutor(BraketExecutorConfig(device_arn="unused", s3_bucket="unused")),
    lambda: Circuit().h(0).to_braket(),
    lambda: Circuit.from_braket(object()),
    lambda: MarqovDevice("local", {})._get_provider_device(),
    lambda: MarqovDevice("marqov-sim", {})._get_provider_device(),
):
    try:
        action()
    except ImportError as error:
        assert str(error) == 'AWS Braket requires pip install "marqov[braket]"'
    else:
        raise AssertionError("Missing provider dependency was accepted")
assert not any(name == "braket" or name.startswith("braket.") for name in sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_broken_provider_installation_is_not_reported_as_an_absent_extra():
    code = '''
import importlib.abc
import sys

class BrokenBraket(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "braket":
            raise ModuleNotFoundError("Broken transitive dependency", name="broken_transitive")

sys.meta_path.insert(0, BrokenBraket())
try:
    import marqov
except ModuleNotFoundError as error:
    assert error.name == "broken_transitive"
else:
    raise AssertionError("Broken provider installation was silently hidden")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
