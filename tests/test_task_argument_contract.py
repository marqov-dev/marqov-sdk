"""Behaviour of task arguments that a fix for marqov-sdk#314 must not change.

marqov-sdk#314: inside a ``@workflow``, a task's arguments are read when the
workflow is dispatched or captured, so changing an object after passing it to
a task changes what that task receives. Every test here passes today and
describes something a fix has to keep:

* ``capture()`` hands adapters the caller's own objects (the capture contract
  in ``docs/workflow-capture.md``), including for an argument cloudpickle
  cannot encode.
* Building, capturing and preparing a workflow never load a pickle in the
  caller's process, and leave the caller's classes untouched.
* The Temporal workflow input keeps its present shapes: a tuple travels as a
  list, a NumPy float as a float, and the same object passed twice as two
  separate values.
* Plain lists and dicts are rebuilt at the task call, so a later ``append``
  does not reach the task.

The tests that show the bug itself live with the fix.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cloudpickle
import numpy as np
import pytest

from marqov import Circuit, task, workflow
from marqov.workflows.graph import TaskNode

REPO_ROOT = Path(__file__).resolve().parents[1]


def _decode(value: Any) -> Any:
    """Decode one encoded argument the way the task child does."""
    from marqov.workflows._task_child import _deserialize_value

    return _deserialize_value(value)


def _nodes(dispatch: Any) -> list[TaskNode]:
    return list(dispatch.graph.nodes.values())


def _wire(dispatch: Any) -> list[dict[str, Any]]:
    return list(dispatch._prepare_workflow_input()["nodes"].values())


@task
def measure(circuit, reference=None, axes=None, prior=None):
    return None


@task
def take(value, other=None):
    return None


# ---------------------------------------------------------------------------
# capture(): adapters receive the caller's own objects
# ---------------------------------------------------------------------------


def test_capture_exposes_the_callers_own_objects():
    circuit = Circuit().h(0)
    array = np.zeros(3)

    @workflow
    def program():
        return [measure(circuit, reference=circuit), take([array], other={"a": array})]

    capture = program().capture()
    first, second = _nodes(capture)

    assert first.args[0] is circuit
    assert first.kwargs["reference"] is circuit
    assert second.args[0][0] is array
    assert second.kwargs["other"]["a"] is array


def test_capture_accepts_an_argument_cloudpickle_cannot_encode():
    """An adapter may bring its own codec; only Temporal dispatch refuses the lock."""
    lock = threading.Lock()

    @workflow
    def program():
        return take(lock)

    dispatch = program()
    (node,) = _nodes(dispatch.capture())
    assert node.args[0] is lock

    with pytest.raises(TypeError, match="Task 'take': positional argument 0"):
        dispatch._prepare_workflow_input()


# ---------------------------------------------------------------------------
# The caller's process: nothing is unpickled, no class is rewritten
# ---------------------------------------------------------------------------


def test_build_capture_and_prepare_never_load_a_pickle_in_the_caller():
    circuit = Circuit().h(0)

    @workflow
    def program():
        return measure(circuit, reference=circuit)

    loads_calls: list[bytes] = []
    real_loads = cloudpickle.loads

    def spy_loads(data: bytes, *args: Any, **kwargs: Any) -> Any:
        loads_calls.append(data[:16])
        return real_loads(data, *args, **kwargs)

    # A spy, not an exception: a caller that caught the exception would hide the load.
    with patch.object(cloudpickle, "loads", side_effect=spy_loads):
        dispatch = program()
        dispatch.capture()
        dispatch._prepare_workflow_input()

    assert loads_calls == []


_CALLER_CLASS_SCRIPT = textwrap.dedent(
    """
    import dataclasses, json
    from marqov import task, workflow

    SCALE = 1
    REGISTRY = []

    @dataclasses.dataclass
    class Model:
        theta: float
        seen = REGISTRY
        def scaled(self):
            return self.theta * SCALE

    @task
    def fit(model, also=None):
        return model.theta

    model = Model(2.0)
    method_before = Model.__dict__["scaled"]

    @workflow
    def program():
        return fit(model, also=[model])

    dispatch = program()
    dispatch.capture()
    dispatch._prepare_workflow_input()

    SCALE = 10
    REGISTRY.append("x")
    print(json.dumps({
        "method_is_the_one_written": Model.__dict__["scaled"] is method_before,
        "method_globals_are_the_module": Model.scaled.__globals__ is globals(),
        "scaled": model.scaled(),
        "class_attribute_is_shared": Model.seen is REGISTRY,
    }))
    """
)


def test_a_class_defined_in_main_is_untouched_by_build_capture_and_prepare(tmp_path):
    """A script or notebook class is pickled by value; loading it back in the same
    process would rewrite the live class (cloudpickle issue 439)."""
    script = tmp_path / "caller_class.py"
    script.write_text(_CALLER_CLASS_SCRIPT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), *sys.path])}

    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120
    )

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {
        "method_is_the_one_written": True,
        "method_globals_are_the_module": True,
        "scaled": 20.0,
        "class_attribute_is_shared": True,
    }


# ---------------------------------------------------------------------------
# The Temporal workflow input keeps its present shapes
# ---------------------------------------------------------------------------


def test_temporal_input_sends_a_tuple_as_a_list_and_a_numpy_float_as_a_float():
    @workflow
    def program():
        return [
            measure(1, axes=("X", "Z")),
            measure(1, axes=["X", "Z"]),
            take(np.float64(0.5)),
        ]

    dispatch = program()
    tuple_call, list_call, scalar_call = _wire(dispatch)

    # The graph keeps what the caller passed.
    assert [type(node.kwargs.get("axes")) for node in _nodes(dispatch)[:2]] == [tuple, list]
    # On the wire the two are the same JSON list.
    assert tuple_call["kwargs"]["axes"] == ["X", "Z"]
    assert list_call["kwargs"]["axes"] == ["X", "Z"]
    assert type(scalar_call["args"][0]) is float


def test_temporal_input_sends_the_same_object_passed_twice_as_two_values():
    array = np.zeros(2)

    @workflow
    def program():
        return take(array, other=array)

    (node,) = _wire(program())
    first, second = _decode(node["args"][0]), _decode(node["kwargs"]["other"])

    np.testing.assert_array_equal(first, array)
    np.testing.assert_array_equal(second, array)
    assert first is not second


def test_temporal_input_keeps_another_tasks_result_as_a_dependency():
    circuit = Circuit().h(0)

    @workflow
    def program():
        earlier = measure(circuit)
        return measure(circuit, prior=earlier)

    dispatch = program()
    earlier, later = _nodes(dispatch)
    sent = _wire(dispatch)[1]

    assert later.dependencies == [earlier.id]
    assert sent["kwargs"]["prior"] == {"__proxy__": True, "node_id": earlier.id}


def test_preparing_the_same_dispatch_twice_gives_the_same_input():
    circuit = Circuit().h(0)
    array = np.arange(4.0)

    @workflow
    def program():
        return measure(circuit, reference=circuit, axes=("X", "Z"), prior=take(array))

    dispatch = program()

    assert dispatch._prepare_workflow_input() == dispatch._prepare_workflow_input()


# ---------------------------------------------------------------------------
# Plain containers are rebuilt at the task call
# ---------------------------------------------------------------------------


def test_a_later_change_to_a_plain_list_or_dict_does_not_reach_the_task():
    values = [1, 2]
    options = {"shots": 100}

    @workflow
    def program():
        recorded = take(values, other=options)
        values.append(3)
        values[0] = 99
        options["shots"] = 200
        return recorded

    dispatch = program()
    (node,) = _nodes(dispatch)
    (sent,) = _wire(dispatch)

    assert node.args[0] == [1, 2]
    assert node.kwargs["other"] == {"shots": 100}
    assert sent["args"][0] == [1, 2]
    assert sent["kwargs"]["other"] == {"shots": 100}
