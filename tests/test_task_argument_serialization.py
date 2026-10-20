"""Task arguments that are not JSON-native must reach the task body intact (marqov-sdk#142).

Results already travel in a ``{"__cloudpickle__": True, "data": ...}``
envelope. These tests pin the argument-side twin: a ``marqov.Circuit``, a
``numpy.ndarray`` or a namedtuple passed to a ``@task`` inside a ``@workflow``
survives Temporal's real data converter and arrives in the child as the same
type, while ``TaskProxy`` placeholders are left for ``prepare_node_inputs``.
The graph itself keeps argument objects as-is (the capture contract); the
encoding happens when the Temporal workflow input is built.
"""

from __future__ import annotations

import base64
import json
import threading
from collections import namedtuple
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import cloudpickle
import numpy as np
import pytest
from temporalio.converter import DataConverter

from marqov import Circuit, task, workflow
from marqov.workflows.decorators import _serialize_arg, _transport_arg
from marqov.workflows.graph import TaskProxy

Point = namedtuple("Point", ["x", "y"])


def _decode_leaf(value: Any) -> Any:
    """Decode the way the child does (``_task_child._deserialize_value``)."""
    from marqov.workflows._task_child import _deserialize_value

    return _deserialize_value(value)


# ---------------------------------------------------------------------------
# The package docstring's headline example, in the shape it is written.
# ``build`` is an ordinary function run at graph-build time; ``measure`` and
# ``compute`` are tasks, so ``measure`` receives a real ``Circuit`` argument.
# ---------------------------------------------------------------------------


def build(theta: float) -> Circuit:
    circuit = Circuit()
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    return circuit


@task
def measure(circuit, pauli):
    probs = np.asarray(circuit.simulate().probabilities())
    signs = {"Z": np.array([1.0, -1.0]), "I": np.array([1.0, 1.0])}
    weights = np.multiply.outer(signs[pauli[0]], signs[pauli[1]])
    return float((probs * weights).sum())


@task
def compute(z0, z1):
    return z0 + z1


@workflow
def vqe_step(theta):
    circuit = build(theta)
    z0 = measure(circuit, "ZI")  # Parallel
    z1 = measure(circuit, "IZ")  # Parallel
    return compute(z0, z1)


@task
def observe(circuit, array):
    return {
        "circuit": [type(circuit).__name__, circuit.num_qubits],
        "array": [type(array).__name__, array.dtype.str, list(array.shape)],
    }


@workflow
def circuit_and_array(circuit, array):
    return observe(circuit, array)


def _circuit() -> Circuit:
    circuit = Circuit()
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _first_node(workflow_input: dict[str, Any]) -> dict[str, Any]:
    return next(iter(workflow_input["nodes"].values()))


# ---------------------------------------------------------------------------
# Real Temporal data converter (acceptance criteria 1, 2, 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_headline_example_input_encodes_with_the_real_converter():
    workflow_input = vqe_step(0.5)._prepare_workflow_input()
    payloads = await DataConverter.default.encode([workflow_input])
    assert len(payloads) == 1


@pytest.mark.asyncio
async def test_circuit_and_ndarray_arguments_survive_the_real_converter():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    workflow_input = circuit_and_array(_circuit(), array)._prepare_workflow_input()

    payloads = await DataConverter.default.encode([workflow_input])
    (decoded,) = await DataConverter.default.decode(payloads, [dict])

    circuit_arg, array_arg = (_decode_leaf(a) for a in _first_node(decoded)["args"])
    assert isinstance(circuit_arg, Circuit)
    assert circuit_arg.num_qubits == 2
    assert isinstance(array_arg, np.ndarray)
    assert array_arg.dtype == np.float32
    assert array_arg.shape == (2, 3)
    np.testing.assert_array_equal(array_arg, array)


# ---------------------------------------------------------------------------
# The real child (acceptance criterion 3): what the task body observed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_task_body_observes_circuit_and_ndarray(mock_activity_ctx):
    from marqov.workflows.activity import _deserialize_value, execute_task

    array = np.zeros((3, 4), dtype=np.int16)
    node = _first_node(circuit_and_array(_circuit(), array)._prepare_workflow_input())

    result_json = await execute_task(
        node["node_id"],
        node["func_ref"],
        json.dumps(node["args"]),
        json.dumps(node["kwargs"]),
    )

    seen = _deserialize_value(json.loads(result_json)["result"])
    assert seen == {
        "circuit": ["Circuit", 2],
        "array": ["ndarray", np.dtype(np.int16).str, [3, 4]],
    }


@pytest.fixture
def mock_activity_ctx():
    with patch("marqov.workflows.activity.activity") as mock_act:
        mock_act.heartbeat = lambda *a, **kw: None
        mock_act.logger = __import__("logging").getLogger("test")
        yield mock_act


# ---------------------------------------------------------------------------
# Containers, namedtuples and proxies (acceptance criteria 4, 5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_namedtuple_argument_round_trips_as_the_same_type():
    @task
    def takes_point(point):
        return point

    @workflow
    def program():
        return takes_point(Point(1, 2.5))

    dispatch = program()
    (graph_arg,) = next(iter(dispatch.graph.nodes.values())).args
    assert type(graph_arg) is Point

    payloads = await DataConverter.default.encode([dispatch._prepare_workflow_input()])
    (decoded,) = await DataConverter.default.decode(payloads, [dict])
    restored = _decode_leaf(_first_node(decoded)["args"][0])

    assert type(restored) is Point
    assert restored == Point(1, 2.5)


def test_namedtuple_holding_a_proxy_keeps_its_type_and_placeholder():
    @task
    def source():
        return 1

    @task
    def sink(point):
        return point

    @workflow
    def program():
        return sink(Point(source(), 3))

    dispatch = program()
    upstream, downstream = dispatch.graph.nodes.values()
    placeholder = {"__proxy__": True, "node_id": upstream.id}
    (point,) = downstream.args
    assert type(point) is Point
    assert point == (placeholder, 3)
    # On the wire the placeholder must stay resolvable, so the namedtuple is
    # sent as a list (as a plain tuple is), not pickled around the placeholder.
    sent = dispatch._prepare_workflow_input()["nodes"][downstream.id]["args"]
    assert sent == [[placeholder, 3]]


def test_plain_values_keep_their_json_shape_on_the_wire():
    assert _transport_arg([1, "a", None]) == [1, "a", None]
    assert _transport_arg((1, 2)) == [1, 2]
    assert _transport_arg({"k": [1.5, True]}) == {"k": [1.5, True]}
    assert _transport_arg(np.float64(0.5)) == 0.5 and type(_transport_arg(np.float64(0.5))) is float
    assert _transport_arg(np.int64(3)) == 3 and type(_transport_arg(np.int64(3))) is int
    assert _transport_arg(np.bool_(True)) is True


def test_graph_keeps_plain_tuples_and_opaque_objects_as_is():
    array = np.zeros(2)
    assert type(_serialize_arg((1, 2))) is tuple
    assert _serialize_arg([array])[0] is array


def test_marker_dicts_pass_through_unchanged():
    envelope = {"__cloudpickle__": True, "data": "abc"}
    placeholder = {"__proxy__": True, "node_id": "n1"}
    assert _transport_arg(envelope) is envelope
    assert _transport_arg(placeholder) is placeholder


def test_task_proxy_is_never_cloudpickled_even_when_nested():
    @task
    def source():
        return 1

    @task
    def sink(*args, **kwargs):
        return args, kwargs

    @workflow
    def program():
        up = source()
        assert isinstance(up, TaskProxy)
        return sink(up, [up, np.zeros(2)], {"inner": up}, kw=up, nested={"list": [up]})

    dispatch = program()
    upstream, downstream = dispatch.graph.nodes.values()
    placeholder = {"__proxy__": True, "node_id": upstream.id}
    sent = dispatch._prepare_workflow_input()["nodes"][downstream.id]

    for args, kwargs in ((downstream.args, downstream.kwargs), (sent["args"], sent["kwargs"])):
        direct, in_list, in_dict = args
        assert direct == placeholder
        assert in_list[0] == placeholder
        assert in_dict == {"inner": placeholder}
        assert kwargs == {"kw": placeholder, "nested": {"list": [placeholder]}}

    # Only the array beside the proxy is pickled, and the pickle holds no proxy.
    blob = sent["args"][1][1]
    assert set(blob) == {"__cloudpickle__", "data"}
    assert not isinstance(cloudpickle.loads(base64.b64decode(blob["data"])), TaskProxy)
    assert "__cloudpickle__" not in json.dumps([sent["args"][0], sent["args"][2], sent["kwargs"]])


# ---------------------------------------------------------------------------
# Attributable failure (acceptance criterion 6)
# ---------------------------------------------------------------------------


def test_unpicklable_argument_names_task_and_position():
    @task
    def uses_lock(value, lock):
        return value

    @workflow
    def program():
        return uses_lock(1, threading.Lock())

    dispatch = program()
    with pytest.raises(TypeError) as excinfo:
        dispatch._prepare_workflow_input()
    message = str(excinfo.value)
    assert "uses_lock" in message
    assert "positional argument 1" in message
    assert "lock" in message.lower()


def test_unpicklable_keyword_argument_names_the_keyword():
    @task
    def uses_lock(value, lock=None):
        return value

    @workflow
    def program():
        return uses_lock(1, lock=[threading.Lock()])

    with pytest.raises(TypeError, match=r"uses_lock.*keyword argument 'lock'"):
        program()._prepare_workflow_input()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run", "start_with_ids", "run_with_ids", "dispatch"])
async def test_unpicklable_argument_is_never_submitted(method):
    @task
    def uses_lock(lock):
        return 1

    @workflow
    def program():
        return uses_lock(threading.Lock())

    client = MagicMock()
    client.start_workflow = AsyncMock()
    with pytest.raises(TypeError, match=r"uses_lock.*positional argument 0"):
        await getattr(program(), method)(client)
    client.start_workflow.assert_not_called()


# ---------------------------------------------------------------------------
# End to end through a real Temporal worker (the documented example)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_headline_example_dispatches_end_to_end_with_no_worker_loads():
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from marqov.workflows.activity import _deserialize_value, execute_task, prepare_node_inputs
    from marqov.workflows.temporal_workflow import JobWorkflow

    theta = 0.5
    expected = 2 * float(np.cos(theta))  # <ZI> == <IZ> == cos(theta) after RY + CX

    loads_calls: list[bytes] = []
    real_loads = cloudpickle.loads

    def spy_loads(data: bytes, *args: Any, **kwargs: Any) -> Any:
        loads_calls.append(data[:16])
        return real_loads(data, *args, **kwargs)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="marqov-workflows",
            workflows=[JobWorkflow],
            activities=[execute_task, prepare_node_inputs],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            # The documented call, decoded client side.
            value = await vqe_step(theta).run(env.client)

            # The same graph with a loads spy over the whole worker run: Circuit
            # arguments are unpickled only inside the task children.
            with patch.object(cloudpickle, "loads", side_effect=spy_loads):
                handle, _, _ = await vqe_step(theta).start_with_ids(env.client)
                raw = await handle.result()

    assert value == pytest.approx(expected)
    assert _deserialize_value(json.loads(raw)["result"]) == pytest.approx(expected)
    assert loads_calls == []


@pytest.mark.asyncio
async def test_argument_over_the_size_cap_names_task_and_argument_and_is_never_submitted():
    from marqov.workflows.activity import MAX_RESULT_BYTES

    @task
    def takes_blob(value, blob=None):
        return 1

    # A str encodes to its length plus two quote bytes.
    at_limit = "x" * (MAX_RESULT_BYTES - 2)
    just_over = "x" * (MAX_RESULT_BYTES - 1)

    @workflow
    def fits():
        return takes_blob(1, blob=at_limit)

    @workflow
    def too_big_positional():
        return takes_blob(just_over)

    @workflow
    def too_big_keyword():
        return takes_blob(1, blob=just_over)

    fits()._prepare_workflow_input()

    with pytest.raises(ValueError, match=r"takes_blob.*keyword argument 'blob'.*over the"):
        too_big_keyword()._prepare_workflow_input()

    client = MagicMock()
    client.start_workflow = AsyncMock()
    with pytest.raises(ValueError, match=r"takes_blob.*positional argument 0.*over the"):
        await too_big_positional().run(client)
    client.start_workflow.assert_not_called()
