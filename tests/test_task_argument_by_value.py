"""A task receives its arguments as they were when it was called (marqov-sdk#314).

Acceptance tests for the fix. They fail on the code they were written against:
inside a ``@workflow`` a task's arguments are read at dispatch, so a change
made after the call reaches the earlier task.

The sweep is the shared acceptance case for both runners. One circuit is
passed twice in one call (``measure(c, reference=c)``), changed, passed again,
and changed once more after the last recorded call, so the whole interval up
to dispatch is covered.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from marqov import Circuit, task, workflow


def _decode(value: Any) -> Any:
    from marqov.workflows._task_child import _deserialize_value

    return _deserialize_value(value)


def _gates(circuit: Circuit) -> list[str]:
    return [gate["gate"] for gate in circuit.to_dict()["gates"]]


@task
def measure(circuit, reference=None, axes=None, prior=None):
    def gates(c):
        return [gate["gate"] for gate in c.to_dict()["gates"]]

    return {
        "circuit": gates(circuit),
        "reference": None if reference is None else gates(reference),
        "same": circuit is reference,
        "axes": None if axes is None else [type(axes).__name__, list(axes)],
        "prior": prior,
    }


@task
def total(values):
    return values.tolist()


@task
def first_of(items):
    return items[0].tolist()


@task
def total_of(values, options=None):
    return None


@task
def kind_of(value):
    return type(value).__name__


@workflow
def sweep():
    c = Circuit().h(0)
    first = measure(c, reference=c, axes=("X", "Z"))
    c.rx(1.5, 0)
    second = measure(c, reference=c, axes=["X", "Z"])
    c.rx(3.0, 0)
    third = measure(c, reference=c, prior=first)
    c.z(0)  # after the last recorded call

    array = np.zeros(3)
    before = total(array)
    array += 1
    after = total(array)
    array[:] = 100  # after the last recorded call

    inner = np.zeros(2)
    nested = first_of([inner])
    inner[0] = 9

    scalar = kind_of(np.float64(0.5))

    return [first, second, third, before, after, nested, scalar]


LABELS = ["first", "second", "third", "before", "after", "nested", "scalar"]
FIRST = ["H"]
SECOND = ["H", "Rx"]
THIRD = ["H", "Rx", "Rx"]


@pytest.mark.asyncio
async def test_each_task_receives_its_arguments_as_they_were_at_the_call():
    """Through the real workflow, activities and task child processes."""
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from marqov.workflows.activity import execute_task, prepare_node_inputs
    from marqov.workflows.temporal_workflow import JobWorkflow

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="marqov-workflows",
            workflows=[JobWorkflow],
            activities=[execute_task, prepare_node_inputs],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            dispatch = sweep()
            by_node = await dispatch.run(env.client)

    # run() returns results by node id; nodes are recorded in call order.
    result = dict(zip(LABELS, (by_node[node_id] for node_id in dispatch.graph.nodes)))

    assert result["first"]["circuit"] == FIRST
    assert result["first"]["reference"] == FIRST
    assert result["second"]["circuit"] == SECOND
    assert result["second"]["reference"] == SECOND
    assert result["third"]["circuit"] == THIRD
    assert result["third"]["reference"] == THIRD
    # The earlier task's result, as that task produced it.
    assert result["third"]["prior"]["circuit"] == FIRST

    assert result["before"] == [0.0, 0.0, 0.0]
    assert result["after"] == [1.0, 1.0, 1.0]
    assert result["nested"] == [0.0, 0.0]

    # The Temporal path's present conversions are unchanged: a tuple arrives as a
    # list, a NumPy float as a float, and one object passed twice as two objects.
    assert result["first"]["axes"] == ["list", ["X", "Z"]]
    assert result["second"]["axes"] == ["list", ["X", "Z"]]
    assert result["scalar"] == "float"
    assert [result[label]["same"] for label in ("first", "second", "third")] == [False] * 3


def test_a_change_after_the_workflow_is_built_does_not_reach_the_prepared_input():
    """One dispatch handle, prepared twice, with the caller's objects changed in between."""
    circuit = Circuit().h(0)
    array = np.zeros(3)

    @workflow
    def program(c, values):
        return [measure(c, reference=c), total(values)]

    dispatch = program(circuit, array)

    def sent() -> list[Any]:
        measured, totalled = dispatch._prepare_workflow_input()["nodes"].values()
        return [
            _gates(_decode(measured["args"][0])),
            _gates(_decode(measured["kwargs"]["reference"])),
            _decode(totalled["args"][0]).tolist(),
        ]

    circuit.x(0)
    array += 1
    once = sent()
    circuit.z(0)
    array += 1
    twice = sent()

    assert once == [["H"], ["H"], [0.0, 0.0, 0.0]]
    assert twice == once


def test_changing_a_prepared_input_does_not_change_the_next_one():
    @workflow
    def program():
        return total_of([1, 2], options={"shots": 10})

    dispatch = program()
    (first,) = dispatch._prepare_workflow_input()["nodes"].values()
    first["args"][0].append("deep")
    first["args"].append("injected")
    first["kwargs"]["options"]["extra"] = 1

    (second,) = dispatch._prepare_workflow_input()["nodes"].values()

    assert second["args"] == [[1, 2]]
    assert second["kwargs"] == {"options": {"shots": 10}}


def test_an_argument_that_cannot_be_encoded_fails_at_dispatch_the_same_way_each_time():
    import gc
    import threading
    import traceback
    import weakref

    class Big:
        pass

    def build():
        unrelated = Big()  # a local the task never receives

        @workflow
        def program():
            return total_of(threading.Lock())

        return program(), weakref.ref(unrelated)

    dispatch, unrelated = build()
    gc.collect()
    # The recorded error does not keep the call's frames, and so its locals, alive.
    assert unrelated() is None

    depths = []
    for _ in range(3):
        with pytest.raises(TypeError, match="Task 'total_of': positional argument 0") as caught:
            dispatch._prepare_workflow_input()
        depths.append(len(traceback.extract_tb(caught.value.__traceback__)))
    assert len(set(depths)) == 1


def test_the_same_array_passed_to_many_tasks_is_held_once():
    array = np.zeros(1000)

    @workflow
    def program():
        return [total(array) for _ in range(5)]

    nodes = list(program().graph.nodes.values())
    payloads = {id(node.snapshot.value[0][0]["data"]) for node in nodes}

    assert len(payloads) == 1


# ---------------------------------------------------------------------------
# A recorded failure is text: no exception object, frame or caller object is kept
# ---------------------------------------------------------------------------


class _Big:
    pass


def test_a_recorded_failure_is_independent_on_every_dispatch():
    """An exception that cannot be copied, edited by the first handler."""
    import traceback

    class BadCopyError(Exception):
        def __copy__(self):
            raise RuntimeError("cannot copy")

    class BadDict:
        @property
        def __dict__(self):
            raise BadCopyError("original")

    @workflow
    def program():
        return total_of(BadDict())

    dispatch = program()
    seen, depths, messages = [], [], []
    for attempt in range(3):
        with pytest.raises(Exception) as caught:
            dispatch._prepare_workflow_input()
        seen.append(caught.value)
        depths.append(len(traceback.extract_tb(caught.value.__traceback__)))
        messages.append(str(caught.value))
        if attempt == 0:
            caught.value.args = ("handler changed it",)

    assert len({id(error) for error in seen}) == 3
    assert len(set(depths)) == 1
    assert messages[1] == messages[2] == messages[0]
    assert "original" in messages[0]


def test_an_exception_group_from_a_reducer_keeps_no_frame_alive():
    import gc
    import weakref

    refs = []

    class GroupArg:
        def __reduce__(self):
            unrelated = _Big()
            refs.append(weakref.ref(unrelated))
            try:
                raise ValueError("leaf")
            except ValueError as error:
                leaf = error
            raise ExceptionGroup("group", [leaf]) from None

    @workflow
    def program():
        return total_of(GroupArg())

    dispatch = program()
    gc.collect()
    assert refs[-1]() is None

    with pytest.raises(Exception):
        dispatch._prepare_workflow_input()
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_a_handler_that_reraises_the_cause_pins_nothing_on_the_graph():
    import gc
    import threading
    import weakref

    @workflow
    def program():
        return total_of(threading.Lock())

    dispatch = program()

    def handle():
        sentinel = _Big()
        ref = weakref.ref(sentinel)
        try:
            dispatch._prepare_workflow_input()
        except Exception as error:
            try:
                raise error.__cause__ or error
            except Exception:
                pass
        return ref

    ref = handle()
    gc.collect()
    assert ref() is None


def test_the_size_limit_in_force_at_dispatch_is_the_one_applied(monkeypatch):
    from marqov.workflows import activity

    @workflow
    def program():
        return total_of("x" * 200)

    monkeypatch.setattr(activity, "MAX_RESULT_BYTES", 10_000)
    built_under_high = program()
    monkeypatch.setattr(activity, "MAX_RESULT_BYTES", 100)
    built_under_low = program()
    with pytest.raises(ValueError, match="over the 100-byte limit"):
        built_under_high._prepare_workflow_input()

    monkeypatch.setattr(activity, "MAX_RESULT_BYTES", 10_000)
    built_under_low._prepare_workflow_input()


# ---------------------------------------------------------------------------
# argument_capture(): one encoder per task call, chosen before the body runs
# ---------------------------------------------------------------------------


class _Counter:
    """Its encoding has a side effect: each reduce returns the next number."""

    def __init__(self):
        self.n = 0

    def __reduce__(self):
        self.n += 1
        return (int, (self.n,))


def test_a_custom_encoder_runs_once_per_call_and_the_temporal_encoder_does_not_run():
    import cloudpickle

    from marqov.workflows import argument_capture

    counter = _Counter()
    calls = []

    def encoder(call):
        calls.append(call)
        return cloudpickle.dumps((call.args, call.kwargs))

    @workflow
    def program():
        first = total_of(counter)
        return total_of(counter, options={"prior": first})

    with argument_capture(encoder, name="test/1"):
        dispatch = program()

    first, second = dispatch.graph.nodes.values()
    assert counter.n == 2  # one reduce per call: the Temporal encoder did not also run
    assert [call.node_id for call in calls] == [first.id, second.id]
    assert all(call.graph is dispatch.graph for call in calls)
    assert first.snapshot.encoder == "test/1" and first.snapshot.status == "ok"
    assert isinstance(first.snapshot.value, bytes)
    assert second.snapshot.dependencies == (first.id,)
    assert calls[1].kwargs["options"]["prior"] == {"__proxy__": True, "node_id": first.id}
    assert cloudpickle.loads(first.snapshot.value)[0] == [1]
    assert cloudpickle.loads(second.snapshot.value)[0] == [2]

    # capture() is not affected and does not run the encoder again.
    dispatch.capture()
    assert len(calls) == 2 and counter.n == 2


def test_the_default_encoder_runs_once_per_call_and_not_again_at_dispatch():
    counter = _Counter()

    @workflow
    def program():
        return total_of(counter)

    dispatch = program()
    assert counter.n == 1
    (sent,) = dispatch._prepare_workflow_input()["nodes"].values()
    dispatch._prepare_workflow_input()
    dispatch.capture()

    assert counter.n == 1
    assert _decode(sent["args"][0]) == 1


def test_a_custom_encoders_failure_is_recorded_as_text_and_not_raised_into_the_body():
    from marqov.workflows import argument_capture

    def encoder(call):
        raise OverflowError("budget spent")

    reached_end = []

    @workflow
    def program():
        recorded = total_of(1)
        reached_end.append(True)
        return recorded

    with argument_capture(encoder, name="test/1"):
        dispatch = program()

    (node,) = dispatch.graph.nodes.values()
    assert reached_end == [True]
    assert (node.snapshot.status, node.snapshot.error_type, node.snapshot.error_message) == (
        "failed",
        "OverflowError",
        "budget spent",
    )
    assert node.snapshot.value is None
    assert node.args == [1]  # the live view is still there for the consumer to inspect


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_in_an_encoder_is_not_recorded(stop):
    from marqov.workflows import argument_capture

    def encoder(call):
        raise stop()

    @workflow
    def program():
        return total_of(1)

    with argument_capture(encoder, name="test/1"), pytest.raises(stop):
        program()


def test_a_workflow_recorded_by_another_encoder_or_none_is_refused_by_temporal_dispatch():
    from marqov.workflows import argument_capture

    @workflow
    def program():
        return total_of(Circuit().h(0))

    with argument_capture(lambda call: b"x", name="test/1"):
        custom = program()
    with argument_capture(None):
        off = program()

    with pytest.raises(RuntimeError, match="recorded by the argument encoder 'test/1'"):
        custom._prepare_workflow_input()
    with pytest.raises(RuntimeError, match="argument capture switched off"):
        off._prepare_workflow_input()

    (node,) = off.graph.nodes.values()
    assert node.snapshot.status == "absent"
    # Outside the block the default is back.
    program()._prepare_workflow_input()


def test_argument_capture_needs_its_own_name():
    from marqov.workflows import argument_capture
    from marqov.workflows.graph import TEMPORAL_ARGUMENT_ENCODER

    for name in (None, "", TEMPORAL_ARGUMENT_ENCODER):
        with pytest.raises(ValueError), argument_capture(lambda call: None, name=name):
            pass


def test_a_node_built_by_hand_or_reset_is_encoded_at_dispatch():
    @workflow
    def program():
        return total_of([1, 2])

    dispatch = program()
    (node,) = dispatch.graph.nodes.values()
    node.args[0].append(3)
    (unchanged,) = dispatch._prepare_workflow_input()["nodes"].values()
    node.snapshot = None
    (rebuilt,) = dispatch._prepare_workflow_input()["nodes"].values()

    assert unchanged["args"] == [[1, 2]]
    assert rebuilt["args"] == [[1, 2, 3]]


def test_dependencies_changed_after_the_call_are_refused():
    @workflow
    def program():
        return total_of(total_of(1))

    dispatch = program()
    later = list(dispatch.graph.nodes.values())[1]
    later.dependencies.clear()

    with pytest.raises(RuntimeError, match="dependencies were changed"):
        dispatch._prepare_workflow_input()


# ---------------------------------------------------------------------------
# Encoding must not change the workflow, another task's argument, or the caller
# ---------------------------------------------------------------------------


@task
def helper(value):
    return value + 1


class _CallsATask:
    """Its reducer calls a decorated task, as ordinary code outside a workflow may."""

    def __reduce__(self):
        return (int, (helper(6),))


def test_a_reducer_that_calls_a_task_runs_it_and_adds_no_node():
    @workflow
    def program():
        return total_of(_CallsATask())

    dispatch = program()
    (node,) = dispatch.graph.nodes.values()
    (sent,) = dispatch._prepare_workflow_input()["nodes"].values()

    assert node.func_name == "total_of"
    assert _decode(sent["args"][0]) == 7


def test_a_custom_encoder_that_calls_a_task_adds_no_node():
    from marqov.workflows import argument_capture

    def encoder(call):
        return helper(1)

    @workflow
    def program():
        return total_of(1)

    with argument_capture(encoder, name="test/1"):
        dispatch = program()

    (node,) = dispatch.graph.nodes.values()
    assert node.snapshot.value == 2


class _CaseBlind(str):
    def __eq__(self, other):
        return str.lower(self) == str.lower(other)

    def __hash__(self):
        return hash(str.lower(self))


class _Unhashable(str):
    __hash__ = None


def test_equal_looking_str_subclasses_are_not_swapped_for_one_another():
    lower, upper = _CaseBlind("a" * 257), _CaseBlind("A" * 257)

    @workflow
    def program():
        return [total_of(lower), total_of(upper), total_of(_Unhashable("b" * 300))]

    first, second, third = program()._prepare_workflow_input()["nodes"].values()

    assert str.__str__(first["args"][0]) == "a" * 257
    assert str.__str__(second["args"][0]) == "A" * 257
    assert str.__str__(third["args"][0]) == "b" * 300


def test_a_failure_whose_message_is_a_str_subclass_keeps_nothing_alive():
    import gc
    import weakref

    from marqov.workflows import argument_capture

    refs = []

    class Clingy(str):
        def __getitem__(self, item):
            return self

        def __str__(self):
            return self

    class Loud(Exception):
        def __str__(self):
            text = Clingy("loud")
            text.owner = self
            return text

    def encoder(call):
        local = _Big()
        refs.append(weakref.ref(local))
        raise Loud()

    @workflow
    def program():
        return total_of(1)

    with argument_capture(encoder, name="test/1"):
        dispatch = program()
    gc.collect()

    (node,) = dispatch.graph.nodes.values()
    assert type(node.snapshot.error_message) is str
    assert node.snapshot.error_message == "loud"
    assert refs[-1]() is None


def test_a_failure_whose_message_cannot_be_read_is_still_recorded():
    from marqov.workflows import argument_capture

    class Slippery(str):
        def __getitem__(self, item):
            raise LookupError("no slicing")

        def __str__(self):
            raise LookupError("no str")

    class Loud(Exception):
        def __str__(self):
            return Slippery("loud")

    def encoder(call):
        raise Loud()

    @workflow
    def program():
        return total_of(1)

    with argument_capture(encoder, name="test/1"):
        dispatch = program()

    (node,) = dispatch.graph.nodes.values()
    assert node.snapshot.status == "failed"
    assert type(node.snapshot.error_message) is str


def test_the_encoder_is_fixed_when_the_workflow_is_entered():
    from marqov.workflows import argument_capture

    def named(tag):
        return lambda call: tag

    @workflow
    def inner():
        return total_of(0)

    built = {}

    @workflow
    def program():
        first = total_of(1)
        with argument_capture(None):
            second = total_of(2)
            with argument_capture(named("b"), name="b/1"):
                third = total_of(3)
                built["inner"] = inner()
        return [first, second, third, total_of(4)]

    with argument_capture(named("a"), name="a/1"):
        dispatch = program()

    # Every call in the workflow used the encoder in force when it was entered.
    assert [node.snapshot.encoder for node in dispatch.graph.nodes.values()] == ["a/1"] * 4
    # A workflow entered inside the body takes the encoder in force at its own entry.
    assert [node.snapshot.encoder for node in built["inner"].graph.nodes.values()] == ["b/1"]


def test_a_recorded_builtin_error_is_rebuilt_from_its_plain_arguments():
    from marqov.workflows.graph import ArgumentSnapshot

    def rebuilt(exc):
        return ArgumentSnapshot.failure("test/1", exc, ()).error()

    key = rebuilt(KeyError("missing"))
    assert type(key) is KeyError and key.args == ("missing",) and str(key) == "'missing'"

    decode = rebuilt(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"))
    assert type(decode) is UnicodeDecodeError and decode.reason == "invalid start byte"

    missing = rebuilt(FileNotFoundError(2, "No such file or directory"))
    assert type(missing) is FileNotFoundError and missing.errno == 2

    class Custom(Exception):
        pass

    custom = rebuilt(Custom("boom"))
    assert type(custom) is TypeError and "Custom: boom" in str(custom)

    # Arguments that are not plain values are never kept.
    holder = _Big()
    kept = ArgumentSnapshot.failure("test/1", ValueError("bad", holder), ())
    assert kept.error_args is None
    assert type(kept.error()) is ValueError


def test_task_called_from_a_plain_thread_runs_immediately_and_is_not_recorded():
    """Pre-existing behaviour, pinned and documented on ``task``: a plain
    thread starts with an empty context, so it sees no workflow being built."""
    import threading

    calls: list[int] = []

    @task
    def record(x):
        calls.append(x)
        return x * 2

    seen: dict[str, Any] = {}

    @workflow
    def wf():
        def in_thread():
            seen["result"] = record(21)

        thread = threading.Thread(target=in_thread)
        thread.start()
        thread.join()
        return record(1)

    graph = wf().graph

    assert seen["result"] == 42
    assert calls == [21]
    assert [node.func_name for node in graph.nodes.values()] == ["record"]
    assert [node.args for node in graph.nodes.values()] == [[1]]
