"""Approach A (marqov-platform#1259): @task must not serialize at decoration time.

These lock in the contract that cloudpickle runs LAZILY at graph-build, never at
decoration/import — the import-time landmine that overflowed the recursion limit
(RecursionError on Linux, hard segfault on macOS/musl) under heavy imports.

Robustness note: the tests that let *real* cloudpickle run at graph-build pickle a
MODULE-LEVEL function (`_plain_add`), which cloudpickle serializes by reference
(cheap — no deep traversal). The tests that assert decoration does NOT pickle use
local functions (the exact fragile #1259 case) but mock ``cloudpickle.dumps``, so
no local is ever actually serialized. Net: deterministic and safe to run under the
full ``[all]`` suite that surfaces the underlying bug.
"""
from unittest.mock import patch

import cloudpickle

from marqov import task, workflow


def _plain_add(x, y):
    """Module-level (importable) function → cloudpickle serializes it by reference."""
    return x + y


def test_decoration_does_not_serialize():
    """@task must NOT call cloudpickle at decoration time — the #1259 landmine."""
    with patch("marqov.workflows.decorators.cloudpickle.dumps") as dumps:

        @task
        def add(x, y):  # a LOCAL function — the exact fragile case
            return x + y

        dumps.assert_not_called()


def test_direct_call_never_serializes():
    """A task called outside a workflow executes directly and never pickles."""
    with patch("marqov.workflows.decorators.cloudpickle.dumps") as dumps:

        @task
        def add(x, y):
            return x + y

        assert add(1, 2) == 3
        dumps.assert_not_called()


def test_serialization_deferred_to_graph_build():
    """Serialization happens when a task builds a graph node inside @workflow."""
    add = task(_plain_add)  # fresh wrapper (fresh cache); pickles by reference

    with patch(
        "marqov.workflows.decorators.cloudpickle.dumps", wraps=cloudpickle.dumps
    ) as dumps:

        @workflow
        def compute():
            return add(1, 2)

        dispatch = compute()
        dumps.assert_called()  # pickled at graph-build, not before

    node = next(iter(dispatch.graph.nodes.values()))
    assert node.func_ref  # populated and non-empty


def test_func_ref_computed_once_and_cached():
    """Repeated builds / multiple nodes reuse the cached func_ref (pickle once)."""
    add = task(_plain_add)  # fresh wrapper so this test's cache starts empty

    with patch(
        "marqov.workflows.decorators.cloudpickle.dumps", wraps=cloudpickle.dumps
    ) as dumps:

        @workflow
        def compute():
            a = add(1, 2)
            b = add(a, 3)  # same task, a second node in the same graph
            return b

        compute()
        compute()  # build the whole graph a second time

        assert dumps.call_count == 1  # cached across every node and every build


def test_eager_func_ref_attribute_removed():
    """The write-only `_task_func_ref` attribute is gone (would defeat laziness)."""

    @task
    def add(x, y):
        return x + y

    assert add._is_task is True
    assert not hasattr(add, "_task_func_ref")
