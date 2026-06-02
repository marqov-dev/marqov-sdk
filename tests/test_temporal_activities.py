"""Tests for marqov.workflows.activity module.

Tests for Temporal activity functions that handle task execution
and dependency resolution in the workflow system.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import patch, MagicMock

import cloudpickle
import pytest

from marqov.workflows.activity import (
    _deserialize_value,
    _serialize_value,
    execute_task,
    prepare_node_inputs,
)


class TestSerializeValue:
    """Tests for _serialize_value function."""

    def test_serialize_none(self) -> None:
        """None passes through unchanged."""
        assert _serialize_value(None) is None

    def test_serialize_bool(self) -> None:
        """Booleans pass through unchanged."""
        assert _serialize_value(True) is True
        assert _serialize_value(False) is False

    def test_serialize_int(self) -> None:
        """Integers pass through unchanged."""
        assert _serialize_value(42) == 42
        assert _serialize_value(-1) == -1
        assert _serialize_value(0) == 0

    def test_serialize_float(self) -> None:
        """Floats pass through unchanged."""
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(-0.5) == -0.5

    def test_serialize_str(self) -> None:
        """Strings pass through unchanged."""
        assert _serialize_value("hello") == "hello"
        assert _serialize_value("") == ""

    def test_serialize_list(self) -> None:
        """Lists are recursively serialized."""
        result = _serialize_value([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_serialize_tuple(self) -> None:
        """Tuples are converted to lists during serialization."""
        result = _serialize_value((1, 2, 3))
        assert result == [1, 2, 3]

    def test_serialize_dict(self) -> None:
        """Dicts are recursively serialized."""
        result = _serialize_value({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_serialize_nested_structure(self) -> None:
        """Nested structures are handled correctly."""
        value = {"list": [1, 2], "nested": {"inner": "value"}}
        result = _serialize_value(value)
        assert result == {"list": [1, 2], "nested": {"inner": "value"}}

    def test_serialize_cloudpickle_marker_preserved(self) -> None:
        """Existing cloudpickle markers are preserved."""
        marker = {"__cloudpickle__": True, "data": "base64data"}
        result = _serialize_value(marker)
        assert result == marker

    def test_serialize_complex_object(self) -> None:
        """Complex objects are cloudpickled."""
        class CustomClass:
            def __init__(self, x: int) -> None:
                self.x = x

        obj = CustomClass(42)
        result = _serialize_value(obj)

        assert "__cloudpickle__" in result
        assert "data" in result
        # Verify we can deserialize it
        decoded = cloudpickle.loads(base64.b64decode(result["data"]))
        assert decoded.x == 42

    def test_serialize_function(self) -> None:
        """Functions are cloudpickled."""
        def my_func(x: int) -> int:
            return x * 2

        result = _serialize_value(my_func)

        assert "__cloudpickle__" in result
        decoded = cloudpickle.loads(base64.b64decode(result["data"]))
        assert decoded(5) == 10

    def test_serialize_list_with_complex_objects(self) -> None:
        """Lists containing complex objects are serialized correctly."""
        class Item:
            def __init__(self, val: int) -> None:
                self.val = val

        result = _serialize_value([1, Item(10), "text"])

        assert result[0] == 1
        assert result[2] == "text"
        assert "__cloudpickle__" in result[1]


class TestDeserializeValue:
    """Tests for _deserialize_value function."""

    def test_deserialize_none(self) -> None:
        """None passes through unchanged."""
        assert _deserialize_value(None) is None

    def test_deserialize_bool(self) -> None:
        """Booleans pass through unchanged."""
        assert _deserialize_value(True) is True
        assert _deserialize_value(False) is False

    def test_deserialize_int(self) -> None:
        """Integers pass through unchanged."""
        assert _deserialize_value(42) == 42

    def test_deserialize_float(self) -> None:
        """Floats pass through unchanged."""
        assert _deserialize_value(3.14) == 3.14

    def test_deserialize_str(self) -> None:
        """Strings pass through unchanged."""
        assert _deserialize_value("hello") == "hello"

    def test_deserialize_list(self) -> None:
        """Lists are recursively deserialized."""
        result = _deserialize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_deserialize_dict(self) -> None:
        """Dicts are recursively deserialized."""
        result = _deserialize_value({"key": "value"})
        assert result == {"key": "value"}

    def test_deserialize_cloudpickle_marker(self) -> None:
        """Cloudpickle markers are decoded."""
        class TestClass:
            value = 100

        pickled = base64.b64encode(cloudpickle.dumps(TestClass)).decode("utf-8")
        marker = {"__cloudpickle__": True, "data": pickled}

        result = _deserialize_value(marker)
        assert result.value == 100

    def test_deserialize_nested_cloudpickle(self) -> None:
        """Nested cloudpickle markers in structures are deserialized."""
        pickled = base64.b64encode(cloudpickle.dumps(lambda x: x + 1)).decode("utf-8")
        value = {
            "simple": 1,
            "complex": {"__cloudpickle__": True, "data": pickled},
        }

        result = _deserialize_value(value)
        assert result["simple"] == 1
        assert result["complex"](5) == 6

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Serialization followed by deserialization preserves values."""
        class MyClass:
            def __init__(self, x: int, y: str) -> None:
                self.x = x
                self.y = y

        original = MyClass(42, "test")
        serialized = _serialize_value(original)
        restored = _deserialize_value(serialized)

        assert restored.x == 42
        assert restored.y == "test"


class TestExecuteTask:
    """Tests for execute_task activity."""

    def _encode_func(self, func: Any) -> str:
        """Helper to encode a function for execute_task."""
        return base64.b64encode(cloudpickle.dumps(func)).decode("utf-8")

    @pytest.mark.asyncio
    async def test_execute_sync_function(self) -> None:
        """Execute a synchronous function."""
        def add(a: int, b: int) -> int:
            return a + b

        func_ref = self._encode_func(add)
        args_json = json.dumps([3, 5])
        kwargs_json = json.dumps({})

        result_json = await execute_task("node1", func_ref, args_json, kwargs_json)
        result = json.loads(result_json)

        assert result["node_id"] == "node1"
        assert result["result"] == 8

    @pytest.mark.asyncio
    async def test_execute_async_function(self) -> None:
        """Execute an asynchronous function."""
        async def async_multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x * y

        func_ref = self._encode_func(async_multiply)
        args_json = json.dumps([4, 7])
        kwargs_json = json.dumps({})

        result_json = await execute_task("node2", func_ref, args_json, kwargs_json)
        result = json.loads(result_json)

        assert result["node_id"] == "node2"
        assert result["result"] == 28

    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self) -> None:
        """Execute function with keyword arguments."""
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        func_ref = self._encode_func(greet)
        args_json = json.dumps(["Alice"])
        kwargs_json = json.dumps({"greeting": "Hi"})

        result_json = await execute_task("node3", func_ref, args_json, kwargs_json)
        result = json.loads(result_json)

        assert result["node_id"] == "node3"
        assert result["result"] == "Hi, Alice!"

    @pytest.mark.asyncio
    async def test_execute_with_complex_args(self) -> None:
        """Execute with cloudpickled arguments."""
        class Data:
            def __init__(self, val: int) -> None:
                self.val = val

        def process(data: Data) -> int:
            return data.val * 2

        func_ref = self._encode_func(process)
        # Serialize the Data object
        data_serialized = _serialize_value(Data(21))
        args_json = json.dumps([data_serialized])
        kwargs_json = json.dumps({})

        result_json = await execute_task("node4", func_ref, args_json, kwargs_json)
        result = json.loads(result_json)

        assert result["node_id"] == "node4"
        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_execute_returns_complex_result(self) -> None:
        """Execute function that returns a complex object."""
        class Result:
            def __init__(self, value: int) -> None:
                self.value = value

        def create_result() -> Result:
            return Result(99)

        func_ref = self._encode_func(create_result)
        args_json = json.dumps([])
        kwargs_json = json.dumps({})

        result_json = await execute_task("node5", func_ref, args_json, kwargs_json)
        result = json.loads(result_json)

        assert result["node_id"] == "node5"
        # Result should be cloudpickled
        assert "__cloudpickle__" in result["result"]
        deserialized = _deserialize_value(result["result"])
        assert deserialized.value == 99

    @pytest.mark.asyncio
    async def test_execute_with_list_result(self) -> None:
        """Execute function returning a list."""
        def make_list(n: int) -> list[int]:
            return list(range(n))

        func_ref = self._encode_func(make_list)
        args_json = json.dumps([5])
        kwargs_json = json.dumps({})

        result_json = await execute_task("node6", func_ref, args_json, kwargs_json)
        result = json.loads(result_json)

        assert result["node_id"] == "node6"
        assert result["result"] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_execute_preserves_node_id(self) -> None:
        """Node ID is preserved in result."""
        def identity(x: Any) -> Any:
            return x

        func_ref = self._encode_func(identity)
        args_json = json.dumps([1])
        kwargs_json = json.dumps({})

        result_json = await execute_task(
            "my-unique-node-id", func_ref, args_json, kwargs_json
        )
        result = json.loads(result_json)

        assert result["node_id"] == "my-unique-node-id"

    @pytest.mark.asyncio
    async def test_execute_empty_args(self) -> None:
        """Execute function with no arguments."""
        def get_constant() -> int:
            return 42

        func_ref = self._encode_func(get_constant)
        args_json = json.dumps([])
        kwargs_json = json.dumps({})

        result_json = await execute_task("node7", func_ref, args_json, kwargs_json)
        result = json.loads(result_json)

        assert result["result"] == 42


class TestPrepareNodeInputs:
    """Tests for prepare_node_inputs activity."""

    @pytest.mark.asyncio
    async def test_prepare_simple_args(self) -> None:
        """Prepare node with simple arguments (no proxies)."""
        node_data = {
            "node_id": "task1",
            "func_ref": "base64funcref",
            "args": [1, 2, 3],
            "kwargs": {"key": "value"},
        }
        completed = {}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        assert result["node_id"] == "task1"
        assert result["func_ref"] == "base64funcref"
        assert result["args"] == [1, 2, 3]
        assert result["kwargs"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_prepare_resolves_proxy(self) -> None:
        """Prepare node resolves proxy references."""
        node_data = {
            "node_id": "task2",
            "func_ref": "funcref",
            "args": [{"__proxy__": True, "node_id": "task1"}],
            "kwargs": {},
        }
        completed = {"task1": 100}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        assert result["args"] == [100]

    @pytest.mark.asyncio
    async def test_prepare_resolves_nested_proxy(self) -> None:
        """Prepare node resolves nested proxy references."""
        node_data = {
            "node_id": "task3",
            "func_ref": "funcref",
            "args": [
                {
                    "data": {"__proxy__": True, "node_id": "task1"},
                    "other": "value",
                }
            ],
            "kwargs": {},
        }
        completed = {"task1": "resolved_value"}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        assert result["args"] == [{"data": "resolved_value", "other": "value"}]

    @pytest.mark.asyncio
    async def test_prepare_resolves_proxy_in_list(self) -> None:
        """Prepare node resolves proxies inside lists."""
        node_data = {
            "node_id": "task4",
            "func_ref": "funcref",
            "args": [[{"__proxy__": True, "node_id": "a"}, {"__proxy__": True, "node_id": "b"}]],
            "kwargs": {},
        }
        completed = {"a": 10, "b": 20}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        assert result["args"] == [[10, 20]]

    @pytest.mark.asyncio
    async def test_prepare_resolves_proxy_in_kwargs(self) -> None:
        """Prepare node resolves proxies in kwargs."""
        node_data = {
            "node_id": "task5",
            "func_ref": "funcref",
            "args": [],
            "kwargs": {
                "input": {"__proxy__": True, "node_id": "prev"},
            },
        }
        completed = {"prev": "previous_result"}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        assert result["kwargs"] == {"input": "previous_result"}

    @pytest.mark.asyncio
    async def test_prepare_raises_on_missing_dependency(self) -> None:
        """Prepare raises error when dependency is not computed."""
        node_data = {
            "node_id": "task6",
            "func_ref": "funcref",
            "args": [{"__proxy__": True, "node_id": "missing_node"}],
            "kwargs": {},
        }
        completed = {}  # No results available

        with pytest.raises(ValueError, match="Dependency missing_node not yet computed"):
            await prepare_node_inputs(json.dumps(node_data), json.dumps(completed))

    @pytest.mark.asyncio
    async def test_prepare_mixed_args(self) -> None:
        """Prepare handles mix of proxies and literal values."""
        node_data = {
            "node_id": "task7",
            "func_ref": "funcref",
            "args": [
                "literal",
                {"__proxy__": True, "node_id": "dep1"},
                42,
                {"__proxy__": True, "node_id": "dep2"},
            ],
            "kwargs": {},
        }
        completed = {"dep1": "first", "dep2": "second"}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        assert result["args"] == ["literal", "first", 42, "second"]

    @pytest.mark.asyncio
    async def test_prepare_preserves_func_ref(self) -> None:
        """Prepare preserves the function reference."""
        func_ref = "base64encodedfunction=="
        node_data = {
            "node_id": "task8",
            "func_ref": func_ref,
            "args": [],
            "kwargs": {},
        }
        completed = {}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        assert result["func_ref"] == func_ref

    @pytest.mark.asyncio
    async def test_prepare_complex_result_reference(self) -> None:
        """Prepare handles complex objects as completed results."""
        # Completed result is a cloudpickle-serialized value
        completed_value = {"__cloudpickle__": True, "data": "somebase64"}
        node_data = {
            "node_id": "task9",
            "func_ref": "funcref",
            "args": [{"__proxy__": True, "node_id": "complex_task"}],
            "kwargs": {},
        }
        completed = {"complex_task": completed_value}

        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        result = json.loads(result_json)

        # The cloudpickle marker should be passed through
        assert result["args"] == [completed_value]


class TestActivityIntegration:
    """Integration tests for activity functions working together."""

    def _encode_func(self, func: Any) -> str:
        """Helper to encode a function for execute_task."""
        return base64.b64encode(cloudpickle.dumps(func)).decode("utf-8")

    @pytest.mark.asyncio
    async def test_prepare_then_execute(self) -> None:
        """Full flow: prepare inputs then execute task."""
        def add(a: int, b: int) -> int:
            return a + b

        func_ref = self._encode_func(add)

        # First task result
        completed = {"first": 10}

        # Second task depends on first
        node_data = {
            "node_id": "second",
            "func_ref": func_ref,
            "args": [{"__proxy__": True, "node_id": "first"}, 5],
            "kwargs": {},
        }

        # Prepare inputs
        prepared_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        prepared = json.loads(prepared_json)

        # Execute task
        result_json = await execute_task(
            prepared["node_id"],
            prepared["func_ref"],
            json.dumps(prepared["args"]),
            json.dumps(prepared["kwargs"]),
        )
        result = json.loads(result_json)

        assert result["node_id"] == "second"
        assert result["result"] == 15  # 10 + 5

    @pytest.mark.asyncio
    async def test_chain_of_tasks(self) -> None:
        """Execute a chain of dependent tasks."""
        def double(x: int) -> int:
            return x * 2

        def add_ten(x: int) -> int:
            return x + 10

        double_ref = self._encode_func(double)
        add_ten_ref = self._encode_func(add_ten)

        # Execute first task
        result1_json = await execute_task(
            "task1", double_ref, json.dumps([5]), json.dumps({})
        )
        result1 = json.loads(result1_json)
        completed = {"task1": result1["result"]}

        # Prepare second task
        node_data = {
            "node_id": "task2",
            "func_ref": add_ten_ref,
            "args": [{"__proxy__": True, "node_id": "task1"}],
            "kwargs": {},
        }
        prepared_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )
        prepared = json.loads(prepared_json)

        # Execute second task
        result2_json = await execute_task(
            prepared["node_id"],
            prepared["func_ref"],
            json.dumps(prepared["args"]),
            json.dumps(prepared["kwargs"]),
        )
        result2 = json.loads(result2_json)

        # 5 * 2 = 10, then 10 + 10 = 20
        assert result2["result"] == 20
