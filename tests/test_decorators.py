"""Tests for @task/@workflow decorators."""

import pytest
from marqov import task, workflow
from marqov.workflows import TransportGraph, TaskProxy


class TestTaskDecorator:
    """Tests for the @task decorator."""

    def test_task_outside_workflow_executes(self):
        """Task called outside workflow should execute normally."""
        @task
        def add(x, y):
            return x + y

        result = add(1, 2)
        assert result == 3

    def test_task_with_parameters(self):
        """Task with parameters should work."""
        @task(executor="braket", timeout=600)
        def measure(circuit):
            return {"counts": {"00": 500}}

        result = measure("test")
        assert result == {"counts": {"00": 500}}

    def test_task_inside_workflow_returns_proxy(self):
        """Task called inside workflow should return proxy."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def compute():
            return add(1, 2)

        dispatch = compute()
        assert len(dispatch.graph.nodes) == 1

    def test_task_is_marked(self):
        """Task decorator should mark the function."""
        @task
        def add(x, y):
            return x + y

        assert hasattr(add, "_is_task")
        assert add._is_task is True


class TestWorkflowDecorator:
    """Tests for the @workflow decorator."""

    def test_workflow_returns_dispatch(self):
        """Workflow should return a WorkflowDispatch object."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def compute():
            return add(1, 2)

        dispatch = compute()
        assert hasattr(dispatch, "graph")
        assert hasattr(dispatch, "visualize")
        assert hasattr(dispatch, "get_parallel_groups")

    def test_workflow_with_name(self):
        """Workflow with name parameter should use that name."""
        @task
        def add(x, y):
            return x + y

        @workflow(name="my-workflow")
        def compute():
            return add(1, 2)

        dispatch = compute()
        assert dispatch.name == "my-workflow"

    def test_workflow_captures_dependencies(self):
        """Workflow should capture task dependencies."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def compute():
            a = add(1, 2)
            b = add(3, 4)
            c = add(a, b)
            return c

        dispatch = compute()
        graph = dispatch.graph

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_workflow_detects_parallel_groups(self):
        """Workflow should detect parallel execution groups."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def compute():
            a = add(1, 2)  # Level 0
            b = add(3, 4)  # Level 0 (parallel with a)
            c = add(a, b)  # Level 1
            return c

        dispatch = compute()
        groups = dispatch.get_parallel_groups()

        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1


class TestTransportGraph:
    """Tests for the TransportGraph class."""

    def test_graph_serialization(self):
        """Graph should serialize to dict correctly."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def compute():
            a = add(1, 2)
            b = add(a, 3)
            return b

        dispatch = compute()
        data = dispatch.graph.to_dict()

        assert "nodes" in data
        assert "edges" in data

    def test_graph_visualization(self):
        """Graph should generate DOT format."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def compute():
            return add(1, 2)

        dispatch = compute()
        dot = dispatch.visualize()

        assert "digraph" in dot
        assert "add" in dot

    def test_output_node_marked(self):
        """Output nodes should be marked in the graph."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def compute():
            return add(1, 2)

        dispatch = compute()
        assert len(dispatch.graph.output_nodes) == 1


class TestComplexWorkflows:
    """Tests for more complex workflow patterns."""

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency graph."""
        @task
        def add(x, y):
            return x + y

        @workflow
        def diamond():
            a = add(1, 2)   # Level 0
            b = add(a, 3)   # Level 1
            c = add(a, 4)   # Level 1 (parallel with b)
            d = add(b, c)   # Level 2
            return d

        dispatch = diamond()
        groups = dispatch.get_parallel_groups()

        assert len(groups) == 3
        assert len(groups[0]) == 1
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1

    def test_vqe_like_pattern(self):
        """Test VQE-like pattern with multiple independent measurements."""
        @task
        def measure(circuit, pauli):
            return {"pauli": pauli, "result": 0.5}

        @task
        def compute_energy(*results):
            return sum(r["result"] for r in results)

        @workflow
        def vqe_step(theta):
            circuit = f"ansatz({theta})"
            z0 = measure(circuit, "ZI")
            z1 = measure(circuit, "IZ")
            zz = measure(circuit, "ZZ")
            xx = measure(circuit, "XX")
            yy = measure(circuit, "YY")
            return compute_energy(z0, z1, zz, xx, yy)

        dispatch = vqe_step(0.5)
        groups = dispatch.get_parallel_groups()

        assert len(groups) == 2
        assert len(groups[0]) == 5
        assert len(groups[1]) == 1
        assert len(dispatch.graph.nodes) == 6
