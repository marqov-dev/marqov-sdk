"""Tests for @task/@workflow decorators and backward compatibility.

Tests both new decorators (@task, @workflow) and deprecated aliases
(@electron, @lattice) to ensure backward compatibility.
"""

import pytest
from marqov import electron, lattice, task, workflow
from marqov.workflows import TransportGraph, TaskProxy, ElectronProxy


class TestElectronDecorator:
    """Tests for the @electron decorator."""

    def test_electron_outside_lattice_executes(self):
        """Electron called outside lattice should execute normally."""
        @electron
        def add(x, y):
            return x + y

        result = add(1, 2)
        assert result == 3

    def test_electron_with_parameters(self):
        """Electron with parameters should work."""
        @electron(executor="braket", timeout=600)
        def measure(circuit):
            return {"counts": {"00": 500}}

        result = measure("test")
        assert result == {"counts": {"00": 500}}

    def test_electron_inside_lattice_returns_proxy(self):
        """Electron called inside lattice should return proxy."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def compute():
            return add(1, 2)

        dispatch = compute()
        # The result should be a LatticeDispatch
        assert len(dispatch.graph.nodes) == 1

    def test_electron_is_marked(self):
        """Electron decorator should mark the function."""
        @electron
        def add(x, y):
            return x + y

        assert hasattr(add, "_is_electron")
        assert add._is_electron is True


class TestLatticeDecorator:
    """Tests for the @lattice decorator."""

    def test_lattice_returns_dispatch(self):
        """Lattice should return a LatticeDispatch object."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def compute():
            return add(1, 2)

        dispatch = compute()
        assert hasattr(dispatch, "graph")
        assert hasattr(dispatch, "visualize")
        assert hasattr(dispatch, "get_parallel_groups")

    def test_lattice_with_name(self):
        """Lattice with name parameter should use that name."""
        @electron
        def add(x, y):
            return x + y

        @lattice(name="my-workflow")
        def compute():
            return add(1, 2)

        dispatch = compute()
        assert dispatch.name == "my-workflow"

    def test_lattice_captures_dependencies(self):
        """Lattice should capture electron dependencies."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def compute():
            a = add(1, 2)
            b = add(3, 4)
            c = add(a, b)  # Depends on a and b
            return c

        dispatch = compute()
        graph = dispatch.graph

        # Should have 3 nodes
        assert len(graph.nodes) == 3

        # Should have 2 edges (a->c, b->c)
        assert len(graph.edges) == 2

    def test_lattice_detects_parallel_groups(self):
        """Lattice should detect parallel execution groups."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def compute():
            a = add(1, 2)  # Level 0
            b = add(3, 4)  # Level 0 (parallel with a)
            c = add(a, b)  # Level 1 (depends on a and b)
            return c

        dispatch = compute()
        groups = dispatch.get_parallel_groups()

        # Should have 2 levels
        assert len(groups) == 2

        # First level has 2 nodes (parallel)
        assert len(groups[0]) == 2

        # Second level has 1 node (depends on first level)
        assert len(groups[1]) == 1


class TestTransportGraph:
    """Tests for the TransportGraph class."""

    def test_graph_serialization(self):
        """Graph should serialize and deserialize correctly."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def compute():
            a = add(1, 2)
            b = add(a, 3)
            return b

        dispatch = compute()
        graph = dispatch.graph

        # Serialize
        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data

        # Deserialize
        restored = TransportGraph.from_dict(data)
        assert len(restored.nodes) == len(graph.nodes)
        assert len(restored.edges) == len(graph.edges)

    def test_graph_visualization(self):
        """Graph should generate DOT format."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def compute():
            return add(1, 2)

        dispatch = compute()
        dot = dispatch.visualize()

        assert "digraph lattice" in dot
        assert "add" in dot

    def test_output_node_marked(self):
        """Output nodes should be marked in the graph."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def compute():
            return add(1, 2)

        dispatch = compute()
        assert len(dispatch.graph.output_nodes) == 1


class TestComplexWorkflows:
    """Tests for more complex workflow patterns."""

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency graph."""
        @electron
        def add(x, y):
            return x + y

        @lattice
        def diamond():
            a = add(1, 2)           # Level 0
            b = add(a, 3)           # Level 1
            c = add(a, 4)           # Level 1 (parallel with b)
            d = add(b, c)           # Level 2
            return d

        dispatch = diamond()
        groups = dispatch.get_parallel_groups()

        # Should have 3 levels
        assert len(groups) == 3

        # Level 0: a
        assert len(groups[0]) == 1

        # Level 1: b, c (parallel)
        assert len(groups[1]) == 2

        # Level 2: d
        assert len(groups[2]) == 1

    def test_vqe_like_pattern(self):
        """Test VQE-like pattern with multiple independent measurements."""
        @electron
        def measure(circuit, pauli):
            return {"pauli": pauli, "result": 0.5}

        @electron
        def compute_energy(*results):
            return sum(r["result"] for r in results)

        @lattice
        def vqe_step(theta):
            circuit = f"ansatz({theta})"
            # 5 independent measurements - should be parallel
            z0 = measure(circuit, "ZI")
            z1 = measure(circuit, "IZ")
            zz = measure(circuit, "ZZ")
            xx = measure(circuit, "XX")
            yy = measure(circuit, "YY")
            # Final computation depends on all 5
            return compute_energy(z0, z1, zz, xx, yy)

        dispatch = vqe_step(0.5)
        groups = dispatch.get_parallel_groups()

        # Should have 2 levels
        assert len(groups) == 2

        # Level 0: 5 parallel measurements
        assert len(groups[0]) == 5

        # Level 1: 1 energy computation
        assert len(groups[1]) == 1

        # Total 6 nodes
        assert len(dispatch.graph.nodes) == 6
