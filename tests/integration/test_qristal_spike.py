"""Integration tests that run against a real qristal build.

Skip unless QRISTAL_AVAILABLE=1 is set. Run inside Dockerfile.simulation.
"""

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRISTAL_AVAILABLE") != "1",
        reason="Requires real qristal build (set QRISTAL_AVAILABLE=1)",
    ),
]


@pytest.fixture
def qb():
    """Import and return qristal.core module."""
    import qristal.core  # noqa: F811

    return qristal.core


@pytest.fixture
def bell_qasm() -> str:
    """Bell state circuit as OpenQASM 2.0."""
    return (
        'OPENQASM 2.0;\n'
        'include "qelib1.inc";\n'
        'qreg q[2];\n'
        'creg c[2];\n'
        'h q[0];\n'
        'cx q[0],q[1];\n'
        'measure q[0] -> c[0];\n'
        'measure q[1] -> c[1];\n'
    )


class TestSessionConfiguration:
    """Verify session parameter setting."""

    def test_session_init(self, qb) -> None:
        """Session initializes without error."""
        session = qb.session()
        session.init()

    def test_session_accepts_qasm(self, qb, bell_qasm) -> None:
        """Session accepts raw QASM via instring."""
        session = qb.session()
        session.init()
        session.acc = "qpp"
        session.qn = [2]
        session.sn = [100]
        session.instring = [[bell_qasm]]
        session.run()
        # Should complete without error

    def test_session_seed(self, qb, bell_qasm) -> None:
        """Same seed produces same results."""
        results = []
        for _ in range(2):
            session = qb.session()
            session.init()
            session.acc = "qpp"
            session.qn = [2]
            session.sn = [100]
            session.seed = 42
            session.instring = [[bell_qasm]]
            session.run()
            counts = {}
            for key in session.results[0][0]:
                bs = "".join("1" if key[i] else "0" for i in range(len(key)))
                counts[bs] = session.results[0][0][key]
            results.append(counts)
        assert results[0] == results[1]


class TestResultTypes:
    """Verify pybind11 result type behavior."""

    def test_result_iteration_yields_keys(self, qb, bell_qasm) -> None:
        """Iterating results yields keys only, not (key, value) pairs."""
        session = qb.session()
        session.init()
        session.acc = "qpp"
        session.qn = [2]
        session.sn = [100]
        session.instring = [[bell_qasm]]
        session.run()

        result_map = session.results[0][0]
        for key in result_map:
            # key should be indexable, not a tuple of (key, value)
            assert hasattr(key, "__getitem__"), f"key type {type(key)} is not indexable"
            count = result_map[key]
            assert isinstance(count, int), f"count type {type(count)} is not int"

    def test_result_has_no_items_method(self, qb, bell_qasm) -> None:
        """Result map has no .items() method."""
        session = qb.session()
        session.init()
        session.acc = "qpp"
        session.qn = [2]
        session.sn = [100]
        session.instring = [[bell_qasm]]
        session.run()

        result_map = session.results[0][0]
        assert not hasattr(result_map, "items"), "Result map should NOT have .items()"


class TestBellStateValidation:
    """Verify bell state produces expected outcomes."""

    def test_bell_state_outcomes(self, qb, bell_qasm) -> None:
        """Bell state produces only 00 and 11."""
        session = qb.session()
        session.init()
        session.acc = "qpp"
        session.qn = [2]
        session.sn = [1000]
        session.instring = [[bell_qasm]]
        session.run()

        counts = {}
        for key in session.results[0][0]:
            bs = "".join("1" if key[i] else "0" for i in range(len(key)))
            counts[bs] = session.results[0][0][key]

        assert set(counts.keys()).issubset({"00", "11"})
        total = sum(counts.values())
        assert total == 1000
        for count in counts.values():
            assert 400 < count < 600, f"Expected ~50%, got {count}/1000"


class TestQubitOrdering:
    """Verify qubit ordering convention."""

    def test_qubit_zero_is_index_zero(self, qb) -> None:
        """Qubit 0 prepared in |1⟩ should have key[0] = True."""
        qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'creg c[2];\n'
            'x q[0];\n'
            'measure q[0] -> c[0];\n'
            'measure q[1] -> c[1];\n'
        )
        session = qb.session()
        session.init()
        session.acc = "qpp"
        session.qn = [2]
        session.sn = [100]
        session.instring = [[qasm]]
        session.run()

        for key in session.results[0][0]:
            # qubit 0 was X'd → should be True
            assert key[0] is True, f"qubit 0 should be True, got {key[0]}"
            # qubit 1 was identity → should be False
            assert key[1] is False, f"qubit 1 should be False, got {key[1]}"


class TestSvdCutoffs:
    """Verify SVD cutoff access patterns."""

    def test_svd_cutoff_write(self, qb) -> None:
        """svd_cutoffs[0][0][0] = value works."""
        session = qb.session()
        session.init()
        session.svd_cutoffs[0][0][0] = 1e-6

    def test_rel_svd_cutoff_write(self, qb) -> None:
        """rel_svd_cutoffs[0][0][0] = value works."""
        session = qb.session()
        session.init()
        session.rel_svd_cutoffs[0][0][0] = 1e-8


class TestNoiseApi:
    """Verify noise model API for Phase 2 noise modeling."""

    def test_noise_model_creation(self, qb) -> None:
        """NoiseModel can be created and named."""
        nm = qb.NoiseModel()
        nm.name = "spike_test"

    def test_depolarizing_channel_takes_qubit_index(self, qb) -> None:
        """DepolarizingChannel.Create first arg is qubit index."""
        channel = qb.DepolarizingChannel.Create(0, 0.01)
        nm = qb.NoiseModel()
        nm.name = "test"
        nm.add_gate_error(channel, "u3", [0])

    def test_readout_error_constructor(self, qb) -> None:
        """Verify ReadoutError constructor signature."""
        # Record actual constructor signature — may need adjustment
        try:
            re = qb.ReadoutError(0, 0.02, 0.01)
            nm = qb.NoiseModel()
            nm.name = "test"
            nm.set_qubit_readout_error(0, re)
        except TypeError as e:
            pytest.fail(
                f"ReadoutError constructor signature mismatch: {e}\n"
                f"Update spec with correct signature."
            )

    def test_noise_requires_aer(self, qb, bell_qasm) -> None:
        """Noise simulation requires aer backend."""
        nm = qb.NoiseModel()
        nm.name = "test"
        channel = qb.DepolarizingChannel.Create(0, 0.1)
        nm.add_gate_error(channel, "u3", [0])

        session = qb.session()
        session.init()
        session.acc = "aer"
        session.qn = [2]
        session.sn = [100]
        session.noise = True
        session.noise_model = nm
        session.instring = [[bell_qasm]]
        session.run()
        # Should complete without error on aer
