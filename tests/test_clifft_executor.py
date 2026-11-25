"""Tests for marqov.executors.clifft (Clifft exact near-Clifford sampler).

Real Clifft runs, no mocks, except where a test must simulate an environment
this machine does not have (Clifft missing, or another Clifft version). Exact
references come from QuantumFlow (``Circuit.simulate``), independent of Clifft.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys

import numpy as np
import pytest

from marqov.circuits import Circuit, ghz_state
from marqov.executors import ClifftExecutor, ClifftExecutorConfig, ExecutionResult
from marqov.executors.clifft import (
    CLIFFT_VERSION,
    SUPPORTED_GATES,
    ClifftUnavailableError,
    to_clifft_text,
)

HAVE_CLIFFT = importlib.util.find_spec("clifft") is not None
needs_clifft = pytest.mark.skipif(not HAVE_CLIFFT, reason='clifft not installed (marqov[clifft])')


def _run(circuit: Circuit, shots: int = 1000, executor: ClifftExecutor | None = None, **kw):
    return asyncio.run((executor or ClifftExecutor()).execute(circuit, shots=shots, **kw))


def _exact_probabilities(circuit: Circuit) -> dict[str, float]:
    """QuantumFlow's outcome probabilities, qubit 0 leftmost."""
    state = circuit.simulate()
    n = circuit.num_qubits
    tensor = np.asarray(state.tensor).reshape((2,) * len(state.qubits))
    order = [list(state.qubits).index(q) for q in range(n)]
    probs = np.abs(tensor.transpose(order).reshape(-1)) ** 2
    return {format(i, f"0{n}b"): float(p) for i, p in enumerate(probs) if p > 1e-12}


def _assert_within_sampling_bound(counts: dict[str, int], exact: dict[str, float], shots: int):
    """Every outcome within 5 binomial sigma of its exact count, and no impossible outcome."""
    assert sum(counts.values()) == shots
    assert set(counts) <= set(exact), f"impossible outcomes {set(counts) - set(exact)}"
    for key, p in exact.items():
        sigma = math.sqrt(shots * p * (1 - p))
        assert abs(counts.get(key, 0) - shots * p) <= 5 * sigma + 1, (key, counts.get(key), p)


def _clifft_state(circuit: Circuit) -> np.ndarray:
    """Clifft's statevector for the adapter's program, reordered to qubit 0 most significant."""
    import clifft

    sv = np.asarray(clifft.get_statevector(clifft.compile(to_clifft_text(circuit))))
    n = circuit.num_qubits
    return sv.reshape((2,) * n).transpose(tuple(reversed(range(n)))).reshape(-1)


def _quantumflow_state(circuit: Circuit) -> np.ndarray:
    state = circuit.simulate()
    n = circuit.num_qubits
    tensor = np.asarray(state.tensor).reshape((2,) * len(state.qubits))
    order = [list(state.qubits).index(q) for q in range(n)]
    return tensor.transpose(order).reshape(-1)


# A generic 3-qubit state with every amplitude nonzero and unequal phases, so a
# wrong gate (or a wrong phase) shows up in the state.
def _prepared() -> Circuit:
    return Circuit().ry(0.4, 0).ry(1.1, 1).ry(2.3, 2).rz(0.9, 0).rz(0.3, 1).rz(1.7, 2)


# QuantumFlow gate name -> how to append it to a 3-qubit circuit.
_GATE_CASES = {
    "H": lambda c: c.h(1),
    "X": lambda c: c.x(1),
    "Y": lambda c: c.y(1),
    "Z": lambda c: c.z(1),
    "S": lambda c: c.s(1),
    "T": lambda c: c.t(1),
    "Rx": lambda c: c.rx(0.77, 1),
    "Ry": lambda c: c.ry(-1.3, 1),
    "Rz": lambda c: c.rz(2.9, 1),
    "CNot": lambda c: c.cnot(2, 0),
    "CZ": lambda c: c.cz(0, 2),
    "Swap": lambda c: c.swap(0, 2),
    "CCNot": lambda c: c.ccx(2, 0, 1),
    "CSwap": lambda c: c.cswap(1, 2, 0),
}


class TestImport:
    def test_missing_clifft_raises_install_hint(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "clifft", None)
        with pytest.raises(ClifftUnavailableError, match=r'pip install "marqov\[clifft\]"'):
            _run(Circuit().h(0))

    def test_missing_clifft_reports_offline(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "clifft", None)
        status = asyncio.run(ClifftExecutor().get_status())
        assert status.status == "offline"

    @needs_clifft
    def test_other_clifft_version_refuses(self, monkeypatch) -> None:
        import clifft

        monkeypatch.setattr(clifft, "__version__", "0.12.0")
        with pytest.raises(ClifftUnavailableError, match=CLIFFT_VERSION):
            _run(Circuit().h(0))

    @needs_clifft
    def test_installed_version_is_the_pinned_one(self) -> None:
        import clifft

        assert clifft.__version__ == CLIFFT_VERSION
        assert asyncio.run(ClifftExecutor().get_status()).status == "online"


@needs_clifft
class TestCountsAndBitOrder:
    def test_bit_order_qubit0_leftmost(self) -> None:
        """X on qubits 0 and 1 of a declared 3-qubit register is '110'."""
        result = _run(Circuit(num_qubits=3).x(0).x(1), shots=100, seed=0)

        assert isinstance(result, ExecutionResult)
        assert result.backend == "clifft"
        assert result.counts == {"110": 100}

    @pytest.mark.parametrize(
        ("circuit", "key"),
        [
            (Circuit().x(0).cz(0, 1), "10"),
            (Circuit().x(1).cz(0, 1), "01"),
            (Circuit().x(2), "001"),  # identity wires 0 and 1 are measured
            (Circuit().x(0).cnot(0, 3), "1001"),
        ],
    )
    def test_matches_local_executor_keys(self, circuit: Circuit, key: str) -> None:
        assert _run(circuit, shots=50, seed=1).counts == {key: 50}

    def test_identity_only_circuit_samples_zeros(self) -> None:
        circuit = Circuit.from_dict({"gates": [{"gate": "I", "qubits": [1], "params": []}]})
        assert _run(circuit, shots=20, seed=0).counts == {"00": 20}

    def test_raw_result_is_the_sample_array(self) -> None:
        result = _run(Circuit().h(0).cnot(0, 1).x(2), shots=64, seed=3)
        samples = result.raw_result

        assert samples.shape == (64, 3) and samples.dtype == np.uint8
        rebuilt: dict[str, int] = {}
        for row in samples:
            key = "".join(str(int(b)) for b in row)
            rebuilt[key] = rebuilt.get(key, 0) + 1
        assert rebuilt == result.counts


@needs_clifft
class TestDistributions:
    """Sampled distributions against QuantumFlow's exact probabilities."""

    @pytest.mark.parametrize(
        "circuit",
        [
            ghz_state(5),
            Circuit().h(0).t(0).h(0),  # one T: P(0) = cos^2(pi/8)
            Circuit().h(0).cnot(0, 1).t(1).h(1).s(0).h(0),
            Circuit().ry(0.7, 0).rx(1.3, 1).cnot(0, 1).ry(2.1, 2).cz(1, 2).h(0),
            Circuit().h(0).h(1).h(2).ccx(0, 1, 2).cswap(2, 0, 1).rz(0.4, 0).h(0),
        ],
    )
    def test_within_sampling_bound(self, circuit: Circuit) -> None:
        shots = 20000
        result = _run(circuit, shots=shots, seed=11)
        _assert_within_sampling_bound(result.counts, _exact_probabilities(circuit), shots)


@needs_clifft
class TestGateMapping:
    def test_every_supported_gate_has_a_case(self) -> None:
        assert set(_GATE_CASES) | {"I"} == SUPPORTED_GATES

    @pytest.mark.parametrize("gate", sorted(_GATE_CASES))
    def test_state_matches_quantumflow(self, gate: str) -> None:
        """Clifft's state for each mapped gate equals QuantumFlow's up to global phase."""
        circuit = _GATE_CASES[gate](_prepared())
        assert circuit.to_dict()["gates"][-1]["gate"] == gate
        ours, ref = _clifft_state(circuit), _quantumflow_state(circuit)
        fidelity = abs(np.vdot(ours, ref)) ** 2 / (np.vdot(ours, ours).real * np.vdot(ref, ref).real)
        assert fidelity == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("gate", sorted(_GATE_CASES))
    def test_sampled_distribution_matches_quantumflow(self, gate: str) -> None:
        circuit = _GATE_CASES[gate](_prepared()).h(0).h(1).h(2)
        shots = 20000
        result = _run(circuit, shots=shots, seed=5)
        _assert_within_sampling_bound(result.counts, _exact_probabilities(circuit), shots)

    def test_rotation_angle_is_in_units_of_pi(self) -> None:
        assert to_clifft_text(Circuit().rx(math.pi / 2, 0)) == "R_X(0.5) 0\nI 0"

    def test_cswap_is_decomposed(self) -> None:
        assert to_clifft_text(Circuit().cswap(0, 1, 2)).splitlines()[:3] == [
            "CX 2 1",
            "CCX 0 1 2",
            "CX 2 1",
        ]

    @pytest.mark.parametrize("basis", range(8))
    def test_cswap_truth_table(self, basis: int) -> None:
        """Controlled-SWAP on every basis input of (control, a, b)."""
        bits = format(basis, "03b")
        circuit = Circuit(num_qubits=3)
        for qubit, bit in enumerate(bits):
            if bit == "1":
                circuit.x(qubit)
        circuit.cswap(0, 1, 2)
        expected = bits[0] + (bits[2] + bits[1] if bits[0] == "1" else bits[1:])
        assert _run(circuit, shots=10, seed=0).counts == {expected: 10}

    def test_unmapped_gate_raises(self) -> None:
        circuit = Circuit.from_dict({"gates": [{"gate": "ISwap", "qubits": [0, 1], "params": []}]})
        with pytest.raises(NotImplementedError, match="ISwap"):
            _run(circuit)

    def test_unbound_angle_raises(self) -> None:
        import sympy

        circuit = Circuit().rx(sympy.Symbol("theta"), 0)
        with pytest.raises(ValueError, match="unbound"):
            _run(circuit)


@needs_clifft
class TestSeedReplay:
    # 2000 shots exercises Clifft's packed (batched) sampling path.
    def _circuit(self) -> Circuit:
        return Circuit().h(0).t(0).cnot(0, 1).ry(0.8, 2).h(1).cz(1, 2).h(2)

    def test_same_seed_replays_samples(self) -> None:
        first = _run(self._circuit(), shots=2000, seed=1234)
        second = _run(self._circuit(), shots=2000, seed=1234)

        assert np.array_equal(first.raw_result, second.raw_result)
        record = first.metadata["reproducibility"]
        assert record["samples_sha256"] == second.metadata["reproducibility"]["samples_sha256"]
        assert record["seed"] == 1234 and record["seed_source"] == "caller"

    def test_different_seeds_differ(self) -> None:
        first = _run(self._circuit(), shots=2000, seed=1)
        second = _run(self._circuit(), shots=2000, seed=2)
        assert not np.array_equal(first.raw_result, second.raw_result)

    def test_config_seed_is_used_and_replays(self) -> None:
        executor = ClifftExecutor(ClifftExecutorConfig(seed=99))
        first = _run(self._circuit(), shots=500, executor=executor)
        second = _run(self._circuit(), shots=500, executor=executor)

        assert first.metadata["reproducibility"]["seed_source"] == "config"
        assert np.array_equal(first.raw_result, second.raw_result)

    def test_generated_seed_is_recorded_and_replays(self) -> None:
        first = _run(self._circuit(), shots=500)
        record = first.metadata["reproducibility"]
        assert record["seed_source"] == "generated"

        replay = _run(self._circuit(), shots=500, seed=record["seed"])
        assert np.array_equal(first.raw_result, replay.raw_result)


def _many_t(width: int, layers: int) -> Circuit:
    """Layers of H and T on every qubit with a CNOT chain: active width reaches the qubit count."""
    circuit = Circuit()
    for _ in range(layers):
        for q in range(width):
            circuit.h(q).t(q)
        for q in range(width - 1):
            circuit.cnot(q, q + 1)
    for q in range(width):
        circuit.h(q)
    return circuit


@needs_clifft
class TestSharedExecution:
    """Clifft's shared execution, which it selects for many-T circuits."""

    def test_selected_and_recorded(self) -> None:
        record = _run(_many_t(8, 12), shots=2000, seed=3).metadata["reproducibility"]
        assert record["execution"]["executor"] == "shared"
        assert record["execution"]["sharing"] is True

    def test_distribution_within_sampling_bound(self) -> None:
        circuit = _many_t(8, 12)
        shots = 50000
        result = _run(circuit, shots=shots, seed=17)
        assert result.metadata["reproducibility"]["execution"]["executor"] == "shared"
        _assert_within_sampling_bound(result.counts, _exact_probabilities(circuit), shots)

    def test_same_seed_replays(self) -> None:
        first = _run(_many_t(8, 12), shots=2000, seed=21)
        second = _run(_many_t(8, 12), shots=2000, seed=21)
        assert np.array_equal(first.raw_result, second.raw_result)


@needs_clifft
class TestReproducibilityRecord:
    def test_record_contents(self) -> None:
        import clifft

        circuit = Circuit().h(0).t(0).h(0).cnot(0, 1)
        result = _run(circuit, shots=100, seed=7)
        record = result.metadata["reproducibility"]

        json.dumps(record)  # JSON-serialisable
        assert record["clifft_version"] == clifft.__version__ == CLIFFT_VERSION
        assert record["clifft_runtime_isa"] == str(clifft.runtime_isa())
        assert record["shots"] == 100
        assert record["seed"] == result.metadata["seed"] == 7
        assert record["threads"] == 1 and record["batch_size"] == "auto"
        assert record["sharing"] == "auto"
        assert set(record["execution"]) == {
            "executor", "batch_size", "sharing", "shot_workers", "intra_shot_workers",
            "noise_preparation",
        }
        assert record["execution"]["executor"] in {"scalar", "packed", "shared"}
        assert record["program_text"].endswith("\nM 0 1")
        assert record["peak_active_width"] == clifft.compile(record["program_text"]).peak_active_width
        assert record["peak_active_width"] >= 1  # H T H cannot be folded into the measurement

    def test_clifford_circuit_has_zero_active_width(self) -> None:
        record = _run(ghz_state(8), shots=10, seed=0).metadata["reproducibility"]
        assert record["peak_active_width"] == 0


class TestOptions:
    def test_unknown_option_is_refused(self) -> None:
        with pytest.raises(TypeError, match="unsupported option"):
            _run(Circuit().h(0), noise_model="depolarizing")

    @pytest.mark.parametrize("shots", [0, -1])
    def test_nonpositive_shots(self, shots: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _run(Circuit().h(0), shots=shots)

    @pytest.mark.parametrize("shots", [1.5, True, "10"])
    def test_non_integer_shots(self, shots) -> None:
        with pytest.raises(TypeError, match="positive integer"):
            _run(Circuit().h(0), shots=shots)

    @pytest.mark.parametrize("seed", [-1, 2**63])
    def test_out_of_range_seed(self, seed: int) -> None:
        with pytest.raises(ValueError, match="seed"):
            _run(Circuit().h(0), seed=seed)

    def test_non_integer_seed(self) -> None:
        with pytest.raises(TypeError, match="seed"):
            _run(Circuit().h(0), seed=1.0)

    def test_invalid_config_seed(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            ClifftExecutor(ClifftExecutorConfig(seed=-5))

    def test_empty_circuit(self) -> None:
        with pytest.raises(ValueError, match="no qubits"):
            _run(Circuit())

    def test_non_circuit_input(self) -> None:
        with pytest.raises(TypeError, match="Marqov Circuit"):
            _run("H 0\nM 0")  # type: ignore[arg-type]
