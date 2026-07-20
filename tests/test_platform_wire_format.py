"""Regression guard: QASM3 is a lossless wire format for marqov.Circuit.

Verifies both semantic equivalence (unitary operator) and structural
equivalence (gate sequence + params) for the full canonical gate set.
"""

import pytest

pytest.importorskip("qiskit")

from marqov import Circuit  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

CASES = {
    "H": Circuit().h(0),
    "X": Circuit().x(0),
    "Y": Circuit().y(0),
    "Z": Circuit().z(0),
    "S": Circuit().s(0),
    "T": Circuit().t(0),
    "Rx": Circuit().rx(0.7, 0),
    "Ry": Circuit().ry(1.3, 0),
    "Rz": Circuit().rz(2.1, 0),
    "CNot": Circuit().cnot(0, 1),
    "CZ": Circuit().cz(0, 1),
    "Swap": Circuit().swap(0, 1),
    "bell": Circuit().h(0).cnot(0, 1),
    "ghz3": Circuit().h(0).cnot(0, 1).cnot(1, 2),
    "mixed": Circuit().h(0).rx(0.5, 0).ry(1.1, 1).cz(0, 1).rz(0.3, 1).swap(0, 1).t(0).s(1),
}


@pytest.mark.parametrize("name", list(CASES))
def test_qasm3_roundtrip_is_semantically_lossless(name):
    c = CASES[name]
    rt = Circuit.from_openqasm(c.to_openqasm(version=3))
    assert Operator(c.to_qiskit()).equiv(Operator(rt.to_qiskit())), (
        f"{name}: unitary changed (semantic — the invariant)"
    )
    g1, g2 = c.to_dict()["gates"], rt.to_dict()["gates"]
    assert [(g["gate"], g["qubits"]) for g in g1] == [(g["gate"], g["qubits"]) for g in g2], (
        f"{name}: gate sequence changed"
    )
    for a, b in zip(g1, g2):
        assert a["params"] == pytest.approx(b["params"]), f"{name}: params changed"
