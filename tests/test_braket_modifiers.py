"""Instruction modifiers must not silently change imported computations."""

from unittest.mock import patch

import numpy as np
import pytest

from marqov import Circuit

bk = pytest.importorskip("braket.circuits")


@pytest.mark.parametrize("target, modifiers", [
    (1, {"control": 0}),
    (0, {"control": 1}),
    (1, {"control": 0, "control_state": "0"}),
    (2, {"control": [1, 0], "control_state": "01"}),
    (1, {"power": 0.5}),
    (0, {"power": -1}),
    (0, {"power": 0}),
    (1, {"power": 2}),
    (1, {"control": 0, "power": -0.5}),
])
def test_modified_instructions_rejected_before_conversion(target, modifiers):
    source = bk.Circuit().h(0).rx(target, 0.37, **modifiers).x(0)
    with patch("marqov.circuits.qf.braket_to_circuit") as converter:
        with pytest.raises(NotImplementedError, match="gate 'Rx' at index 1"):
            Circuit.from_braket(source)
        converter.assert_not_called()


@pytest.mark.parametrize("source", [
    bk.Circuit().x(1, control=0),
    bk.Circuit().x(0, power=0.5),
])
def test_reported_silent_loss_cases_rejected(source):
    with pytest.raises(NotImplementedError, match="unsupported modifiers.*gate 'X'"):
        Circuit.from_braket(source)


@pytest.mark.parametrize("angle", [0.37, -1.1, 4.0])
@pytest.mark.parametrize("control, target", [(0, 1), (1, 0)])
def test_unmodified_import_matches_source_operator(angle, control, target):
    source = (bk.Circuit().h(control).ry(target, angle).t(control)
              .cnot(control, target).xx(target, control, angle).rz(control, -angle))
    restored = Circuit.from_braket(source)
    # Both operators order axes by ascending active wire labels, most significant
    # first. Explicit CNot is an ordinary two-wire gate, not an extra control.
    assert tuple(restored._qf.qubits) == (0, 1)
    assert np.allclose(restored._qf.asgate().asoperator(), source.to_unitary(), atol=1e-12)


def test_explicit_default_power_unchanged():
    source = bk.Circuit().rx(0, 0.37, power=1)
    assert Circuit.from_braket(source).to_dict() == Circuit().rx(0.37, 0).to_dict()
