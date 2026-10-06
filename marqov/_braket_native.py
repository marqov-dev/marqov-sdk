"""Explicit Rigetti-basis export and shared Braket submission preparation."""

import math
import re
from numbers import Integral, Real
from typing import Any

import numpy as np
import quantumflow as qf

from marqov._optional import require_braket


def native_braket(circuit):
    """Export native gates without translating or compacting source labels."""
    require_braket()
    from braket.circuits import Circuit as BraketCircuit

    output = BraketCircuit()
    gates = {qf.Rx: ("rx", 1, 1), qf.Rz: ("rz", 1, 1), qf.CZ: ("cz", 2, 0), qf.XY: ("xy", 2, 1)}
    for index, operation in enumerate(circuit):
        spec = gates.get(type(operation))
        if spec is None:
            raise ValueError(
                f"unsupported native gate '{operation.name}' at index {index}; "
                "Rigetti basis requires Rx/Rz/CZ/XY without translation"
            )
        name, arity, parameter_count = spec
        qubits = operation.qubits
        if (
            len(qubits) != arity
            or not all(
                isinstance(q, Integral) and not isinstance(q, (bool, np.bool_)) and q >= 0
                for q in qubits
            )
            or len(set(qubits)) != arity
        ):
            raise ValueError(
                f"native gate '{operation.name}' at index {index} requires "
                "distinct nonnegative integer qubit labels"
            )
        params = operation.params
        message = (
            f"native gate '{operation.name}' at index {index} requires "
            "finite real numeric parameters"
        )
        if len(params) != parameter_count or not all(
            isinstance(p, Real) and not isinstance(p, (bool, np.bool_)) for p in params
        ):
            raise ValueError(message)
        try:
            angles = [float(p) for p in params]
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(message) from error
        if not all(math.isfinite(angle) for angle in angles):
            raise ValueError(message)
        if name == "xy":
            # QuantumFlow XY(t) uses turns and the opposite Braket sign.
            angles = [-2 * math.pi * angles[0]]
            if not math.isfinite(angles[0]):
                raise ValueError(f"native gate 'XY' at index {index} has an unrepresentable angle")
        getattr(output, name)(*[int(q) for q in qubits], *angles)
    if not output.instructions:
        raise ValueError("native Braket export requires a nonempty circuit")
    return output


def validate_options(preserve_qubit_labels: bool, options: dict[str, Any], device_arn: str | None):
    """Validate explicit intent before provider initialization or submission."""
    flags = {"preserve_qubit_labels": preserve_qubit_labels}
    flags.update(
        {key: options[key] for key in ("verbatim", "disable_qubit_rewiring") if key in options}
    )
    for key, value in flags.items():
        if type(value) is not bool:
            raise ValueError(f"{key} must be a bool")
    if preserve_qubit_labels:
        if not options.get("verbatim", False) or not options.get("disable_qubit_rewiring", False):
            raise ValueError(
                "preserve_qubit_labels=True requires verbatim=True and disable_qubit_rewiring=True"
            )
        # Structural provider/resource intent only, not device existence/capability.
        if (
            not isinstance(device_arn, str)
            or re.fullmatch(
                r"arn:aws:braket:[a-z]{2}(?:-[a-z]+)+-[0-9]+::device/qpu/rigetti/"
                r"[A-Za-z0-9][A-Za-z0-9._-]*",
                device_arn,
            )
            is None
        ):
            raise ValueError(
                "preserve_qubit_labels=True is supported only for configured "
                "AWS Braket Rigetti QPU ARNs; hardware capability is unqualified"
            )


def prepare_braket(
    circuit, *, preserve_qubit_labels, options, device_arn, circuit_factory=None, converted=None
):
    """Prepare one submission using the same validation in both SDK paths."""
    validate_options(preserve_qubit_labels, options, device_arn)
    output = (
        circuit.to_braket_native()
        if preserve_qubit_labels
        else converted
        if converted is not None
        else circuit.to_braket()
    )
    if options.get("verbatim", False):
        non_native = [
            instruction.operator.name
            for instruction in output.instructions
            if instruction.operator.name.lower() not in {"rx", "rz", "cz", "xy"}
        ]
        if non_native:
            raise ValueError(
                "verbatim=True requires Rigetti native gates only "
                "(1Q: Rx/Rz, 2Q: CZ/XY; no explicit Measure — Braket cannot box a "
                "measured subcircuit and adds measurement implicitly via shots). "
                f"Found non-native gates: {sorted(set(non_native))}. "
                "Use clifford_to_circuit_native() or set SRBConfig.use_native_gates=True."
            )
        if circuit_factory is None:
            require_braket()
            from braket.circuits import Circuit as circuit_factory
        output = circuit_factory().add_verbatim_box(output)
    return output
