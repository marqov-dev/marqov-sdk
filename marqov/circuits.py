"""Backend-agnostic quantum circuit abstraction.

This module wraps QuantumFlow to provide a simple, fluent API for
building quantum circuits that can be transpiled to any backend.

Example:
    >>> from marqov.circuits import Circuit
    >>> circuit = Circuit().h(0).cnot(0, 1)
    >>> braket_circ = circuit.to_braket()
    >>> state = circuit.simulate()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import quantumflow as qf

if TYPE_CHECKING:
    from braket.circuits import Circuit as BraketCircuit
    from pyquil import Program as PyQuilProgram


class Circuit:
    """Backend-agnostic quantum circuit.

    Provides a fluent API for building circuits that can be converted
    to any supported backend (Braket, Qiskit, Cirq, PyQuil).

    Example:
        >>> circuit = Circuit()
        >>> circuit.h(0).cnot(0, 1)  # Bell state
        >>> result = circuit.simulate()
    """

    def __init__(self) -> None:
        """Initialize an empty circuit."""
        self._qf = qf.Circuit()

    # Single-qubit gates

    def h(self, qubit: int) -> Circuit:
        """Apply Hadamard gate.

        Args:
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.H(qubit)
        return self

    def x(self, qubit: int) -> Circuit:
        """Apply Pauli-X (NOT) gate.

        Args:
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.X(qubit)
        return self

    def y(self, qubit: int) -> Circuit:
        """Apply Pauli-Y gate.

        Args:
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.Y(qubit)
        return self

    def z(self, qubit: int) -> Circuit:
        """Apply Pauli-Z gate.

        Args:
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.Z(qubit)
        return self

    def s(self, qubit: int) -> Circuit:
        """Apply S (phase) gate.

        Args:
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.S(qubit)
        return self

    def t(self, qubit: int) -> Circuit:
        """Apply T gate.

        Args:
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.T(qubit)
        return self

    # Rotation gates

    def rx(self, angle: float, qubit: int) -> Circuit:
        """Apply Rx rotation gate.

        Args:
            angle: Rotation angle in radians.
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.Rx(angle, qubit)
        return self

    def ry(self, angle: float, qubit: int) -> Circuit:
        """Apply Ry rotation gate.

        Args:
            angle: Rotation angle in radians.
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.Ry(angle, qubit)
        return self

    def rz(self, angle: float, qubit: int) -> Circuit:
        """Apply Rz rotation gate.

        Args:
            angle: Rotation angle in radians.
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.Rz(angle, qubit)
        return self

    # Two-qubit gates

    def cnot(self, control: int, target: int) -> Circuit:
        """Apply CNOT (CX) gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.CNot(control, target)
        return self

    def cx(self, control: int, target: int) -> Circuit:
        """Apply CX gate (alias for cnot).

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self for method chaining.
        """
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> Circuit:
        """Apply CZ gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.CZ(control, target)
        return self

    def swap(self, qubit0: int, qubit1: int) -> Circuit:
        """Apply SWAP gate.

        Args:
            qubit0: First qubit index.
            qubit1: Second qubit index.

        Returns:
            Self for method chaining.
        """
        self._qf += qf.Swap(qubit0, qubit1)
        return self

    # Backend conversion methods

    def to_braket(self) -> BraketCircuit:
        """Convert to Amazon Braket circuit.

        Returns:
            Braket Circuit object ready for execution.
        """
        return qf.circuit_to_braket(self._qf, translate=True)

    def to_qiskit(self):
        """Convert to IBM Qiskit circuit.

        Returns:
            Qiskit QuantumCircuit object.
        """
        return qf.transpile(self._qf, output_format="qiskit")

    def to_cirq(self):
        """Convert to Google Cirq circuit.

        Returns:
            Cirq Circuit object.
        """
        return qf.transpile(self._qf, output_format="cirq")

    def to_pyquil(self):
        """Convert to Rigetti PyQuil program.

        Returns:
            PyQuil Program object.
        """
        return qf.transpile(self._qf, output_format="pyquil")

    def to_pytket(self):
        """Convert to pytket Circuit.

        Returns:
            pytket Circuit object.

        Raises:
            ImportError: If pytket is not installed.
            NotImplementedError: If the circuit contains an unsupported gate.
        """
        try:
            from pytket.extensions.qiskit import qiskit_to_tk
        except ImportError:
            raise ImportError(
                "pytket is required for Circuit.to_pytket(). "
                "Install with: pip install pytket pytket-qiskit"
            )

        qiskit_circuit = self.to_qiskit()

        for instruction in qiskit_circuit.data:
            name = instruction.operation.name
            if name in self._SKIP_INSTRUCTIONS:
                continue
            if name not in self._QISKIT_GATE_MAP:
                raise NotImplementedError(
                    f"Unsupported gate '{name}' after decomposition. "
                    f"Supported gates: {', '.join(sorted(self._QISKIT_GATE_MAP))}"
                )

        return qiskit_to_tk(qiskit_circuit)

    def to_openqasm(self, version: int = 2) -> str:
        """Export circuit as an OpenQASM string.

        Args:
            version: QASM version to export (2 or 3). Defaults to 2.

        Returns:
            OpenQASM string representation.

        Raises:
            ImportError: If qiskit is not installed.
            ValueError: If version is not 2 or 3.
        """
        try:
            from qiskit import qasm2, qasm3
        except ImportError:
            raise ImportError(
                "Qiskit is required for to_openqasm(). "
                "Install with: pip install marqov[openqasm]"
            )

        if version not in (2, 3):
            raise ValueError(f"Unsupported QASM version: {version}. Must be 2 or 3.")

        qiskit_circuit = self.to_qiskit()
        if version == 2:
            return qasm2.dumps(qiskit_circuit)
        return qasm3.dumps(qiskit_circuit)

    # Simulation

    def simulate(self) -> qf.State:
        """Run circuit on local simulator.

        Returns:
            QuantumFlow State object with simulation results.
        """
        return self._qf.run()

    # Import methods

    @classmethod
    def from_braket(cls, braket_circuit: BraketCircuit) -> Circuit:
        """Import from existing Braket circuit.

        Args:
            braket_circuit: Braket Circuit to import.

        Returns:
            New Circuit instance.
        """
        circuit = cls()
        circuit._qf = qf.braket_to_circuit(braket_circuit)
        return circuit

    # Qiskit gate name -> Circuit fluent method mapping.
    # Used by from_qiskit() to convert decomposed Qiskit circuits.
    _QISKIT_GATE_MAP: dict[str, str] = {
        "h": "h",
        "x": "x",
        "y": "y",
        "z": "z",
        "s": "s",
        "t": "t",
        "rx": "rx",
        "ry": "ry",
        "rz": "rz",
        "cx": "cnot",
        "cz": "cz",
        "swap": "swap",
    }

    # Basis gates for Qiskit transpiler decomposition.
    _QISKIT_BASIS_GATES: list[str] = list(_QISKIT_GATE_MAP.keys())

    # Gates that take rotation angle parameters.
    _ROTATION_GATES: set[str] = {"rx", "ry", "rz"}

    # Non-gate instructions to skip silently. `measure` is handled separately
    # since it is only implicit (and thus safe to skip) when terminal; `reset`
    # and `delay` are never implicit and always raise (see
    # `_qiskit_measurement_is_terminal`).
    _SKIP_INSTRUCTIONS: set[str] = {"barrier"}

    @classmethod
    def _qiskit_measurement_is_terminal(cls, data, index: int, qubit, clbit) -> bool:
        """Whether a measurement at ``data[index]`` is terminal.

        A measurement is terminal iff nothing after it touches its qubit and
        nothing after it reads its classical bit — the latter check catches
        classically-conditioned operations on a *different* qubit (e.g.
        teleportation's correction step), not just gates on the same qubit.
        Compiler-hint instructions (e.g. `barrier`) are skipped since they
        carry no computational meaning.
        """
        for later in data[index + 1 :]:
            if later.operation.name in cls._SKIP_INSTRUCTIONS:
                continue
            if qubit in later.qubits or clbit in later.clbits:
                return False
        return True

    @classmethod
    def from_qiskit(cls, qiskit_circuit) -> "Circuit":
        """Import from a Qiskit QuantumCircuit.

        Unsupported gates are automatically decomposed into the supported
        basis set via Qiskit's transpiler. This covers any circuit built from
        Qiskit's standard gate set, but a circuit containing an opaque custom
        gate (no `_define`) makes the transpiler itself raise — that error is
        not caught here and propagates to the caller.

        Requires Qiskit to be installed (``pip install marqov[qiskit]``).

        Args:
            qiskit_circuit: A Qiskit ``QuantumCircuit`` instance.

        Returns:
            New Circuit instance.

        Raises:
            ImportError: If Qiskit is not installed.
            TypeError: If the input is not a Qiskit QuantumCircuit.
            ValueError: If a gate cannot be mapped after decomposition, or if
                the circuit contains a mid-circuit measurement, a `reset`, or
                a `delay` — none of these are implicit under this SDK's
                gate-only circuit model, and only a terminal measurement
                (nothing after it touches its qubit or reads its classical
                bit) can be safely dropped.
            qiskit.transpiler.exceptions.TranspilerError: If the transpiler
                cannot decompose a gate (e.g. an opaque custom gate).
        """
        try:
            from qiskit import QuantumCircuit, transpile
        except ImportError:
            raise ImportError(
                "Qiskit is required for Circuit.from_qiskit(). "
                "Install with: pip install marqov[qiskit]"
            )

        if not isinstance(qiskit_circuit, QuantumCircuit):
            raise TypeError(
                f"Expected a Qiskit QuantumCircuit, got {type(qiskit_circuit).__name__}"
            )

        # Decompose to our supported basis gate set.
        decomposed = transpile(
            qiskit_circuit,
            basis_gates=cls._QISKIT_BASIS_GATES,
            optimization_level=0,
        )

        circuit = cls()

        for index, instruction in enumerate(decomposed.data):
            name = instruction.operation.name

            if name in cls._SKIP_INSTRUCTIONS:
                continue

            if name == "measure":
                qubit = instruction.qubits[0]
                clbit = instruction.clbits[0]
                if cls._qiskit_measurement_is_terminal(decomposed.data, index, qubit, clbit):
                    continue
                raise ValueError(
                    f"Circuit.from_qiskit() does not support mid-circuit measurement "
                    f"(qubit {decomposed.find_bit(qubit).index}): a later instruction "
                    "touches this qubit or reads its classical bit, so dropping the "
                    "measurement would silently change what the circuit computes."
                )

            if name in ("reset", "delay"):
                qubit = instruction.qubits[0]
                raise ValueError(
                    f"Circuit.from_qiskit() does not support '{name}' "
                    f"(qubit {decomposed.find_bit(qubit).index}): it is never implicit "
                    "under this SDK's gate-only circuit model."
                )

            if name not in cls._QISKIT_GATE_MAP:
                raise ValueError(
                    f"Unsupported gate '{name}' after decomposition. "
                    f"Supported gates: {', '.join(sorted(cls._QISKIT_GATE_MAP))}"
                )

            qubits = [decomposed.find_bit(q).index for q in instruction.qubits]
            method_name = cls._QISKIT_GATE_MAP[name]

            if name in cls._ROTATION_GATES:
                angle = float(instruction.operation.params[0])
                getattr(circuit, method_name)(angle, qubits[0])
            elif len(qubits) == 1:
                getattr(circuit, method_name)(qubits[0])
            else:
                getattr(circuit, method_name)(qubits[0], qubits[1])

        return circuit

    @classmethod
    def _flatten_cirq_operations(cls, operations) -> list:
        """Flatten operations exactly as the mapping pass will consume them.

        Unrolls `CircuitOperation` subcircuits and decomposes unmappable
        gates recursively, without mutating a real `Circuit` (a throwaway
        instance absorbs `_map_cirq_operation`'s side effects). This must run
        before terminal-measurement detection so "what comes after a
        measurement" is checked against the same flat sequence the mapping
        pass sees — otherwise a measurement nested in a subcircuit, with
        nothing following it, would be invisible to that check and wrongly
        treated as mid-circuit.

        Measurement, wait/delay, and reset gates are kept as opaque leaves:
        measurement and reset must be handled by the explicit policy checks
        below rather than falling into the generic "unsupported gate" path
        by accident, and `decompose_once` on a `WaitGate` returns an empty
        list, not `None`, which would otherwise make it silently disappear
        during flattening.
        """
        import cirq

        scratch = cls()
        flat: list = []

        def _walk(ops) -> None:
            for op in ops:
                if isinstance(op.gate, (cirq.MeasurementGate, cirq.WaitGate, cirq.ResetChannel)):
                    flat.append(op)
                    continue
                if cls._map_cirq_operation(op, scratch):
                    flat.append(op)
                    continue
                decomposed = cirq.decompose_once(op, default=None)
                if decomposed is not None:
                    _walk(decomposed)
                    continue
                flat.append(op)

        _walk(operations)
        return flat

    @classmethod
    def _cirq_terminal_measurement_indices(cls, operations: list) -> set[int]:
        """Return the index of each measurement operation that is terminal.

        A measurement is terminal iff nothing after it touches its qubits and
        nothing after it is classically controlled on its key — the latter
        check catches conditioning on a *different* qubit (e.g.
        teleportation's correction step), not just gates on the same qubit.

        Tracked by position, not `id()`: a repeated `CircuitOperation`
        (`repetitions > 1`) decomposes to the *same* operation object for
        every repetition, so `id()`-based tracking would make an early,
        genuinely mid-circuit repetition match the terminal id of the last
        one — silently reintroducing the bug this check exists to catch.
        """
        import cirq

        terminal_indices = set()
        for index, op in enumerate(operations):
            if not isinstance(op.gate, cirq.MeasurementGate):
                continue

            qubits = set(op.qubits)
            key = op.gate.key
            is_terminal = True
            for later in operations[index + 1 :]:
                if qubits & set(later.qubits):
                    is_terminal = False
                    break
                if any(key in cond.keys for cond in later.classical_controls):
                    is_terminal = False
                    break
            if is_terminal:
                terminal_indices.add(index)
        return terminal_indices

    @classmethod
    def _map_cirq_operation(cls, op, circuit: "Circuit") -> bool:
        """Try to map a single Cirq operation to a Circuit method.

        Args:
            op: A Cirq Operation.
            circuit: The Circuit instance to append the gate to.

        Returns:
            True if the operation was mapped, False if unrecognised.
        """
        import cirq
        import math

        gate = op.gate
        if gate is None:
            return False

        qubits = [q.x for q in op.qubits]

        # Single-qubit fixed gates.
        if isinstance(gate, cirq.HPowGate) and gate.exponent == 1:
            circuit.h(qubits[0])
            return True

        # Power gates — map full-turn to named gate, fractional to rotation.
        for pow_cls, full_method, rot_method in (
            (cirq.XPowGate, "x", "rx"),
            (cirq.YPowGate, "y", "ry"),
            (cirq.ZPowGate, "z", "rz"),
        ):
            if isinstance(gate, pow_cls):
                exp = gate.exponent
                if exp == 1:
                    getattr(circuit, full_method)(qubits[0])
                elif isinstance(gate, cirq.ZPowGate) and exp == 0.5:
                    circuit.s(qubits[0])
                elif isinstance(gate, cirq.ZPowGate) and exp == 0.25:
                    circuit.t(qubits[0])
                else:
                    getattr(circuit, rot_method)(math.pi * exp, qubits[0])
                return True

        # Two-qubit gates — only map full-exponent versions.
        if isinstance(gate, cirq.CXPowGate) and gate.exponent == 1:
            circuit.cnot(qubits[0], qubits[1])
            return True
        if isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
            circuit.cz(qubits[0], qubits[1])
            return True
        if isinstance(gate, cirq.SwapPowGate) and gate.exponent == 1:
            circuit.swap(qubits[0], qubits[1])
            return True

        return False

    @classmethod
    def from_cirq(cls, cirq_circuit) -> "Circuit":
        """Import from a Cirq Circuit.

        Known gates are mapped directly. Unsupported gates are automatically
        decomposed via ``cirq.decompose()`` until they resolve to gates we
        can map. If decomposition fails, a ``ValueError`` is raised and the
        gate type is reported to Sentry for observability.

        Requires Cirq to be installed (``pip install marqov[cirq]``).

        Only ``cirq.LineQubit`` circuits are accepted. For ``GridQubit`` or
        ``NamedQubit`` circuits, convert to ``LineQubit`` first.

        Args:
            cirq_circuit: A Cirq ``Circuit`` instance using ``LineQubit`` qubits.

        Returns:
            New Circuit instance.

        Raises:
            ImportError: If Cirq is not installed.
            TypeError: If the input is not a Cirq Circuit or uses non-LineQubit qubits.
            ValueError: If a gate cannot be mapped after decomposition, or if
                the circuit contains a mid-circuit measurement, a `reset`, or
                a `wait`/delay — none of these are implicit under this SDK's
                gate-only circuit model, and only a terminal measurement
                (nothing after it touches its qubits or is classically
                controlled on its key) can be safely dropped.
        """
        try:
            import cirq
        except ImportError:
            raise ImportError(
                "Cirq is required for Circuit.from_cirq(). "
                "Install with: pip install marqov[cirq]"
            )

        if not isinstance(cirq_circuit, cirq.Circuit):
            raise TypeError(
                f"Expected a Cirq Circuit, got {type(cirq_circuit).__name__}"
            )

        # Validate qubit types — only LineQubit is supported.
        for qubit in cirq_circuit.all_qubits():
            if not isinstance(qubit, cirq.LineQubit):
                raise TypeError(
                    f"Circuit.from_cirq() requires LineQubit qubits, but found "
                    f"{type(qubit).__name__}. Convert your circuit to use "
                    f"cirq.LineQubit before calling from_cirq()."
                )

        circuit = cls()
        flat_operations = cls._flatten_cirq_operations(list(cirq_circuit.all_operations()))
        terminal_measurement_indices = cls._cirq_terminal_measurement_indices(flat_operations)

        for index, op in enumerate(flat_operations):
            if isinstance(op.gate, cirq.MeasurementGate):
                if index in terminal_measurement_indices:
                    continue
                raise ValueError(
                    f"Circuit.from_cirq() does not support mid-circuit measurement "
                    f"(qubits {[q.x for q in op.qubits]}): a later operation touches "
                    "these qubits or is classically controlled on this measurement's "
                    "key, so dropping it would silently change what the circuit computes."
                )

            if isinstance(op.gate, cirq.WaitGate):
                raise ValueError(
                    f"Circuit.from_cirq() does not support 'wait'/delay "
                    f"(qubits {[q.x for q in op.qubits]}): it is never implicit "
                    "under this SDK's gate-only circuit model."
                )

            if isinstance(op.gate, cirq.ResetChannel):
                raise ValueError(
                    f"Circuit.from_cirq() does not support 'reset' "
                    f"(qubits {[q.x for q in op.qubits]}): it is never implicit "
                    "under this SDK's gate-only circuit model."
                )

            if cls._map_cirq_operation(op, circuit):
                continue

            # Leaf op that survived flattening unmapped and undecomposable.
            gate_type = type(op.gate).__name__ if op.gate else type(op).__name__
            try:
                import sentry_sdk
                sentry_sdk.capture_message(
                    f"Circuit.from_cirq(): unmappable gate '{gate_type}'",
                    level="warning",
                )
            except ImportError:
                pass

            raise ValueError(
                f"Unsupported Cirq gate '{gate_type}' could not be decomposed. "
                f"Decompose it manually before calling from_cirq(), or file an "
                f"issue at https://github.com/marqov-dev/marqov-sdk/issues"
            )

        return circuit

    # PennyLane gate name -> Circuit fluent method mapping.
    _PENNYLANE_GATE_MAP: dict[str, str] = {
        "Hadamard": "h",
        "PauliX": "x",
        "PauliY": "y",
        "PauliZ": "z",
        "S": "s",
        "T": "t",
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
        "CNOT": "cnot",
        "CZ": "cz",
        "SWAP": "swap",
    }

    _PENNYLANE_ROTATION_GATES: set[str] = {"RX", "RY", "RZ"}

    _PENNYLANE_SKIP: set[str] = {"Barrier"}

    @classmethod
    def from_pennylane(cls, tape) -> "Circuit":
        """Import from a PennyLane QuantumTape or QuantumScript.

        Known gates are mapped directly. Unsupported gates are automatically
        decomposed via ``op.decomposition()`` until they resolve to gates we
        can map. If decomposition fails, a ``ValueError`` is raised and the
        gate type is reported to Sentry for observability.

        Requires PennyLane to be installed (``pip install marqov[pennylane]``).

        Only integer wires are accepted. For string wires, remap to integers
        before calling this method.

        Args:
            tape: A PennyLane ``QuantumTape`` or ``QuantumScript`` instance.

        Returns:
            New Circuit instance.

        Raises:
            ImportError: If PennyLane is not installed.
            TypeError: If the input is not a QuantumScript or uses non-integer wires.
            ValueError: If a gate cannot be mapped after decomposition.
        """
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError(
                "PennyLane is required for Circuit.from_pennylane(). "
                "Install with: pip install marqov[pennylane]"
            )

        if not isinstance(tape, qml.tape.QuantumScript):
            raise TypeError(
                f"Expected a PennyLane QuantumTape or QuantumScript, "
                f"got {type(tape).__name__}"
            )

        # Validate wire types — only integer wires are supported.
        for wire in tape.wires:
            if not isinstance(wire, int):
                raise TypeError(
                    f"Circuit.from_pennylane() requires integer wires, but found "
                    f"{type(wire).__name__} wire '{wire}'. Remap your tape to use "
                    f"integer wires before calling from_pennylane()."
                )

        circuit = cls()

        def _process_operations(operations) -> None:
            """Map operations, decomposing unknowns recursively."""
            for op in operations:
                name = op.name

                if name in cls._PENNYLANE_SKIP:
                    continue

                wires = op.wires.tolist()

                if name in cls._PENNYLANE_GATE_MAP:
                    method_name = cls._PENNYLANE_GATE_MAP[name]
                    if name in cls._PENNYLANE_ROTATION_GATES:
                        angle = float(op.parameters[0])
                        getattr(circuit, method_name)(angle, wires[0])
                    elif len(wires) == 1:
                        getattr(circuit, method_name)(wires[0])
                    else:
                        getattr(circuit, method_name)(wires[0], wires[1])
                    continue

                # Unknown gate — try decomposing.
                try:
                    decomposed = op.decomposition()
                    _process_operations(decomposed)
                    continue
                except Exception:
                    pass

                # Decomposition failed — log to Sentry and raise.
                try:
                    import sentry_sdk
                    sentry_sdk.capture_message(
                        f"Circuit.from_pennylane(): unmappable gate '{name}'",
                        level="warning",
                    )
                except ImportError:
                    pass

                raise ValueError(
                    f"Unsupported PennyLane gate '{name}' could not be decomposed. "
                    f"Decompose it manually before calling from_pennylane(), or file "
                    f"an issue at https://github.com/marqov-dev/marqov-sdk/issues"
                )

        # tape.operations excludes measurements — no skip logic needed.
        _process_operations(tape.operations)
        return circuit

    # PyQuil gate name -> Circuit fluent method mapping.
    _PYQUIL_GATE_MAP: dict[str, str] = {
        "H": "h",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "S": "s",
        "T": "t",
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
        "CNOT": "cnot",
        "CX": "cnot",
        "CZ": "cz",
        "SWAP": "swap",
    }

    _PYQUIL_ROTATION_GATES: set[str] = {"RX", "RY", "RZ"}

    # Classical declarations do not affect the unitary circuit representation
    # imported by this SDK. `Measurement` is handled separately since it is
    # only implicit (and thus safe to skip) when terminal. `Reset` and
    # `Delay` are deliberately absent from this set: neither is a `Gate` nor
    # in the skip set, so they fall through to the generic "unsupported
    # instruction" `NotImplementedError` below — always raising, since
    # neither is ever implicit.
    _PYQUIL_SKIP_INSTRUCTIONS: set[str] = {"Declare", "Halt", "Pragma"}

    @classmethod
    def _pyquil_measurement_is_terminal(cls, instructions: list, index: int, qubits: set[int]) -> bool:
        """Whether a Measurement at ``instructions[index]`` is terminal.

        Terminal iff nothing after it touches the measured qubit. (Any
        classical control-flow reading the measured bit already raises
        `NotImplementedError` on its own, as an unrecognized instruction type,
        before this check would even matter.)
        """
        for later in instructions[index + 1 :]:
            get_qubit_indices = getattr(later, "get_qubit_indices", None)
            if get_qubit_indices is not None and qubits & set(get_qubit_indices()):
                return False
        return True

    @classmethod
    def _pyquil_float_param(cls, value) -> float:
        """Return a real-valued PyQuil gate parameter."""
        try:
            numeric = complex(value)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError(
                "Circuit.from_pyquil() only supports numeric real-valued "
                "rotation parameters."
            ) from exc

        if abs(numeric.imag) > 1e-12:
            raise NotImplementedError(
                "Circuit.from_pyquil() only supports real-valued rotation parameters."
            )
        return float(numeric.real)

    @classmethod
    def _try_consume_swap_from_cnots(cls, instructions: list, index: int) -> tuple[int, int, int] | None:
        """Detect a three-CNOT SWAP decomposition and return (q0, q1, consumed)."""
        if index + 2 >= len(instructions):
            return None

        gates = instructions[index : index + 3]
        try:
            from pyquil.quilbase import Gate
        except ImportError:
            return None

        if not all(isinstance(gate, Gate) for gate in gates):
            return None

        names = [gate.name.upper() for gate in gates]
        if names != ["CNOT", "CNOT", "CNOT"]:
            return None

        q0, q1 = gates[0].get_qubit_indices()
        if (
            gates[1].get_qubit_indices() == [q1, q0]
            and gates[2].get_qubit_indices() == [q0, q1]
        ):
            return q0, q1, 3

        return None

    @classmethod
    def from_pyquil(cls, program: "PyQuilProgram") -> "Circuit":
        """Import from a PyQuil Program.

        Known gates in the Marqov canonical gate set are mapped directly.
        SWAP is accepted either as a native ``SWAP`` instruction or as the
        standard three-``CNOT`` decomposition. Classical declarations and
        terminal measurements (nothing after them touches the measured qubit)
        are skipped. Quil-native gates outside the canonical set, and
        mid-circuit measurements, raise ``NotImplementedError`` so callers can
        decompose them explicitly before importing.

        Requires PyQuil to be installed (``pip install marqov[pyquil]``).

        Args:
            program: A PyQuil ``Program`` instance.

        Returns:
            New Circuit instance.

        Raises:
            ImportError: If PyQuil is not installed.
            TypeError: If the input is not a PyQuil Program or uses non-integer qubits.
            NotImplementedError: If a Quil instruction cannot be mapped.
        """
        try:
            from pyquil import Program
            from pyquil.quilbase import Gate, Measurement
        except ImportError:
            raise ImportError(
                "PyQuil is required for Circuit.from_pyquil(). "
                "Install with: pip install marqov[pyquil]"
            )

        if not isinstance(program, Program):
            raise TypeError(f"Expected a PyQuil Program, got {type(program).__name__}")

        circuit = cls()
        instructions = list(program.instructions)
        index = 0

        while index < len(instructions):
            swap_match = cls._try_consume_swap_from_cnots(instructions, index)
            if swap_match is not None:
                q0, q1, consumed = swap_match
                circuit.swap(q0, q1)
                index += consumed
                continue

            instruction = instructions[index]
            this_index = index
            index += 1

            if not isinstance(instruction, Gate):
                if isinstance(instruction, Measurement):
                    measured_qubits = set(instruction.get_qubit_indices())
                    if cls._pyquil_measurement_is_terminal(instructions, this_index, measured_qubits):
                        continue
                    raise NotImplementedError(
                        f"Circuit.from_pyquil() does not support mid-circuit measurement "
                        f"(qubits {sorted(measured_qubits)}): a later instruction touches "
                        "these qubits, so dropping the measurement would silently change "
                        "what the circuit computes."
                    )

                instruction_type = type(instruction).__name__
                if instruction_type in cls._PYQUIL_SKIP_INSTRUCTIONS:
                    continue
                raise NotImplementedError(
                    f"Unsupported PyQuil instruction '{instruction_type}'. "
                    "Only canonical gates plus declarations and measurements are supported."
                )

            name = instruction.name.upper()
            if name not in cls._PYQUIL_GATE_MAP:
                raise NotImplementedError(
                    f"Unsupported PyQuil gate '{name}'. Supported gates: "
                    f"{', '.join(sorted(cls._PYQUIL_GATE_MAP))}"
                )

            qubits = list(instruction.get_qubit_indices())
            method_name = cls._PYQUIL_GATE_MAP[name]

            if name in cls._PYQUIL_ROTATION_GATES:
                angle = cls._pyquil_float_param(instruction.params[0])
                getattr(circuit, method_name)(angle, qubits[0])
            elif len(qubits) == 1:
                getattr(circuit, method_name)(qubits[0])
            else:
                getattr(circuit, method_name)(qubits[0], qubits[1])

        return circuit

    @classmethod
    def from_openqasm(cls, qasm_string: str) -> Circuit:
        """Import a circuit from an OpenQASM 2.0 or 3.0 string.

        Auto-detects the QASM version from the header and parses via Qiskit.

        Args:
            qasm_string: OpenQASM program string.

        Returns:
            New Circuit instance.

        Raises:
            ImportError: If qiskit is not installed.
            ValueError: If the QASM string cannot be parsed.
        """
        try:
            from qiskit import qasm2, qasm3
        except ImportError:
            raise ImportError(
                "Qiskit is required for Circuit.from_openqasm(). "
                "Install with: pip install marqov[openqasm]"
            )

        stripped = qasm_string.strip()
        if stripped.startswith("OPENQASM 3"):
            qiskit_circuit = qasm3.loads(stripped)
        else:
            qiskit_circuit = qasm2.loads(stripped)

        return cls.from_qiskit(qiskit_circuit)

    # Serialization methods

    def to_dict(self) -> dict:
        """Serialize circuit to a dictionary.

        The dictionary contains the gate sequence for reconstruction. Not used
        by Temporal activities today — those serialize `Circuit` (and
        everything else) opaquely via cloudpickle, not through this method
        (see `marqov/workflows/activity.py`). This is a plain gate-sequence
        dump for callers who want a JSON-friendly representation.

        Returns:
            Dictionary representation of the circuit.
        """
        gates = []
        for op in self._qf._elements:
            gate_name = op.name
            qubits = list(op.qubits)
            params = list(op.params) if hasattr(op, "params") and op.params else []
            gates.append({
                "gate": gate_name,
                "qubits": qubits,
                "params": params,
            })
        return {"gates": gates}

    @classmethod
    def from_dict(cls, data: dict) -> Circuit:
        """Reconstruct circuit from dictionary.

        Args:
            data: Dictionary from to_dict().

        Returns:
            Reconstructed Circuit instance.
        """
        circuit = cls()
        gate_map = {
            "H": lambda q, p: circuit.h(q[0]),
            "X": lambda q, p: circuit.x(q[0]),
            "Y": lambda q, p: circuit.y(q[0]),
            "Z": lambda q, p: circuit.z(q[0]),
            "S": lambda q, p: circuit.s(q[0]),
            "T": lambda q, p: circuit.t(q[0]),
            "Rx": lambda q, p: circuit.rx(p[0], q[0]),
            "Ry": lambda q, p: circuit.ry(p[0], q[0]),
            "Rz": lambda q, p: circuit.rz(p[0], q[0]),
            "CNot": lambda q, p: circuit.cnot(q[0], q[1]),
            "CZ": lambda q, p: circuit.cz(q[0], q[1]),
            "Swap": lambda q, p: circuit.swap(q[0], q[1]),
        }
        for gate_data in data.get("gates", []):
            gate_name = gate_data["gate"]
            qubits = gate_data["qubits"]
            params = gate_data.get("params", [])
            if gate_name in gate_map:
                gate_map[gate_name](qubits, params)
        return circuit

    # Utility methods

    @property
    def num_qubits(self) -> int:
        """Return number of qubits in circuit."""
        return self._qf.qubit_nb

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Circuit(qubits={self.num_qubits}, gates={len(self._qf)})"


# Convenience constructors


def bell_state() -> Circuit:
    """Create a Bell state circuit (|00⟩ + |11⟩)/√2.

    Returns:
        Circuit that creates a Bell state.
    """
    return Circuit().h(0).cnot(0, 1)


def ghz_state(num_qubits: int) -> Circuit:
    """Create a GHZ state circuit.

    Args:
        num_qubits: Number of qubits in the GHZ state.

    Returns:
        Circuit that creates a GHZ state.
    """
    circuit = Circuit().h(0)
    for i in range(num_qubits - 1):
        circuit.cnot(i, i + 1)
    return circuit
