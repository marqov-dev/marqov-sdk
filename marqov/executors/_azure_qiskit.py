"""The program and counts shared by both Azure Quantum paths.

``AzureQuantumExecutor`` and the Azure branch of ``MarqovDevice`` both submit
through azure-quantum's Qiskit provider:
``AzureQuantumProvider(workspace).get_backend(target).run(circuit, shots=...)``.
This module prepares the circuit they pass and reads the counts back, so
the two paths send the same program and report the same keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marqov.circuits import Circuit

#: Name of the classical register the measurements are written to.
MEASUREMENT_REGISTER = "meas"

# Providers whose Azure targets take no gate-model circuit. azure-quantum
# 3.12.0 builds no Qiskit or Cirq backend for them, so get_backend would
# report the target as missing from the workspace.
_REFUSED_PROVIDERS = {
    "pasqal": (
        "PASQAL targets take pulse-level Pulser programs (pasqal.pulser.v1), not "
        "gate circuits, and azure-quantum's Qiskit and Cirq providers have no "
        "backend for them"
    ),
}


def refuse_unsupported_target(target: str) -> None:
    """Refuse a target this SDK cannot submit a circuit to, before Azure is contacted.

    Raises:
        ValueError: Naming the target and why it cannot run a circuit.
    """
    reason = _REFUSED_PROVIDERS.get(target.split(".", 1)[0].strip().lower())
    if reason is not None:
        raise ValueError(f"Azure Quantum target {target!r} cannot run this circuit: {reason}.")


def prepare_circuit(circuit: Circuit, target: str) -> Any:
    """Return the Qiskit circuit to pass to the Azure backend's ``run``.

    Every barrier is removed: a barrier never changes the counts, and
    azure-quantum hands the circuit to qdk without its barrier-removal pass,
    so a barrier reaches the QIR and Rigetti's adaptor refuses it. Every
    qubit ``i`` is then measured into bit ``i`` of one register named
    ``meas``, explicitly rather than with ``measure_all()``, which inserts a
    barrier first. Identity wires are measured too, so every key has
    ``circuit.num_qubits`` characters.

    Raises:
        ValueError: If the circuit has no qubits, so nothing can be measured.
    """
    width = circuit.num_qubits
    if width == 0:
        raise ValueError(
            f"Azure Quantum target {target!r}: the circuit has no qubits, so there "
            "is nothing to measure and no program to submit."
        )

    from qiskit import ClassicalRegister, QuantumCircuit

    exported = circuit.to_qiskit()
    meas = ClassicalRegister(width, MEASUREMENT_REGISTER)
    prepared = QuantumCircuit(
        *exported.qregs, meas, name=exported.name, global_phase=exported.global_phase,
    )
    for instruction in exported.data:
        if instruction.operation.name != "barrier":
            prepared.append(instruction.operation, instruction.qubits, instruction.clbits)
    for index, qubit in enumerate(exported.qubits):
        prepared.measure(qubit, meas[index])
    return prepared


def counts_from_result(result: Any, target: str) -> dict[str, int]:
    """Return counts keyed with qubit 0 leftmost from a provider job's Result.

    The provider's counts follow Qiskit's order (bit 0 rightmost), so each
    key is reversed; register separators are dropped first.

    Raises:
        RuntimeError: If the job did not succeed, naming the target.
    """
    if not result.success:
        error_msg = "Job failed"
        if hasattr(result, "status"):
            error_msg = f"Job failed with status: {result.status}"
        if hasattr(result, "results") and result.results:
            exp_result = result.results[0]
            if hasattr(exp_result, "status"):
                error_msg += f", experiment status: {exp_result.status}"
            if hasattr(exp_result, "header") and hasattr(exp_result.header, "status_msg"):
                error_msg += f", message: {exp_result.header.status_msg}"
        raise RuntimeError(f"{error_msg} (Azure Quantum target {target!r})")

    raw_counts = dict(result.get_counts())
    return {key.replace(" ", "")[::-1]: int(value) for key, value in raw_counts.items()}
