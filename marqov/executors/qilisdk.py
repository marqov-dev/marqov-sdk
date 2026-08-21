"""Local executor using Qilimanjaro's qilisdk simulators.

Runs circuits on qilisdk's own open-source simulator stack — QiliSim (their
C++ simulator, ships in the base `qilisdk` package) or QutipBackend (pure-
Python reference sim). No SpeQtrum account or network call required.

`qilisdk` is installed separately (``pip install qilisdk``), not via a
`marqov[...]` extra: its numpy floor (>=2.3 on macOS, >=2.4.1 elsewhere)
is incompatible with marqov's own numpy<2.4 core pin outside a narrow
macOS overlap window, which makes it unresolvable as a formal extra in
marqov's own dependency lock.

See https://github.com/qilimanjaro-tech/qilisdk.
"""

import time
from dataclasses import dataclass
from typing import Any, Literal

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult

# Extra pip-install hint appended to the ImportError for backends with their
# own optional dependency beyond the base `qilisdk` package.
_SIMULATOR_EXTRA = {"qutip": ' (qutip extra: pip install "qilisdk[qutip]")'}


@dataclass
class QiliSDKExecutorConfig:
    """Configuration for the qilisdk local-simulator executor.

    Attributes:
        simulator: Which qilisdk backend to run on — "qilisim" (their C++
            simulator, ships in the base `qilisdk` package, `pip install
            qilisdk`) or "qutip" (pure-Python reference sim, `pip install
            "qilisdk[qutip]"`).
    """

    simulator: Literal["qilisim", "qutip"] = "qilisim"


class QiliSDKExecutor(BaseExecutor):
    """Execute circuits on Qilimanjaro's qilisdk local simulators.

    Runs entirely locally against qilisdk's own simulator stack (QiliSim or
    QutipBackend) — no SpeQtrum account or cloud dependency required.

    Example:
        >>> executor = QiliSDKExecutor()
        >>> circuit = Circuit().h(0).cnot(0, 1)
        >>> result = await executor.execute(circuit, shots=1000)
        >>> print(result.counts)  # {"00": ~500, "11": ~500}

    For the pure-Python reference simulator instead of QiliSim:
        >>> executor = QiliSDKExecutor(QiliSDKExecutorConfig(simulator="qutip"))
    """

    def __init__(self, config: QiliSDKExecutorConfig | None = None) -> None:
        """Initialize the qilisdk executor.

        Args:
            config: Configuration options. Uses defaults (QiliSim) if not provided.

        Raises:
            ImportError: If qilisdk (or, for the qutip simulator, its qutip
                extra) is not installed.
        """
        self.config = config or QiliSDKExecutorConfig()
        if self.config.simulator not in ("qilisim", "qutip"):
            raise ValueError(
                f"Unknown qilisdk simulator '{self.config.simulator}'. "
                "Supported: 'qilisim', 'qutip'."
            )
        try:
            if self.config.simulator == "qutip":
                from qilisdk.backends import QutipBackend

                self._backend = QutipBackend()
            else:
                from qilisdk.backends import QiliSim

                self._backend = QiliSim()
        except ImportError as exc:
            extra = _SIMULATOR_EXTRA.get(self.config.simulator, "")
            raise ImportError(
                f"qilisdk is required for QiliSDKExecutor. "
                f"Install with: pip install qilisdk{extra}"
            ) from exc

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Run circuit on a qilisdk local simulator.

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional options (ignored today).

        Returns:
            ExecutionResult with measurement counts.
        """
        circuit = self._validate_circuit(circuit)

        from qilisdk.functionals import DigitalPropagation
        from qilisdk.readout import Readout

        start_time = time.perf_counter()

        qili_circuit = self._to_qilisdk_circuit(circuit)
        propagation = DigitalPropagation(circuit=qili_circuit)
        readout = Readout().with_sampling(nshots=shots)

        result = self._backend.execute(propagation, readout)
        counts = result.get_samples()

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return ExecutionResult(
            counts=counts,
            backend=f"qilisdk-{self.config.simulator}",
            execution_time_ms=execution_time_ms,
            shots=shots,
            raw_result=result,
            metadata={"simulator": self.config.simulator},
        )

    def _to_qilisdk_circuit(self, circuit: Circuit) -> Any:
        """Translate a Marqov Circuit into a qilisdk digital Circuit.

        Walks the same internal gate list `Circuit.to_dict()`/`from_dict()`
        consume (`circuit._qf._elements`) rather than adding a second export
        path. Marqov's canonical gate set maps 1:1 onto qilisdk's digital
        gate constructors.

        Args:
            circuit: The Marqov circuit to translate.

        Returns:
            An equivalent qilisdk.digital.Circuit.

        Raises:
            NotImplementedError: If the circuit contains a gate outside
                Marqov's canonical set (H/X/Y/Z/S/T/Rx/Ry/Rz/CNOT/CZ/SWAP).
        """
        from qilisdk.digital import CNOT, CZ, RX, RY, RZ, SWAP, H, S, T, X, Y, Z
        from qilisdk.digital import Circuit as QiliCircuit

        qili_circuit = QiliCircuit(circuit.num_qubits)

        for op in circuit._qf._elements:
            name = op.name
            qubits = list(op.qubits)
            params = list(op.params) if hasattr(op, "params") and op.params else []

            if name == "H":
                qili_circuit.add(H(qubits[0]))
            elif name == "X":
                qili_circuit.add(X(qubits[0]))
            elif name == "Y":
                qili_circuit.add(Y(qubits[0]))
            elif name == "Z":
                qili_circuit.add(Z(qubits[0]))
            elif name == "S":
                qili_circuit.add(S(qubits[0]))
            elif name == "T":
                qili_circuit.add(T(qubits[0]))
            elif name == "Rx":
                qili_circuit.add(RX(qubits[0], theta=float(params[0])))
            elif name == "Ry":
                qili_circuit.add(RY(qubits[0], theta=float(params[0])))
            elif name == "Rz":
                qili_circuit.add(RZ(qubits[0], phi=float(params[0])))
            elif name == "CNot":
                qili_circuit.add(CNOT(qubits[0], qubits[1]))
            elif name == "CZ":
                qili_circuit.add(CZ(qubits[0], qubits[1]))
            elif name == "Swap":
                qili_circuit.add(SWAP(qubits[0], qubits[1]))
            else:
                raise NotImplementedError(
                    f"QiliSDKExecutor: unsupported gate '{name}'. "
                    "Supported: H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP."
                )

        return qili_circuit
