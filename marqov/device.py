"""MarqovDevice — unified interface for quantum backend execution."""

from __future__ import annotations

from marqov.backends import is_azure, is_braket, is_ibm, is_simulator
from marqov.circuits import Circuit


class MarqovDevice:
    """Wraps a quantum backend and provides a uniform run() interface.

    Scripts receive a MarqovDevice from get_device() and call device.run(circuit, shots)
    without needing vendor-specific branching. Accepts any supported circuit type
    (Braket, Qiskit, Cirq, PennyLane, QASM string, or Marqov Circuit) and
    automatically converts to the target backend's native format.
    """

    def __init__(self, backend: str, params: dict) -> None:
        self._backend = backend
        self._params = params
        self._provider_device = None

    @property
    def backend_name(self) -> str:
        """Return the backend identifier (e.g. 'sv1', 'ionq-aria-1')."""
        return self._backend

    @property
    def is_simulator(self) -> bool:
        """Return True if this device targets a simulator backend."""
        return is_simulator(self._backend)

    def _get_provider_device(self):
        """Lazy-load and return the underlying provider device."""
        if self._provider_device is not None:
            return self._provider_device

        if self._backend in ("local", "marqov-sim"):
            from braket.devices import LocalSimulator

            self._provider_device = LocalSimulator()

        elif is_ibm(self._params):
            from qiskit_ibm_runtime import QiskitRuntimeService

            kwargs = {
                "channel": self._params.get("ibm_channel", "ibm_quantum"),
                "instance": self._params.get("ibm_instance", "ibm-q/open/main"),
            }
            if self._params.get("ibm_token"):
                kwargs["token"] = self._params["ibm_token"]

            service = QiskitRuntimeService(**kwargs)
            self._provider_device = service.backend(self._backend)

        elif is_azure(self._params):
            from azure.quantum import Workspace

            workspace = Workspace(
                subscription_id=self._params["azure_subscription_id"],
                resource_group=self._params["azure_resource_group"],
                name=self._params["azure_workspace_name"],
                location=self._params.get("azure_location", "eastus"),
            )
            targets = workspace.get_targets(name=self._backend)
            self._provider_device = targets

        elif is_braket(self._params):
            from braket.aws import AwsDevice
            from braket.aws.aws_session import AwsSession
            import boto3

            device_arn = self._params["device_arn"]

            # Extract region from ARN for explicit session
            # ARN format: arn:aws:braket:<region>::device/...
            arn_parts = device_arn.split(":")
            region = arn_parts[3] if len(arn_parts) > 3 and arn_parts[3] else "us-east-1"

            session = AwsSession(boto3.Session(region_name=region))
            self._provider_device = AwsDevice(device_arn, aws_session=session)

        else:
            raise ValueError(
                f"Cannot determine provider for backend '{self._backend}'. "
                f"Params must include one of: device_arn (AWS Braket), "
                f"ibm_token/ibm_channel (IBM Quantum), azure_subscription_id (Azure Quantum)."
            )

        return self._provider_device

    def _normalize_circuit(self, circuit) -> Circuit:
        """Convert any supported circuit type to a marqov.Circuit.

        Supports: marqov.Circuit, str (QASM), Braket Circuit, Qiskit
        QuantumCircuit, Cirq Circuit, PennyLane QuantumScript.

        Raises:
            TypeError: If the circuit type is not supported.
        """
        if isinstance(circuit, Circuit):
            return circuit

        if isinstance(circuit, str):
            return Circuit.from_openqasm(circuit)

        # Braket Circuit
        try:
            from braket.circuits import Circuit as BraketCircuit

            if isinstance(circuit, BraketCircuit):
                return Circuit.from_braket(circuit)
        except ImportError:
            pass

        # Qiskit QuantumCircuit
        try:
            from qiskit import QuantumCircuit

            if isinstance(circuit, QuantumCircuit):
                return Circuit.from_qiskit(circuit)
        except ImportError:
            pass

        # Cirq Circuit
        try:
            import cirq

            if isinstance(circuit, cirq.Circuit):
                return Circuit.from_cirq(circuit)
        except ImportError:
            pass

        # PennyLane QuantumScript / QuantumTape
        try:
            import pennylane as qml

            if isinstance(circuit, qml.tape.QuantumScript):
                return Circuit.from_pennylane(circuit)
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported circuit type: {type(circuit).__name__}. "
            f"Supported types: marqov.Circuit, str (QASM), braket.circuits.Circuit, "
            f"qiskit.QuantumCircuit, cirq.Circuit, pennylane.tape.QuantumScript"
        )

    def _validate_circuit(self, marqov_circuit: Circuit) -> None:
        """Pre-flight check: verify circuit fits the target device."""
        # Check qubit count for QPU backends
        if not self.is_simulator and self._provider_device is not None:
            try:
                device_qubits = self._provider_device.properties.paradigm.qubitCount
                circuit_qubits = marqov_circuit.num_qubits
                if circuit_qubits > device_qubits:
                    raise ValueError(
                        f"Circuit requires {circuit_qubits} qubits but "
                        f"{self._backend} supports {device_qubits}. "
                        f"Reduce circuit size or use a simulator."
                    )
            except (AttributeError, TypeError):
                pass  # Not all devices expose qubit count this way

    def _to_backend_format(self, marqov_circuit: Circuit):
        """Convert a marqov.Circuit to the target backend's native format.

        - Braket backends (local, AWS): .to_braket() — auto-measures all qubits
        - IBM/Azure backends: .to_qiskit() + measure_all() if no measurements present
        """
        if is_ibm(self._params) or is_azure(self._params):
            qc = marqov_circuit.to_qiskit()
            if not qc.cregs:
                qc.measure_all()
            return qc

        # Braket format: local simulators and AWS Braket devices
        return marqov_circuit.to_braket()

    def run(self, circuit, shots: int = 1000, **kwargs) -> dict[str, int]:
        """Execute a circuit and return measurement counts.

        Accepts any supported circuit type — automatically normalizes to
        marqov.Circuit and converts to the target backend's native format.

        Args:
            circuit: Any supported circuit (Braket, Qiskit, Cirq, PennyLane,
                     QASM string, or marqov.Circuit).
            shots: Number of measurement shots.
            **kwargs: Backend-specific options. Braket backends accept
                      disable_qubit_rewiring (bool) to prevent qubit remapping.

        Returns:
            Dictionary mapping bitstring outcomes to their counts.
        """
        marqov_circuit = self._normalize_circuit(circuit)
        native_circuit = self._to_backend_format(marqov_circuit)
        device = self._get_provider_device()
        self._validate_circuit(marqov_circuit)

        if self._backend in ("local", "marqov-sim"):
            result = device.run(native_circuit, shots=shots).result()
            return dict(result.measurement_counts)

        elif is_ibm(self._params):
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
            from qiskit_ibm_runtime import SamplerV2

            optimization_level = self._params.get("ibm_optimization_level", 1)
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level,
                backend=device,
            )
            transpiled = pm.run(native_circuit)

            sampler = SamplerV2(mode=device)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()

            # Extract counts from SamplerV2 result
            pub_result = result[0]
            data_bin = pub_result.data
            creg_names = [
                attr for attr in dir(data_bin) if not attr.startswith("_")
            ]
            if creg_names:
                bit_array = getattr(data_bin, creg_names[0])
                return dict(bit_array.get_counts())
            return {}

        elif is_azure(self._params):
            job = device.submit(native_circuit, shots=shots)
            job.wait_until_completed()
            results = job.get_results()
            return dict(results)

        elif is_braket(self._params):
            s3_folder = self._params.get("s3_destination_folder")
            if not s3_folder:
                # Construct from separate bucket/prefix params (worker passes these)
                s3_bucket = self._params.get("s3_bucket")
                s3_prefix = self._params.get("s3_prefix")
                if s3_bucket and s3_prefix:
                    s3_folder = (s3_bucket, s3_prefix)
                else:
                    raise ValueError(
                        "s3_destination_folder or s3_bucket+s3_prefix required for AWS device execution"
                    )
            try:
                disable_rewiring = kwargs.get("disable_qubit_rewiring", False)
                task = device.run(
                    native_circuit, s3_folder, shots=shots,
                    disable_qubit_rewiring=disable_rewiring,
                )
                result = task.result()
                counts = dict(result.measurement_counts)
                if not counts:
                    # Some QPU backends (e.g. IonQ Forte-1) return measurement_probabilities
                    # instead of raw shot counts. Convert to synthetic counts using shots.
                    probs = getattr(result, 'measurement_probabilities', {}) or {}
                    if probs:
                        counts = {bs: round(float(prob) * shots) for bs, prob in dict(probs).items()}
                return counts
            except Exception as e:
                error_msg = str(e)
                if "DeviceRetiredException" in error_msg or "retired" in error_msg.lower():
                    raise RuntimeError(
                        f"Device '{self._backend}' has been retired by the provider. "
                        f"Choose a different backend. Original error: {error_msg}"
                    ) from e
                elif "DeviceOfflineException" in error_msg or "OFFLINE" in error_msg:
                    raise RuntimeError(
                        f"Device '{self._backend}' is currently offline. "
                        f"Try again later or choose a different backend. Original error: {error_msg}"
                    ) from e
                raise

        else:
            raise ValueError(
                f"Cannot determine provider for backend '{self._backend}'. "
                f"Params must include one of: device_arn (AWS Braket), "
                f"ibm_token/ibm_channel (IBM Quantum), azure_subscription_id (Azure Quantum)."
            )


def get_device(params: dict) -> MarqovDevice:
    """Factory function — create a MarqovDevice from execution parameters.

    Args:
        params: Dictionary containing at minimum a 'backend' key.

    Returns:
        A configured MarqovDevice instance.

    Raises:
        ValueError: If 'backend' is missing from params.
    """
    backend = params.get("backend")
    if not backend:
        raise ValueError("'backend' is required in params")
    return MarqovDevice(backend, params)
