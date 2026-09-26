"""MarqovDevice — unified interface for quantum backend execution."""

from __future__ import annotations

import asyncio
from typing import Any

from marqov.backends import is_azure, is_braket, is_ibm, is_simulator
from marqov.circuits import Circuit
from marqov._optional import require_braket


def _run_loop_safe(fn):
    """Run a blocking callable safely regardless of the caller's async context.

    Braket's ``AwsQuantumTask.result()`` drives its polling via
    ``asyncio.get_event_loop().run_until_complete(...)``, which raises
    "This event loop is already running" when called from within a running loop
    (e.g. an async experiment runner). If a loop is running in this thread, run
    ``fn`` in a worker thread that has its own event loop; otherwise call it
    directly. See marqov-sdk#67.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()  # no running loop — safe to call directly

    import concurrent.futures

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return fn()
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_worker).result()


def _qir_outcome_to_qiskit_bitstring(outcome: Any) -> str:
    """Join an Azure QIR outcome into a bitstring the way Qiskit sees it.

    Mirrors ``AzureQuantumJob._qir_to_qiskit_bitstring`` in the azure-quantum
    Qiskit provider: a display string is a Python literal, a tuple is one
    classical register per item, and a list is the bits of one register.
    Reproducing it here keeps MarqovDevice's Azure counts identical to the
    ones AzureQuantumExecutor gets from that provider.
    """
    import ast
    import re

    value = outcome
    if isinstance(value, str) and not re.match(r"[\d\s]+$", value):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # already a raw bitstring, e.g. '01'

    if isinstance(value, tuple):
        return " ".join(_qir_outcome_to_qiskit_bitstring(term) for term in value)
    if isinstance(value, list):
        return "".join(str(bit) for bit in value)
    return str(value)


def _qir_outcome_to_bitstring(outcome: Any) -> str:
    """Convert an Azure QIR outcome to the SDK's qubit-0-leftmost bitstring.

    The reversal is the same one AzureQuantumExecutor applies to the Qiskit
    provider's counts, so both paths agree on endianness.
    """
    return _qir_outcome_to_qiskit_bitstring(outcome).replace(" ", "")[::-1]


def _azure_results_payload(job: Any) -> dict[str, Any]:
    """Download and decode the raw Azure results blob.

    ``Job.get_results()`` is deliberately not used: for both Microsoft output
    formats it returns normalized probabilities keyed by display strings such
    as '[0]', which is neither the declared return type of run() nor a
    bitstring. The raw blob still carries the shot counts.
    """
    import json

    payload = job.download_data(job.details.output_data_uri)
    if isinstance(payload, bytes):
        payload = payload.decode("utf8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(
            "Azure job results are not a JSON object; cannot extract counts "
            f"(got {type(payload).__name__})."
        )
    return payload


def _azure_counts_from_job(job: Any, shots: int) -> dict[str, int]:
    """Return measurement counts for a completed Azure Quantum job.

    Args:
        job: A completed azure-quantum Job.
        shots: The number of shots requested, used to allocate counts when the
            provider reports probabilities rather than counts.

    Returns:
        Mapping of bitstring (qubit 0 leftmost) to integer count.

    Raises:
        ValueError: If the job's output data format is one this path cannot
            turn into counts.
    """
    output_format = getattr(job.details, "output_data_format", None)
    payload = _azure_results_payload(job)

    if output_format == "microsoft.quantum-results.v2":
        results = payload.get("Results")
        if not results:
            raise ValueError(
                "Azure job results are missing the 'Results' array required "
                "by the microsoft.quantum-results.v2 output format."
            )
        histogram = results[0].get("Histogram")
        if histogram is None:
            raise ValueError(
                "Azure job results are missing the 'Histogram' array required "
                "by the microsoft.quantum-results.v2 output format."
            )
        # The v2 histogram carries the raw per-outcome Count, which
        # get_results() divides away into a probability.
        counts: dict[str, int] = {}
        for entry in histogram:
            bitstring = _qir_outcome_to_bitstring(entry["Display"])
            counts[bitstring] = counts.get(bitstring, 0) + int(entry["Count"])
        return counts

    if output_format == "microsoft.quantum-results.v1":
        histogram = payload.get("Histogram")
        if histogram is None:
            raise ValueError(
                "Azure job results are missing the 'Histogram' array required "
                "by the microsoft.quantum-results.v1 output format."
            )
        if len(histogram) % 2 != 0:
            raise ValueError(
                "Azure 'Histogram' array is malformed: an even number of "
                "items (display, probability) is expected."
            )
        # v1 reports probabilities only, so allocate the requested shots with
        # the largest-remainder method rather than rounding each bin.
        probabilities: dict[str, float] = {}
        for i in range(0, len(histogram), 2):
            bitstring = _qir_outcome_to_bitstring(histogram[i])
            probabilities[bitstring] = probabilities.get(bitstring, 0.0) + float(
                histogram[i + 1]
            )
        from marqov.executors._counts import allocate_counts

        return allocate_counts(probabilities, shots)

    raise ValueError(
        f"Azure output data format '{output_format}' is not supported by "
        "MarqovDevice.run: counts can only be extracted from the "
        "microsoft.quantum-results.v1 and microsoft.quantum-results.v2 "
        "formats."
    )


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
            require_braket()
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
        # Local/marqov-sim always run on the Braket LocalSimulator (see
        # _get_provider_device and run()), regardless of any stray
        # ibm_token/azure_subscription_id left in params — this must be
        # checked before is_ibm/is_azure or the two other dispatch sites
        # disagree with this one, feeding a Qiskit circuit to Braket's
        # simulator.
        if self._backend in ("local", "marqov-sim"):
            return marqov_circuit.to_braket()

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
            **kwargs: Backend-specific options, honored only on the cloud
                      AWS Braket path (`is_braket(self._params)` below) — the
                      local/`marqov-sim` path (a Braket `LocalSimulator`) does
                      not read `**kwargs` at all and silently ignores it:
                      - disable_qubit_rewiring (bool): prevent qubit remapping.
                      - verbatim (bool): submit under a verbatim box so the
                        compiler runs the gates exactly as given (required for
                        randomized benchmarking on Rigetti, or the compiler
                        optimizes the sequence away). Requires native gates only
                        (1Q: Rx/Rz, 2Q: CZ/XY); raises ValueError otherwise.

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
            import qiskit_ibm_runtime
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )

            from marqov.executors._counts import extract_sampler_counts

            optimization_level = self._params.get("ibm_optimization_level", 1)
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level,
                backend=device,
            )
            transpiled = pm.run(native_circuit)

            # Attribute access, not "from ... import SamplerV2": it keeps the
            # provider SDK boundary patchable for the tests that drive this
            # branch without stubbing out run()'s own internals.
            sampler = qiskit_ibm_runtime.SamplerV2(mode=device)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()

            # One extraction, shared with IBMExecutor: it resolves the
            # classical register by capability and reverses Qiskit's
            # little-endian bitstrings into the SDK's qubit-0-leftmost
            # convention. See marqov-sdk#161.
            return extract_sampler_counts(result)

        elif is_azure(self._params):
            job = device.submit(native_circuit, shots=shots)
            job.wait_until_completed()
            return _azure_counts_from_job(job, shots)

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
            # Wrap in a verbatim box for devices that require it (e.g. Rigetti
            # QPUs), mirroring BraketExecutor. Without it, the compiler folds
            # Clifford-plus-inverse sequences to identity and survival ≈ 1.0 at
            # every length. The circuit must already use only native gates —
            # e.g. clifford_to_circuit_native() / SRBConfig.use_native_gates=True.
            # The allowed set is Rigetti-specific; make this a device-aware
            # lookup when IQM or other verbatim providers are added.
            if kwargs.get("verbatim"):
                from braket.circuits import Circuit as BraketCircuit

                # Measure is deliberately NOT allowed — see BraketExecutor.execute:
                # Braket refuses to box a measured subcircuit and adds measurement
                # implicitly via `shots`.
                _RIGETTI_VERBATIM_ALLOWED = {"rx", "rz", "cz", "xy"}
                non_native = [
                    instr.operator.name
                    for instr in native_circuit.instructions
                    if instr.operator.name.lower() not in _RIGETTI_VERBATIM_ALLOWED
                ]
                if non_native:
                    raise ValueError(
                        f"verbatim=True requires Rigetti native gates only "
                        f"(1Q: Rx/Rz, 2Q: CZ/XY; no explicit Measure — Braket cannot box a "
                        f"measured subcircuit and adds measurement implicitly via shots). "
                        f"Found non-native gates: {sorted(set(non_native))}. "
                        f"Use clifford_to_circuit_native() or set "
                        f"SRBConfig.use_native_gates=True."
                    )
                native_circuit = BraketCircuit().add_verbatim_box(native_circuit)

            try:
                disable_rewiring = kwargs.get("disable_qubit_rewiring", False)

                def _submit_and_wait():
                    task = device.run(
                        native_circuit, s3_folder, shots=shots,
                        disable_qubit_rewiring=disable_rewiring,
                    )
                    return task.result()

                # Braket's result() polls via run_until_complete; run it
                # loop-safe so this works from an async runner too (marqov-sdk#67).
                result = _run_loop_safe(_submit_and_wait)
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
