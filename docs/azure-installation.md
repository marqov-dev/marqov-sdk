# Azure adapter installation

Install `marqov[azure]` for both Azure Qiskit and Cirq adapters, or `marqov[all]`
for the broader backend bundle. Both select
`azure-quantum[qiskit,cirq]>=3.12.0,<3.13.0`, so the vendor supplies its adapter
requirements, including Q# support and its supported Cirq range. Framework
packages alone do not supply these dependencies. A plain `pip install marqov`
does not install `azure-quantum`; Azure paths then raise `ImportError` naming
`marqov[azure]`.

The range stops below 3.13.0 because that release requires
`cryptography<=46.0.3`, which would hold back cryptography in the whole
environment. 3.12.0 has no cryptography dependency. The 3.12 floor identifies
the vendor release tested. Widen the ceiling once a later release is confirmed
free of the cap, after a compatibility check.

Credential-free checks can run outside the repository after installation:

```bash
python -c 'from azure.quantum.qiskit import AzureQuantumProvider; from azure.quantum.cirq import AzureQuantumService; print("Azure adapters imported")'
```

The clean Python 3.12 wheel installation was checked with Azure Quantum 3.12.0,
Qiskit 2.5.2, Q# (`qsharp`) 1.31.0, QDK 1.33.1, Cirq Core 1.6.1 and
`marqov-quantumflow` 1.0.2. The Azure-only and all-extras wheel environments
resolve; the all-extras environment also undergoes installed-package tests.
These versions record verification evidence rather than an exact user lock.
Use the repository's updated `uv.lock` for its frozen development environment.

`tests/test_azure_dependencies.py` imports both real vendor adapters without
skipping missing dependencies, and checks the real native Qiskit job's timeout
and workspace cancellation delegation using injected offline objects. Normal
CI and the installed-wheel release gate collect these tests through `[all]`.
No Azure credentials, workspace requests or provider jobs are required.

This resolves the installation gap in marqov-sdk#223. Import and offline tests
do not qualify connected execution, billing or deployed runtime compatibility.
Those require the platform's separate adoption and qualification process.

## How circuits are submitted

`AzureQuantumExecutor` with `framework="qiskit"` (the default) and the Azure
branch of `MarqovDevice` submit the same way, through azure-quantum's Qiskit
provider: `AzureQuantumProvider(workspace).get_backend(target).run(circuit,
shots=shots)`. `MarqovDevice` uses its backend name as the Azure target name.
In azure-quantum 3.12.0, `get_backend` returns the QIR backend for IonQ,
Quantinuum and Rigetti targets, so the provider and qdk translate the circuit
to QIR. `framework="cirq"` submits through azure-quantum's Cirq service
instead and is not changed by this section.

The circuit passed to `run` (`prepare_circuit` in
`marqov/executors/_azure_qiskit.py`):

- has every barrier removed. A barrier never changes the counts, and
  azure-quantum 3.12.0 calls qdk with `skip_transpilation=True`, while qdk
  removes barriers only during transpilation. A barrier would reach the QIR,
  which Rigetti's adaptor refuses (`QATTransformationFailed`).
- measures every qubit `i` into bit `i` of one classical register named
  `meas`, explicitly. It never uses Qiskit's `measure_all()`, which inserts a
  barrier first. Qubits no gate touches are measured too, so every counts key
  has `circuit.num_qubits` characters, with qubit 0 leftmost, as elsewhere in
  the SDK.

Refused before Azure is contacted, with a `ValueError` naming the target:

- PASQAL targets (`pasqal.*`), on both frameworks: they take pulse-level
  Pulser programs, and azure-quantum has no Qiskit or Cirq backend for them.
- A circuit with no qubits, on the Qiskit route: there is nothing to measure.

A job that does not succeed raises `RuntimeError` naming the target.

`tests/test_azure_provider_route.py` checks this offline: it builds the real
provider over an offline workspace and replaces only the upload, then
inspects the circuit given to `run`, the QIR it becomes (no barrier, one
`meas` register measured in qubit order) and the counts keys. As above,
offline tests do not qualify a connected run on any target.
