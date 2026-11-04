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
