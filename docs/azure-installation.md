# Azure adapter installation

Install `marqov[azure]` for both Azure Qiskit and Cirq adapters, or `marqov[all]`
for the broader backend bundle. Both select
`azure-quantum[qiskit,cirq]>=3.13.0,<4.0.0`, so the vendor supplies its adapter
requirements, including Q# support and its supported Cirq range. Framework
packages alone do not supply these dependencies. Core Azure dependencies remain
available without enabling the adapter extras.

The 3.13 floor identifies the vendor release tested for this fix. It does not
mean earlier Azure releases could never work with their own correct extras.
The next major release requires a separate compatibility check.

Credential-free checks can run outside the repository after installation:

```bash
python -c 'from azure.quantum.qiskit import AzureQuantumProvider; from azure.quantum.cirq import AzureQuantumService; print("Azure adapters imported")'
```

The clean Python 3.12 wheel installation was checked with Azure Quantum 3.13.0,
Qiskit 2.5.2, Q# (`qsharp`) 1.31.0, QDK 1.33.0, Cirq Core 1.6.1 and
`marqov-quantumflow` 1.0.1. The Azure-only and all-extras wheel environments
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
