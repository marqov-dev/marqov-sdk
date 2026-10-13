"""Credential-free qualification of the installed SDK and a selected integration."""

import asyncio
import importlib.metadata as md
import json
from pathlib import Path
import sys
import sysconfig

import numpy as np
import marqov
from marqov import Circuit
from marqov.executors import LocalExecutor

profile = sys.argv[1]
assert Path(marqov.__file__).resolve().is_relative_to(Path(sysconfig.get_path("purelib")).resolve())
assert md.version("marqov-quantumflow") == "1.0.2"
try:
    md.version("quantumflow")
except md.PackageNotFoundError:
    pass
else:
    raise AssertionError("upstream QuantumFlow must not coexist with the fork")

circuit = Circuit().x(0).cz(0, 1)
expected = circuit.simulate().tensor
assert Circuit.from_dict(json.loads(json.dumps(circuit.to_dict()))).to_dict() == circuit.to_dict()


async def local():
    result = await LocalExecutor().execute(circuit, shots=32)
    assert result.counts == {"10": 32}, result.counts


asyncio.run(local())
if profile in {
    "braket",
    "qiskit",
    "cirq",
    "pennylane",
    "pyquil",
    "rigetti",
    "ionq",
    "ibm",
    "openqasm",
    "azure",
}:
    framework = {
        "rigetti": "pyquil",
        "ionq": "qiskit",
        "ibm": "qiskit",
        "openqasm": "qiskit",
        "azure": "qiskit",
    }.get(profile, profile)
    native = getattr(circuit, "to_" + framework)()
    restored = getattr(Circuit, "from_" + framework)(native)
    assert np.allclose(restored.simulate().tensor, expected)
    if profile == "openqasm":
        assert np.allclose(Circuit.from_openqasm(circuit.to_openqasm()).simulate().tensor, expected)
    if profile == "ibm":
        from qiskit_ibm_runtime import SamplerV2

        assert callable(SamplerV2)
    if profile == "azure":
        from azure.quantum.qiskit import AzureQuantumProvider
        from azure.quantum.cirq import AzureQuantumService

        assert callable(AzureQuantumProvider) and callable(AzureQuantumService)
elif profile in {"pytket", "quantinuum"}:
    assert np.allclose(circuit.to_pytket().get_unitary()[:, 0].reshape(2, 2), expected)
    if profile == "quantinuum":
        from pytket.extensions.quantinuum import QuantinuumBackend

        assert callable(QuantinuumBackend)
elif profile == "qutip":
    import qutip
    from marqov.qutip import record
    import contextlib
    import io

    stream = io.StringIO()
    result = qutip.mesolve(qutip.sigmaz(), qutip.basis(2, 0), [0, 0.1], e_ops=[qutip.sigmaz()])
    with contextlib.redirect_stdout(stream):
        record(result, observable_names=["z"])
    assert json.loads(stream.getvalue())["observables"]["z"] == [1.0, 1.0]
elif profile == "lightning-kokkos":
    from marqov.executors import LightningExecutor, LightningExecutorConfig

    async def kokkos():
        result = await LightningExecutor(
            LightningExecutorConfig(device="lightning.kokkos")
        ).execute(circuit, shots=32)
        assert result.counts == {"10": 32}, result.counts

    asyncio.run(kokkos())
elif profile == "cudaq":
    from marqov.executors import CudaqExecutor, CudaqExecutorConfig

    async def cudaq_check():
        result = await CudaqExecutor(CudaqExecutorConfig(target="qpp-cpu")).execute(
            circuit, shots=32
        )
        assert result.counts == {"10": 32}, result.counts

    asyncio.run(cudaq_check())
elif profile == "qilisdk":
    from marqov.executors import QiliSDKExecutor

    async def qili():
        result = await QiliSDKExecutor().execute(circuit, shots=32)
        assert result.counts == {"10": 32}, result.counts

    asyncio.run(qili())
print(
    json.dumps(
        {
            "profile": profile,
            "sdk": marqov.__version__,
            "import": marqov.__file__,
            "local_counts": {"10": 32},
        }
    )
)
