"""Repeat the existing IBM basis-state qualification in isolated Linux Python 3.12."""
import asyncio
import json
from importlib.metadata import version
from qiskit_aer import AerSimulator
from marqov.circuits import Circuit
from marqov.executors import IBMExecutor, IBMExecutorConfig

async def main():
    results = []
    for qubit, key in [(0, "10"), (1, "01")]:
        executor = IBMExecutor(IBMExecutorConfig(backend_name="aer_simulator"))
        # Deliberate offline backend, never a remote error fallback.
        executor._backend = AerSimulator(seed_simulator=7)
        result = await executor.execute(Circuit().x(qubit).cz(0, 1), shots=32)
        assert result.counts == {key: 32}
        assert result.metadata["access_path"] == "local"
        assert result.metadata["engine"] == "SamplerV2"
        assert result.metadata["job_id"]
        assert len(result.metadata["reproducibility"]["counts_sha256"]) == 64
        results.append({"counts": result.counts, "metadata": result.metadata})
    print(json.dumps({"scope": "offline Linux SDK execution, not hosted or IBM cloud", "packages": {p: version(p) for p in ["marqov", "qiskit", "qiskit-ibm-runtime", "qiskit-aer"]}, "results": results}, indent=2))

asyncio.run(main())
