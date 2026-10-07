"""Run a small parameterised circuit on PennyLane Lightning CPU engines.

Requires ``pip install "marqov[pennylane]"`` (lightning.qubit) and, optionally,
``pip install "marqov[lightning-kokkos]"`` (lightning.kokkos). Runs locally; no
account needed. From a source checkout:

    python examples/lightning_cpu_local.py

For each available CPU engine it prints the counts and the reproducibility
record, then replays the same seed on a fresh device to show within-device
replay. Samples are not expected to match across engines.
"""

from __future__ import annotations

import asyncio
import json

from marqov.circuits import Circuit
from marqov.executors import ExecutorFactory
from marqov.executors.lightning import probe_lightning_device

SEED = 2026
SHOTS = 1000


def parameterised_circuit(theta: float, phi: float) -> Circuit:
    """Three-qubit ansatz: two parameters, entangled, many outcomes."""
    return (
        Circuit()
        .ry(theta, 0).ry(phi, 1).rx(theta / 2, 2)
        .cnot(0, 1).cnot(1, 2)
        .rz(phi, 2).h(0)
    )


async def run_engine(slug: str, device: str, circuit: Circuit) -> None:
    probe = probe_lightning_device(device)
    if not probe.available:
        print(f"\n## {device}: unavailable here ({probe.reason})")
        return

    executor = ExecutorFactory.create_executor(slug, {"provider": "PennyLane Lightning"})
    first = await executor.execute(circuit, shots=SHOTS, seed=SEED)
    replay = await executor.execute(circuit, shots=SHOTS, seed=SEED)
    other = await executor.execute(circuit, shots=SHOTS, seed=SEED + 1)

    record = first.metadata["reproducibility"]
    print(f"\n## {device}")
    print("counts:", json.dumps(first.counts, sort_keys=True))
    provenance = {k: first.metadata[k] for k in
                  ("vendor", "framework", "engine", "access_path", "compute_provider")}
    print("provenance:", json.dumps(provenance))
    print("reproducibility record:")
    print(json.dumps(record, indent=2, sort_keys=True))
    same = replay.metadata["reproducibility"]["samples_sha256"] == record["samples_sha256"]
    differs = other.metadata["reproducibility"]["samples_sha256"] != record["samples_sha256"]
    print(f"replay with seed {SEED} on a fresh device: identical samples = {same}")
    print(f"seed {SEED + 1}: different samples = {differs}")


async def main() -> None:
    circuit = parameterised_circuit(theta=0.9, phi=1.7)
    for slug, device in (("lightning-qubit", "lightning.qubit"),
                         ("lightning-kokkos", "lightning.kokkos")):
        await run_engine(slug, device, circuit)
    for device in ("lightning.gpu", "lightning.tensor"):
        probe = probe_lightning_device(device)
        state = "installed (not run by this CPU example)" if probe.available else probe.reason
        print(f"\n## {device}: {state}")


if __name__ == "__main__":
    asyncio.run(main())
