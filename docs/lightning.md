# PennyLane Lightning (CPU) executor

`LightningExecutor` runs Marqov circuits locally on Xanadu's PennyLane
Lightning simulators and attaches a reproducibility record to every result.

```bash
pip install "marqov[pennylane]"          # lightning.qubit
pip install "marqov[lightning-kokkos]"   # adds lightning.kokkos (Linux x86_64/aarch64, macOS arm64)
```

```python
from marqov import Circuit
from marqov.executors import ExecutorFactory

executor = ExecutorFactory.create_executor("lightning-qubit", {"provider": "PennyLane Lightning"})
result = await executor.execute(Circuit().ry(0.9, 0).cnot(0, 1), shots=1000, seed=42)
result.counts                                   # e.g. {"00": ..., "11": ...}
result.metadata["reproducibility"]["seed"]      # 42
```

A copy-and-run demo, including seed replay, is `examples/lightning_cpu_local.py`.

## Scope

- Devices: `lightning.qubit` and `lightning.kokkos`. Asking for `lightning.gpu`
  or `lightning.tensor` constructs the device to check this environment: if
  that fails, `LightningDeviceUnavailableError` carries PennyLane's own error;
  if it succeeds, the run is refused with `NotImplementedError` because this
  executor has not been validated there. No other device or simulator is
  substituted. `probe_lightning_device(name)` runs the same check on its own.
- Each CPU device must pass a qualification probe once per process and precision before the
  first user circuit: an asymmetric basis state must produce the SDK's count
  key (`"10"` for X on qubit 0 of 2), and a Bell state must stay within 5 sigma
  of 50/50. The outcome is in the record under `qualification`.
- Circuits: the canonical gate set with bound (numeric) angles. Gates outside
  it, symbolic angles and empty circuits raise.
- Measurement: finite shots, computational-basis samples of every qubit the
  circuit uses. Count keys list those qubits in ascending order, lowest index
  leftmost, the same keys `LocalExecutor` returns (sparse indices included).
- Precision: `precision="double"` (complex128, default) or `"single"` (complex64).

The circuit is converted by `Circuit.to_pennylane()`, a direct gate-by-gate
export (no OpenQASM round trip) adapted from marqov-sdk#38. It returns a
`QuantumScript` with operations only; the executor adds the measurement and shots.

## Seeds and replay

Every run builds a fresh device and passes the seed at construction, so no
random state carries over between calls. The seed comes from the call
(`seed=`), else `LightningExecutorConfig.seed`, else a generated 32-bit seed;
the record names which (`seed_source`). Re-running with the recorded seed on
the same device, PennyLane/Lightning versions and platform reproduces the
samples (`samples_sha256`).

Samples are not expected to match across devices (`lightning.qubit` and
`lightning.kokkos` sample differently for the same seed) or across releases;
seed behaviour is version-specific.

## The record

`result.metadata` carries provenance as separate fields (`vendor: "Xanadu"`,
`framework: "PennyLane"`, `engine`, `access_path: "local"`, `compute_provider`,
taken from the config, default `"local"`) and `reproducibility`, a
JSON-serialisable dict:

| Field | Content |
|---|---|
| `record_version` | Record schema version (1) |
| `pennylane_version`, `plugin_version` | Installed versions (`pennylane-lightning==…`, `pennylane-lightning-kokkos==…`) |
| `device_requested`, `device_name`, `device_class` | What was asked for and what PennyLane constructed |
| `wheel_tag`, `binary`, `binary_sha256` | Plugin wheel tag; compiled module file and its SHA-256 |
| `backend_info` | Engine `compile_info`/`runtime_info`, Kokkos version and default execution space, `scipy-openblas32` version |
| `precision`, `n_wires`, `wire_order`, `bit_order` | State dtype; measured wires and how bitstrings map onto them |
| `measurements`, `shots` | Measurement definition and shots, read from the executed tape |
| `seed`, `seed_kind`, `seed_source`, `rng_policy`, `device_call_index` | Seed used and its origin; `"fresh device per run; seed applied at construction"`; always 0 |
| `diff_method` | Always `None` (sampling only) |
| `omp_threads_env`, `omp_threads_note` | `OMP_NUM_THREADS` as found; PennyLane does not read it, and it had no observed effect on the macOS arm64 0.45 wheels |
| `qualification` | Outcome of the per-process probe |
| `host` | OS/platform, machine, CPU, cores, Python, numpy, scipy |
| `circuit_sha256`, `input_sha256` | Hash of `Circuit.to_dict()`; hash of the executed operations, measurement and shots |
| `samples_sha256`, `counts_sha256` | Hash of the raw `(shots, n_wires)` samples; hash of the counts |

Field names follow the per-run device record used in the Marqov Xanadu lab
(E1) where the two overlap.

Validation: real Lightning runs on CPython 3.12, macOS arm64, PennyLane 0.45.1
with `pennylane-lightning` and `pennylane-lightning-kokkos` 0.45.0 (see
`tests/test_lightning_executor.py`). Linux Kokkos builds (which may use OpenMP
rather than the Serial execution space of the macOS wheel) have not been run.
