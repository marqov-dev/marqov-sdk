# Clifft executor (exact near-Clifford simulation)

`ClifftExecutor` runs Marqov circuits locally on
[Clifft](https://github.com/unitaryfoundation/clifft), Unitary Foundation's
exact simulator for Clifford circuits with a small number of non-Clifford
gates. It is opt-in: it is never the default executor, and nothing runs on it
unless you create a `ClifftExecutor`.

```bash
pip install "marqov[clifft]"   # also included in marqov[all]
```

The extra pins `clifft==0.14.0`, the release the executor is tested against.
Clifft's API and seeded sampling change between 0.x releases, so the executor
refuses to run on any other installed version (`ClifftUnavailableError`) rather
than guess. Wheels exist for macOS arm64 (macOS 15 or later), Linux
x86_64/aarch64 and Windows x86_64; on other platforms the extra installs nothing
and the executor reports itself offline.

```python
import asyncio
from marqov.circuits import Circuit
from marqov.executors import ClifftExecutor

async def main():
    circuit = Circuit().h(0)
    for q in range(49):
        circuit.cnot(q, q + 1)          # 50-qubit GHZ state (Clifford)
    circuit.h(0).t(0).h(0)              # one non-Clifford gate
    result = await ClifftExecutor().execute(circuit, shots=1000, seed=42)
    print(len(next(iter(result.counts))))                         # 50
    print(result.metadata["reproducibility"]["peak_active_width"])  # 1

asyncio.run(main())
```

## When to choose it

Clifft tracks Clifford gates symbolically and keeps a dense state only over the
qubits that non-Clifford gates (T, arbitrary-angle rotations, Toffoli) have made
"active". Its cost grows with the compiled *peak active width* (a dense state of
`2**peak_active_width` amplitudes), not with the circuit's qubit count.

Choose it for **wide circuits with a narrow non-Clifford part**: many qubits,
mostly Clifford gates, and a handful of T gates or rotations. Such circuits can
be far wider than a statevector simulator can hold in memory.

Do not choose it for:

- **Small circuits.** A statevector simulator (`LocalExecutor`,
  `LightningExecutor`) holds a small circuit's whole state cheaply, and
  Clifft's compile step makes it no faster, usually slower.
- **Circuits whose active width approaches the qubit count.** Each T gate or
  arbitrary rotation on a new qubit can widen the active state. The number of
  non-Clifford gates matters less than how wide they make it: Clifft 0.14
  shares one evolving state across shots where it can, so many T gates on a
  narrow active state stay cheap. Once the peak active width is close to
  `num_qubits`, Clifft does the same dense work as a statevector simulator plus
  its own bookkeeping, and is usually slower, increasingly so as the width
  grows.

Every result records `peak_active_width`, so the choice can be checked on the
actual circuit: a value close to `num_qubits` means a statevector simulator is
the better fit.

## Scope

- Gates: H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP, CCX (Toffoli), CSWAP
  (Fredkin) and the identity, with bound (numeric) angles. Rotations are sent
  with the angle in units of pi (Clifft's convention). Clifft has no
  controlled-SWAP, so `cswap(c, a, b)` is sent as `CX b a; CCX c a b; CX b a`.
  Each gate keeps the same matrix up to global phase, which sampling cannot
  observe. Any other gate (for example one imported from Qiskit or Braket, such
  as ISwap) raises `NotImplementedError` naming the gate; symbolic angles raise
  `ValueError`.
- Measurement: every qubit below `Circuit.num_qubits` is measured in the
  computational basis at the end, identity wires included. Count keys list the
  qubits in ascending order, lowest index leftmost, the same keys
  `LocalExecutor` returns (`Circuit(num_qubits=3).x(0).x(1)` gives `"110"`).
- Options: `seed` is the only keyword option; any other raises `TypeError`.
  Sampling always uses one thread.

### Noise, detectors and error correction

A `Circuit` carries gates only: no noise channels, no mid-circuit measurements,
no detectors or observables. So `ClifftExecutor` runs noiseless circuits with
terminal measurement only, and none of Clifft's error-correction features
(noisy sampling, detector and observable records, post-selection) are reachable
through it. For error-correction work, call Clifft directly inside a `@task`
with a Stim-format program, and install `clifft` in the task's environment
yourself.

## Seeds and replay

Every run compiles a fresh program and passes the seed to `clifft.sample`, so
no random state carries over between calls. The seed comes from the call
(`seed=`), else `ClifftExecutorConfig.seed`, else a generated 32-bit seed; the
record names which (`seed_source`). Seeds are integers in `[0, 2**63-1]`.

Re-running the same circuit with the recorded seed and shot count, on the same
Clifft version and platform, reproduces the samples (`samples_sha256`). Clifft
documents that rows for one seed can differ between execution configurations
(scalar, packed or shared execution, batch capacity, thread count). The
executor fixes what it requests (`threads=1`, `batch_size="auto"`,
`sharing="auto"`), and records what Clifft then selected under `execution`.
Samples are not expected to match across Clifft releases.

## The record

`result.metadata` carries `vendor: "Unitary Foundation"`, `engine: "clifft"`,
`access_path: "local"`, `seed`, and `reproducibility`, a JSON-serialisable dict:

| Field | Content |
|---|---|
| `record_version` | Record schema version (1) |
| `clifft_version`, `numpy_version`, `marqov_version` | Installed versions |
| `clifft_runtime_isa` | The CPU kernel Clifft selected at runtime (`clifft.runtime_isa()`) |
| `platform`, `machine`, `python` | Host platform and Python version |
| `n_qubits`, `bit_order`, `measurements`, `shots` | Measured qubits and how bitstrings map onto them |
| `seed`, `seed_source`, `rng_policy` | Seed used and its origin; `"fresh program per run; seed passed to clifft.sample"` |
| `threads`, `batch_size`, `sharing` | Requested sampling configuration, always `1`, `"auto"` and `"auto"` |
| `execution` | What Clifft selected (`result.execution`): `executor` (`"scalar"`, `"packed"` or `"shared"`), `batch_size`, `sharing`, `shot_workers`, `intra_shot_workers`, `noise_preparation` |
| `peak_active_width` | The compiled program's peak active width |
| `program_text` | The exact Clifft program that ran, measurement included |
| `circuit_sha256`, `program_sha256` | Hash of `Circuit.to_dict()`; hash of `program_text` |
| `samples_sha256`, `counts_sha256` | Hash of the raw `(shots, n_qubits)` samples; hash of the counts |
| `compile_ms` | Time spent in `clifft.compile` |

`raw_result` is the `(shots, n_qubits)` uint8 sample array.

Validation: real Clifft 0.14.0 runs on CPython 3.12, macOS arm64 (see
`tests/test_clifft_executor.py`). Each mapped gate's state is compared with
QuantumFlow's up to global phase, sampled distributions are checked against
QuantumFlow's exact probabilities within 5 binomial standard deviations
(including a many-T circuit that Clifft runs with shared execution), seeded
runs are replayed, and the CSWAP decomposition is checked on all eight basis
inputs.
