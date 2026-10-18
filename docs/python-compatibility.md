# Python compatibility: core SDK and provider integrations

Qualification snapshot: **October 17, 2026** (Linux) and **October 13, 2026**
(macOS). Python 3.13 checks passed for **17 profiles on Linux x86_64** and
**16 on macOS arm64**. Rigetti, pyQuil and `[all]` failed installation on both.
CUDA-Q native execution is Linux-only.

## Which SDK was tested?

These are **development/release-candidate results**, not qualification of the
published PyPI package. Linux used a wheel built by CI from `main` at
`b289b8fed3f93547d3211b832f7f6e4eae72cb87` (after the 0.9.1 release and the
October review fixes); macOS used the hash-verified 0.9.1 TestPyPI release
candidate from `b5e718be7ec217e9d77cc268378f301949297f93`. The Linux commit is
on `main` and can be checked out to reproduce the evidence; the macOS commit
was the head of the squash-merged release-candidate branch and is no longer on
a branch, so its reproducible pointer is the TestPyPI 0.9.1 wheel and its
SHA-256 in the record. This qualification changes no SDK runtime or dependency
constraints. Both wheels require exactly `marqov-quantumflow==1.0.2`.

The [machine-readable record](qualification/python313.json) identifies each
wheel SHA-256, tested source commit, Python patch version and exact installed
package versions. Linux used CPython 3.13.16; macOS used CPython 3.13.13.
OS and Python patch versions differ, so these are measured combinations rather
than an experiment isolating operating-system effects.

## Python 3.13 matrix

“Checks pass” means installation, dependency consistency, site-packages imports
and the stated credential-free checks passed. It does **not** mean live QPU,
GPU or deployed-runtime qualification. Some rows are only conversion/import
probes; others include real local execution and selected regression tests.

| Integration | Linux x86_64 | macOS arm64 | Linux check scope |
|---|---|---|---|
| Core / LocalExecutor | Checks pass | Checks pass | 73 workflow/sampling tests + local execution |
| Braket | Checks pass | Checks pass | 149 tests + circuit conversion |
| Qiskit | Checks pass | Checks pass | Circuit conversion probe |
| Cirq | Checks pass | Checks pass | Circuit conversion probe |
| PennyLane / Lightning | Checks pass | Checks pass | 60 passed, 3 Kokkos skips + Lightning tests |
| Lightning Kokkos | Checks pass | Checks pass | 63 tests + real Kokkos execution |
| CUDA-Q | Checks pass | Native unavailable | 41 tests + real CPU execution on Linux |
| QuTiP | Checks pass | Checks pass | 13 solver/record tests |
| pytket | Checks pass | Checks pass | Unitary export probe |
| Quantinuum | Checks pass | Checks pass | Unitary export + vendor import |
| OpenQASM | Checks pass | Checks pass | QASM round-trip probe |
| IBM Runtime | Checks pass | Checks pass | 28 channel tests + vendor import/conversion |
| Azure Quantum | Checks pass | Checks pass | 47 adapter tests + vendor imports/conversion |
| IonQ | Checks pass | Checks pass | 64 adapter tests + conversion |
| QiliSDK / SpeQtrum | Checks pass | Checks pass | 93 tests + local execution |
| Alice & Bob | Checks pass | Checks pass | Local vendor model + adapter tests |
| QB remote | Checks pass | Checks pass | SDK payload construction probe |
| pyQuil | Install fails | Install fails | Installation failed; tests not run |
| Rigetti | Install fails | Install fails | Installation failed; tests not run |
| [all] bundle | Install fails | Install fails | Installation failed; tests not run |

All successful probes include real asymmetric LocalExecutor counts in Marqov's
qubit-0-leftmost convention. Linux framework round-trip probes also compare a
parameterized complex statevector; macOS's initial round-trip probes use the
asymmetric basis circuit. pytket/Quantinuum exercise export and vendor imports,
not remote execution. IBM's selected tests cover channel behavior; this row does
not certify the remote Sampler service. Azure and IonQ tests include injected
vendor/transport objects. QB remote constructs its native payload without sending
it. Alice & Bob runs the vendor's local model, not physical hardware.

The PennyLane environment does not install Kokkos: its three existing Kokkos
checks skip. The dedicated Kokkos profile runs all 62 checks without skips.
macOS CUDA-Q runs 37 mock/conversion tests and skips its real-native test; the
separate native probe fails because the CUDA-Q package is absent. It is therefore
**not** classified as passing native qualification.

Qristal and CUNQA native environments are **not qualified on Python 3.13** by
this work. They require separate vendor builds/runtime environments. Windows,
Python 3.14 and free-threaded Python builds were not tested. Existing Python 3.12
full-suite and installed-wheel release gates remain required; see
[the release procedure](../RELEASING.md).

## Known restrictions and installation choices

The measured failing chain is `pyquil==4.22.0` → `quil==0.18.0` → PyO3 0.20.3.
Its build rejects Python 3.13 as newer than its maximum supported Python 3.12.
Those installation failures prevent imports and tests; they are not failing SDK
assertions. Use Python 3.12 for `[pyquil]`, `[rigetti]` and `[all]`.

Use selected extras on Python 3.13 rather than `[all]`. The bundle includes pyQuil
and does not silently drop that integration. It also does not contain every
native provider dependency: CUDA-Q, Kokkos, QiliSDK and Alice & Bob have separate
installation paths. See [Alice & Bob](alice-bob.md) and [QiliSDK](qilisdk.md) for
vendor pins and isolation requirements; QiliSDK was tested at 0.3.0 with
`qutip-qip>=0.4.0`, and Alice & Bob with the documented 1.2.0/Qiskit 1.x environment.
Fresh resolutions were tested, not every dependency version allowed by our floors.

## Reproduce and maintain the evidence

From a source checkout, build a wheel and run:

```bash
uv build
uv venv --python 3.13 .venv-qualification
QUALIFICATION_ROOT=$(mktemp -d)
.venv-qualification/bin/python tools/qualify_python.py \
  --wheel /absolute/path/to/marqov-X.Y.Z-py3-none-any.whl \
  --output "$QUALIFICATION_ROOT/results" \
  --source-commit "$(git rev-parse HEAD)" \
  --profiles core braket qiskit cirq azure
```

The runner creates one fresh environment per profile, installs the wheel rather
than the checkout, and adds only minimal test tools (not `[dev]`, which would
install other integrations). It checks distribution ownership and saves package
versions, probe output, pytest totals and failure logs, then removes each test
environment. A failed install or probe exits nonzero; it is never called a pass.

[Linux qualification run](https://github.com/marqov-dev/marqov-sdk/actions/runs/39059594315)
(the `CI` run for `main` at `b289b8f`) contains downloadable
`python313-<profile>` evidence artifacts; every push to `main` produces a
fresh set. The 17 qualified Linux profiles now feed the required `CI Success`
gate. Rigetti, pyQuil and `[all]` retain separate advisory jobs with visible
failure evidence. If upstream compatibility changes, rerun the isolated checks,
review the results, update this snapshot and promote newly qualified profiles
into the required gate.

A platform runtime needs its own Python/SDK/vendor/image qualification. No
managed image, deployed service or live provider job was changed or qualified
by this work.
