# QuTiP: local simulation and result recording

Marqov 0.6.1 adds an optional QuTiP installation extra. The existing
`marqov.qutip.record` helper prints solver observables as JSON; it does not run a
solver, submit a job, or upload files. These examples run locally without an
account. Managed execution requires separately qualified compiler and execution
environments; installing an extra does not establish managed-service support.

## Install

Use Python 3.12 or later in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "marqov[qutip]==0.6.1"
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
`marqov[qutip]` adds QuTiP `>=5.3.0,<6.0.0`. `marqov[all]` also includes it;
`marqov[qutip,qiskit]` combines selected extras. A base `marqov` installation does
not require QuTiP. These are options on the same SDK distribution, not separate
SDK packages.

## Run a decay simulation

Copy this complete program into `qutip_decay.py`, then run
`python qutip_decay.py > decay.json`. Installing the wheel does not install the
repository's `examples/` directory. If you have a source checkout, the same
program is available as [examples/qutip_decay.py](../examples/qutip_decay.py).

```python
import numpy as np
from qutip import basis, mesolve, sigmam, sigmaz
from marqov.qutip import record

times = np.linspace(0, 5, 11)
result = mesolve(
    0.5 * sigmaz(),
    basis(2, 0),
    times,
    c_ops=[np.sqrt(0.4) * sigmam()],
    e_ops=[sigmaz()],
    options={"store_states": False, "progress_bar": ""},
)
record(result, observable_names=["sigma_z"])
```

The recorded JSON contains `result_type: "open-system-dynamics"`,
`schema_version: 1`, eleven `times`, and eleven values in
`observables.sigma_z`. For this model, the expected curve is
`2 * exp(-0.4 * t) - 1`: it starts at +1 and approaches -1.

`record()` returns `None` and prints one JSON object followed by a newline.
Keep progress bars and other output off stdout when capturing JSON. Send your
own diagnostic messages to stderr instead.

## Names and supported values

Pass one unique observable name per expectation series. Explicit
`observable_names` takes precedence; otherwise the helper uses string keys from
QuTiP's dictionary `e_ops` result when available, then defaults to `obs_0`,
`obs_1`, and so on. Duplicate names would overwrite entries in the JSON mapping,
so callers must avoid them.

Times and observables must be finite. Real observables are supported; small
imaginary numerical residuals are discarded only when the largest imaginary
magnitude is at most `1e-6 * max(1, max(abs(real_values)))` across that series.
Larger imaginary components raise `ValueError`; complex-valued series are not
encoded as real/imaginary pairs.

## Reproduce Monte Carlo trajectories

This standalone example records and reconstructs the per-trajectory seeds.
Use the full saved seed objects, including their spawn keys, rather than just
their entropy. The helper import is `marqov.qutip.record.seed_from_json`.

```python
import contextlib
import io
import json
import numpy as np
from qutip import basis, mcsolve, sigmam, sigmaz
from marqov.qutip import record
from marqov.qutip.record import seed_from_json

times = np.linspace(0, 5, 11)
kwargs = dict(
    c_ops=[np.sqrt(0.4) * sigmam()],
    e_ops=[sigmaz()],
    ntraj=32,
    options={"map": "serial", "store_states": False, "progress_bar": ""},
)
first = mcsolve(0.5 * sigmaz(), basis(2, 0), times, seeds=12345, **kwargs)
output = io.StringIO()
with contextlib.redirect_stdout(output):
    record(first, observable_names=["sigma_z"])
payload = json.loads(output.getvalue())
replayed = mcsolve(
    0.5 * sigmaz(), basis(2, 0), times,
    seeds=[seed_from_json(seed) for seed in payload["seeds"]],
    **kwargs,
)
np.testing.assert_allclose(first.expect[0], replayed.expect[0], rtol=0, atol=1e-12)
print("Seed replay passed")
```

Keep the Hamiltonian, initial state, time grid, collapse operators, trajectory
count, solver options and dependency versions alongside the seeds. The recorded
JSON does not contain all these inputs. Seeds alone are not a complete experiment
archive or a guarantee of equality across dependency versions and machines.
Parallel reductions can differ at floating-point precision; the example uses
serial execution and checks agreement within tolerance.

See QuTiP's [Monte Carlo reproducibility documentation](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-monte.html)
for the underlying solver seed behavior.

## Save quantum states separately

For observables only, use `store_states=False` as above. If you request stored
states, save them yourself before passing their reference to `record()`.
This standalone example saves dense matrices to a local NumPy archive:

```python
from pathlib import Path
import numpy as np
from qutip import basis, mesolve, sigmam, sigmaz
from marqov.qutip import record

times = np.linspace(0, 5, 11)
result = mesolve(
    0.5 * sigmaz(), basis(2, 0), times,
    c_ops=[np.sqrt(0.4) * sigmam()], e_ops=[sigmaz()],
    options={"store_states": True, "progress_bar": ""},
)
path = Path("decay_states.npz")
np.savez(path, times=times, states=np.stack([state.full() for state in result.states]))
record(result, observable_names=["sigma_z"], states_artifact_path=str(path))
```

`states_artifact_path` adds a `states_artifact` string to the JSON. It does not
write, upload, validate or check access to the file. A local filename is useful
only to a consumer that can access that file. Arrange storage and access yourself
when sharing results. The dense-array example does not preserve all QuTiP object
metadata, such as subsystem dimensions; choose an appropriate format for your
own state-reconstruction needs and data size.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError: qutip` | Install the `[qutip]` extra using the Python environment that runs your script. |
| States present without an artifact reference | Disable state storage or save the states yourself and supply the reference. |
| Stochastic result has no seeds | Use the solver result with its per-trajectory seeds intact. |
| Observable count does not match names | Supply exactly one name per expectation series. |
| Non-finite times or observables | Inspect the input time grid and solver output for NaN/Inf; the helper refuses invalid values. |
| Genuinely complex expectation values | Use suitable Hermitian observables or handle complex results separately. |
| JSON parsing fails | Ensure stdout contains a single `record()` output, with no progress bars or diagnostic messages. |

The [wheel smoke check](../tools/smoke_qutip_wheel.py) verifies base-versus-extra
installation and the analytic decay example in a temporary environment. See
[the release procedure](../RELEASING.md) for its invocation.
