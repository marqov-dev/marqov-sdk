# Provider Resolution Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace credential-sniffing provider dispatch in `MarqovDevice` with a single explicit provider resolution step, eliminating the duplicated if/elif chains, the dead else branch, and the format/device dispatch mismatch.

**Architecture:** A new `detect_provider(backend, params)` function in `marqov/backends.py` resolves the provider exactly once (honoring an explicit `provider` param first, then sniffing with documented priority). `MarqovDevice.__init__` stores the result as `self._provider`; all three dispatch sites (`_get_provider_device`, `run`, `_to_backend_format`) branch on that resolved string instead of re-sniffing params. The ARN-region helper moves to `backends.py` so `device.py` stops duplicating it.

**Tech Stack:** Python 3.12, pytest (run via `uv run pytest`), ruff. No new dependencies.

**Background / why:** The SDK was Braket-native (implicit `else` → Braket) and was deliberately made vendor-neutral in commit `8259451`. That conversion left: (1) two parallel if/elif chains in `_get_provider_device` and `run()` that must be kept in sync by hand, (2) a provably unreachable `else: raise` in `run()` (because `_get_provider_device()` raises first at device.py:215), (3) `_to_backend_format` dispatching on `is_ibm()/is_azure()` without the local-backend short-circuit the other two methods have — so `{"backend": "local", "ibm_token": ...}` produces a Qiskit circuit fed to a Braket `LocalSimulator` (crash), and (4) accidental sniffing priority: `is_ibm` beats `is_braket` only because IBM was added to the chain earlier, so params mixing a stray `ibm_channel` with an explicit `device_arn` silently route to IBM. Industry standard (Terraform, Libcloud, LiteLLM, the codebase's own `ExecutorFactory`) is an explicit provider identifier; sniffing is kept only as a fallback with documented priority.

**Documented sniffing priority (new):** explicit `params["provider"]` → backend name `local`/`marqov-sim` → `device_arn` (most provider-specific signal) → `ibm_token`/`ibm_channel` → `azure_subscription_id`. Note this deliberately promotes Braket above IBM relative to the old elif order; no in-repo caller passes mixed-provider params (verified 2026-06-11 review), so this is a documentation-and-correctness change, not a breaking one.

**Behavior change to be aware of:** provider resolution errors (unknown provider, blank `device_arn`, missing required params for an explicit provider) now raise at `MarqovDevice(...)` construction instead of at first `_get_provider_device()` call. Fail-fast is intended; Task 3 updates the two affected tests. This is a public SDK — Task 8 records both observable changes (Braket-over-IBM sniffing priority, construction-time errors) in a CHANGELOG and bumps the version to 0.3.0.

## Cross-cutting invariants

These must hold after every task and are pinned by tests:

| Invariant | Machine check |
|---|---|
| Provider is resolved exactly once, at construction; all dispatch branches only on `self._provider` | `TestDispatchAgreement` (Task 6) — behavior-level: format output type must match the resolved provider for mixed-param inputs. (A source-grep test was considered and rejected as brittle.) |
| Format dispatch and device dispatch always agree | `TestDispatchAgreement` parametrized over mixed-param combinations (Task 6 Step 1) |
| Sniffing priority: explicit `provider` → local backend names → `device_arn` → IBM keys → Azure keys | `test_device_arn_beats_stray_ibm_key`, `test_blank_arn_with_ibm_key_resolves_ibm` (Task 1) |
| Resolution errors raise at construction, never at `run()` | `TestProviderDetectionErrors` (Task 3) |
| Explicit `provider` implies its required params are present | `TestExplicitProviderValidation` (Task 1) — `braket` requires non-empty `device_arn`; `azure` requires the three workspace keys |

---

### Task 1: `detect_provider()` in backends.py

**Files:**
- Modify: `marqov/backends.py`
- Create: `tests/test_backends.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backends.py`:

```python
"""Tests for backend/provider detection utilities."""

import pytest

from marqov.backends import detect_provider


class TestDetectProvider:
    def test_explicit_provider_wins(self):
        # Explicit declaration beats every sniffed signal
        params = {"provider": "azure", "device_arn": "arn:aws:braket:::device/x", "ibm_token": "t"}
        assert detect_provider("some-backend", params) == "azure"

    def test_explicit_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider 'not-a-provider'"):
            detect_provider("some-backend", {"provider": "not-a-provider"})

    def test_local_backend_names(self):
        assert detect_provider("local", {}) == "local"
        assert detect_provider("marqov-sim", {}) == "local"

    def test_device_arn_resolves_braket(self):
        params = {"device_arn": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"}
        assert detect_provider("rigetti-ankaa-3", params) == "braket"

    def test_device_arn_beats_stray_ibm_key(self):
        # device_arn is the most provider-specific signal; a stray ibm_channel
        # merged in by a config layer must not hijack routing (documented priority)
        params = {"device_arn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1", "ibm_channel": "ibm_quantum"}
        assert detect_provider("sv1", params) == "braket"

    def test_ibm_keys_resolve_ibm(self):
        assert detect_provider("ibm-kyoto", {"ibm_token": "t"}) == "ibm"
        assert detect_provider("ibm-kyoto", {"ibm_channel": "ibm_quantum"}) == "ibm"

    def test_azure_key_resolves_azure(self):
        assert detect_provider("quantinuum-h1", {"azure_subscription_id": "sub"}) == "azure"

    def test_blank_device_arn_raises_targeted_error(self):
        with pytest.raises(
            ValueError,
            match="device_arn is present but empty for backend 'rigetti-test'",
        ):
            detect_provider("rigetti-test", {"device_arn": ""})

    def test_blank_arn_with_ibm_key_resolves_ibm(self):
        # Pins the sniffing order: a blank device_arn is not a Braket signal,
        # so other credentials may still resolve. Deliberate, not accidental.
        params = {"device_arn": "", "ibm_token": "t"}
        assert detect_provider("some-backend", params) == "ibm"

    def test_no_signals_raises_generic_error(self):
        with pytest.raises(ValueError, match="Cannot determine provider for backend 'mystery'"):
            detect_provider("mystery", {})


class TestExplicitProviderValidation:
    """Explicit provider declaration must come with its required params.

    Sniffed resolution couples branch-entry to key existence; the explicit
    path must validate the same keys or downstream code raises bare KeyError.
    """

    def test_explicit_braket_requires_device_arn(self):
        with pytest.raises(
            ValueError, match="provider 'braket' requires a non-empty device_arn"
        ):
            detect_provider("sv1", {"provider": "braket"})

    def test_explicit_braket_rejects_blank_device_arn(self):
        with pytest.raises(
            ValueError, match="provider 'braket' requires a non-empty device_arn"
        ):
            detect_provider("sv1", {"provider": "braket", "device_arn": ""})

    def test_explicit_azure_requires_workspace_params(self):
        with pytest.raises(
            ValueError,
            match="provider 'azure' requires azure_subscription_id, azure_resource_group, azure_workspace_name",
        ):
            detect_provider("quantinuum-h1", {"provider": "azure", "azure_subscription_id": "sub"})

    def test_explicit_ibm_requires_no_params(self):
        # QiskitRuntimeService falls back to saved credentials; channel/instance
        # have defaults in _get_provider_device, so no params are mandatory.
        assert detect_provider("ibm-kyoto", {"provider": "ibm"}) == "ibm"

    def test_explicit_local_requires_no_params(self):
        assert detect_provider("local", {"provider": "local"}) == "local"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backends.py -q`
Expected: ImportError — `cannot import name 'detect_provider' from 'marqov.backends'`

- [ ] **Step 3: Implement `detect_provider`**

Append to `marqov/backends.py`:

```python
PROVIDERS = ("local", "braket", "ibm", "azure")

_AZURE_REQUIRED = ("azure_subscription_id", "azure_resource_group", "azure_workspace_name")


def detect_provider(backend: str, params: dict) -> str:
    """Resolve the provider for a backend/params pair.

    Note: these are MarqovDevice-level providers, distinct from the executor
    registry in marqov.executors.factory (new executor providers do not
    automatically appear here).

    Resolution order (first match wins):
      1. Explicit params["provider"] — the declared provider always wins,
         and its required params are validated immediately.
      2. Backend name "local"/"marqov-sim" → "local".
      3. device_arn present and non-empty → "braket" (most specific signal).
      4. ibm_token or ibm_channel → "ibm".
      5. azure_subscription_id → "azure".

    Sniffing (steps 3-5) exists for callers that predate explicit provider
    declaration; new callers should pass params["provider"].

    Raises:
        ValueError: If the provider is unknown, cannot be determined, or an
            explicitly declared provider is missing its required params.
    """
    explicit = params.get("provider")
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{explicit}'. Supported: {', '.join(PROVIDERS)}."
            )
        if explicit == "braket" and not params.get("device_arn"):
            raise ValueError(
                f"provider 'braket' requires a non-empty device_arn "
                f"(backend '{backend}')."
            )
        if explicit == "azure" and not all(params.get(k) for k in _AZURE_REQUIRED):
            raise ValueError(
                f"provider 'azure' requires {', '.join(_AZURE_REQUIRED)} "
                f"(backend '{backend}')."
            )
        return explicit

    if backend in ("local", "marqov-sim"):
        return "local"
    if is_braket(params):
        return "braket"
    if is_ibm(params):
        return "ibm"
    if is_azure(params):
        return "azure"

    if "device_arn" in params:
        raise ValueError(
            f"device_arn is present but empty for backend '{backend}'. "
            f"Provide a valid AWS Braket device ARN."
        )
    raise ValueError(
        f"Cannot determine provider for backend '{backend}'. "
        f"Params must include one of: provider, device_arn (AWS Braket), "
        f"ibm_token/ibm_channel (IBM Quantum), azure_subscription_id (Azure Quantum)."
    )
```

Note: the sniffed azure path (step 5) only checks `azure_subscription_id`; `_get_provider_device` would still KeyError on a sniffed-azure params dict missing `azure_resource_group`/`azure_workspace_name`. That mirrors current behavior exactly and is out of scope here — tightening sniffed-path validation is a follow-up once explicit `provider` is adopted.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backends.py -q`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add marqov/backends.py tests/test_backends.py
git commit -m "feat(backends): add detect_provider with explicit-provider priority"
```

---

### Task 2: Move ARN-region helper to backends.py

**Files:**
- Modify: `marqov/backends.py`
- Modify: `marqov/executors/braket.py:37-53` (replace function with import)
- Modify: `tests/test_executors.py:16` (import path)
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backends.py`:

```python
class TestExtractRegionFromArn:
    def test_qpu_arn_region(self):
        from marqov.backends import extract_region_from_arn

        arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
        assert extract_region_from_arn(arn) == "eu-north-1"

    def test_simulator_arn_empty_region_falls_back(self):
        from marqov.backends import extract_region_from_arn

        arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        assert extract_region_from_arn(arn) == "us-east-1"

    def test_malformed_arn_falls_back(self):
        from marqov.backends import extract_region_from_arn

        assert extract_region_from_arn("not-an-arn") == "us-east-1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backends.py::TestExtractRegionFromArn -q`
Expected: ImportError — `cannot import name 'extract_region_from_arn'`

- [ ] **Step 3: Move the function**

Append to `marqov/backends.py` (verbatim move of the body from `marqov/executors/braket.py:37-53`, renamed public):

```python
def extract_region_from_arn(arn: str) -> str:
    """Extract AWS region from a Braket device ARN.

    ARN format: arn:aws:braket:{region}::device/{type}/{provider}/{name}
    Simulator ARNs use empty region (:::) which means us-east-1.
    """
    parts = arn.split(":")
    if len(parts) >= 4:
        region = parts[3]
        return region if region else "us-east-1"
    return "us-east-1"
```

In `marqov/executors/braket.py`, delete the `_extract_region_from_arn` function definition (lines 37-53) and add to the imports at the top of the file:

```python
from marqov.backends import extract_region_from_arn as _extract_region_from_arn
```

(The alias keeps the call site at `braket.py:114` unchanged.)

In `tests/test_executors.py` line 16, change:

```python
from marqov.executors.braket import BraketExecutorConfig, _extract_region_from_arn
```

to:

```python
from marqov.backends import extract_region_from_arn as _extract_region_from_arn
from marqov.executors.braket import BraketExecutorConfig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backends.py tests/test_executors.py -q`
Expected: all pass (3 new + existing executor tests)

- [ ] **Step 5: Commit**

```bash
git add marqov/backends.py marqov/executors/braket.py tests/test_backends.py tests/test_executors.py
git commit -m "refactor(backends): move ARN region extraction to backends.py"
```

---

### Task 3: Resolve provider once in MarqovDevice.__init__

**Files:**
- Modify: `marqov/device.py:5,18-21`
- Modify: `tests/test_device_integration.py` (TestProviderDetectionErrors)

- [ ] **Step 1: Update the error-path tests (construction-time failure)**

In `tests/test_device_integration.py`, replace the `TestProviderDetectionErrors` class body with:

```python
class TestProviderDetectionErrors:
    """Provider resolution fails fast at construction."""

    def test_blank_device_arn_names_the_real_problem(self):
        with pytest.raises(
            ValueError,
            match="device_arn is present but empty for backend 'rigetti-test'",
        ):
            MarqovDevice("rigetti-test", {"device_arn": ""})

    def test_missing_provider_keys_lists_options(self):
        with pytest.raises(ValueError, match="Cannot determine provider"):
            MarqovDevice("mystery", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_device_integration.py::TestProviderDetectionErrors -q`
Expected: 2 failed — `DID NOT RAISE` (construction currently succeeds; errors only raise later)

- [ ] **Step 3: Resolve provider in __init__**

In `marqov/device.py`, change the import line (line 5) — keep the `is_*` helpers, they are still used by `run()` and `_to_backend_format` until Tasks 5-6:

```python
from marqov.backends import (
    detect_provider,
    extract_region_from_arn,
    is_azure,
    is_braket,
    is_ibm,
    is_simulator,
)
```

and `__init__` (lines 18-21):

```python
    def __init__(self, backend: str, params: dict) -> None:
        self._backend = backend
        self._params = params
        self._provider = detect_provider(backend, params)
        self._provider_device = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_device_integration.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add marqov/device.py tests/test_device_integration.py
git commit -m "refactor(device): resolve provider once at construction"
```

---

### Task 4: Dispatch _get_provider_device on resolved provider

**Files:**
- Modify: `marqov/device.py:33-95`

- [ ] **Step 1: Rewrite _get_provider_device**

Replace the whole method (currently device.py:33-95) with:

```python
    def _get_provider_device(self):
        """Lazy-load and return the underlying provider device."""
        if self._provider_device is not None:
            return self._provider_device

        if self._provider == "local":
            from braket.devices import LocalSimulator

            self._provider_device = LocalSimulator()

        elif self._provider == "ibm":
            from qiskit_ibm_runtime import QiskitRuntimeService

            kwargs = {
                "channel": self._params.get("ibm_channel", "ibm_quantum"),
                "instance": self._params.get("ibm_instance", "ibm-q/open/main"),
            }
            if self._params.get("ibm_token"):
                kwargs["token"] = self._params["ibm_token"]

            service = QiskitRuntimeService(**kwargs)
            self._provider_device = service.backend(self._backend)

        elif self._provider == "azure":
            from azure.quantum import Workspace

            workspace = Workspace(
                subscription_id=self._params["azure_subscription_id"],
                resource_group=self._params["azure_resource_group"],
                name=self._params["azure_workspace_name"],
                location=self._params.get("azure_location", "eastus"),
            )
            targets = workspace.get_targets(name=self._backend)
            self._provider_device = targets

        else:  # "braket" — detect_provider guarantees membership in PROVIDERS
            # Import style matches marqov/executors/braket.py:28 for grep-ability
            from braket.aws import AwsDevice, AwsSession
            import boto3

            device_arn = self._params["device_arn"]
            region = extract_region_from_arn(device_arn)
            session = AwsSession(boto3.Session(region_name=region))
            self._provider_device = AwsDevice(device_arn, aws_session=session)

        return self._provider_device
```

This deletes the inline ARN parsing (old lines 75-78) and the entire error-raising else block (old lines 83-93) — `detect_provider` already raised at construction for those cases.

- [ ] **Step 2: Run the device and backend tests**

Run: `uv run pytest tests/test_device_integration.py tests/test_backends.py -q`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add marqov/device.py
git commit -m "refactor(device): dispatch _get_provider_device on resolved provider"
```

---

### Task 5: Dispatch run() on resolved provider, delete dead else

**Files:**
- Modify: `marqov/device.py` (run method, currently lines 197-293)

- [ ] **Step 1: Rewrite run()'s dispatch conditions**

In `run()`, change the four branch conditions (keep every branch body exactly as-is):

- `if self._backend in ("local", "marqov-sim"):` → `if self._provider == "local":`
- `elif is_ibm(self._params):` → `elif self._provider == "ibm":`
- `elif is_azure(self._params):` → `elif self._provider == "azure":`
- `elif is_braket(self._params):` → `else:  # "braket"`

Delete the final `else: raise ValueError("Cannot determine provider...")` block entirely (currently lines 288-293) — it was unreachable (the identical check already ran in `_get_provider_device`, and now runs at construction).

- [ ] **Step 2: Run the full device tests**

Run: `uv run pytest tests/test_device_integration.py -q`
Expected: all pass (the LocalSimulator end-to-end tests in TestRunIntegration exercise the "local" branch for real)

- [ ] **Step 3: Commit**

```bash
git add marqov/device.py
git commit -m "refactor(device): dispatch run() on resolved provider, drop unreachable else"
```

---

### Task 6: Fix _to_backend_format dispatch (the local+ibm_token bug)

**Files:**
- Modify: `marqov/device.py` (_to_backend_format, currently lines 170-183; import line 5)
- Test: `tests/test_device_integration.py` (TestToBackendFormat)

- [ ] **Step 1: Write the failing regression tests (parametrized dispatch-agreement invariant)**

Add to `tests/test_device_integration.py` (alongside `TestToBackendFormat`):

```python
class TestDispatchAgreement:
    """Invariant: format dispatch and device dispatch always agree, including
    for mixed-param inputs where sniffed signals conflict. Both must follow
    the provider resolved at construction, never raw param-sniffing."""

    @pytest.mark.parametrize(
        ("backend", "params", "expected_format"),
        [
            # local backend + stray IBM key: previously produced a Qiskit
            # circuit fed to a Braket LocalSimulator (crash)
            ("local", {"backend": "local", "ibm_token": "stray"}, "braket"),
            # device_arn + stray IBM key: resolved provider is braket,
            # format must be braket too
            (
                "sv1",
                {
                    "device_arn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                    "ibm_channel": "ibm_quantum",
                },
                "braket",
            ),
            # plain IBM params: qiskit format
            ("ibm-kyoto", {"ibm_token": "t"}, "qiskit"),
            # explicit provider overrides every sniffed signal
            (
                "quantinuum-h1",
                {
                    "provider": "azure",
                    "azure_subscription_id": "sub",
                    "azure_resource_group": "rg",
                    "azure_workspace_name": "ws",
                    "device_arn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                },
                "qiskit",
            ),
        ],
    )
    def test_format_matches_resolved_provider(self, backend, params, expected_format):
        from braket.circuits import Circuit as BraketCircuit
        from qiskit import QuantumCircuit

        device = MarqovDevice(backend, params)
        result = device._to_backend_format(Circuit().h(0).cnot(0, 1))
        if expected_format == "braket":
            assert isinstance(result, BraketCircuit)
        else:
            assert isinstance(result, QuantumCircuit)
```

- [ ] **Step 2: Run tests to verify the conflict cases fail**

Run: `uv run pytest tests/test_device_integration.py::TestDispatchAgreement -q`
Expected: the two mixed-param cases FAIL (`is_ibm()` is True so a Qiskit `QuantumCircuit` is returned where Braket is expected); the plain-IBM and explicit-azure cases pass

- [ ] **Step 3: Fix the dispatch**

Replace `_to_backend_format` (currently device.py:170-183) with:

```python
    def _to_backend_format(self, marqov_circuit: Circuit):
        """Convert a marqov.Circuit to the target backend's native format.

        - Braket backends (local, AWS): .to_braket() — auto-measures all qubits
        - IBM/Azure backends: .to_qiskit() + measure_all() if no measurements present
        """
        if self._provider in ("ibm", "azure"):
            qc = marqov_circuit.to_qiskit()
            if not qc.cregs:
                qc.measure_all()
            return qc

        # Braket format: local simulators and AWS Braket devices
        return marqov_circuit.to_braket()
```

- [ ] **Step 4: Run tests to verify they pass; remove now-unused imports**

Run: `uv run pytest tests/test_device_integration.py -q`
Expected: all pass (including all four TestDispatchAgreement cases)

`is_azure`, `is_braket`, `is_ibm` now have no remaining uses in device.py. The import line 5 should end as:

```python
from marqov.backends import detect_provider, extract_region_from_arn, is_simulator
```

- [ ] **Step 5: Full suite + lint**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: 330+ passed, 13 skipped; "All checks passed!"

- [ ] **Step 6: Commit**

```bash
git add marqov/device.py tests/test_device_integration.py
git commit -m "fix(device): make circuit format dispatch agree with provider dispatch"
```

---

### Task 7: Document the resolution order in the class docstring

**Files:**
- Modify: `marqov/device.py:9-16` (class docstring)

- [ ] **Step 1: Extend the MarqovDevice docstring**

Replace the class docstring (device.py:9-16) with:

```python
    """Wraps a quantum backend and provides a uniform run() interface.

    Scripts receive a MarqovDevice from get_device() and call device.run(circuit, shots)
    without needing vendor-specific branching. Accepts any supported circuit type
    (Braket, Qiskit, Cirq, PennyLane, QASM string, or Marqov Circuit) and
    automatically converts to the target backend's native format.

    Provider resolution happens once at construction via
    marqov.backends.detect_provider: an explicit params["provider"]
    ("local", "braket", "ibm", "azure") always wins; otherwise the provider
    is inferred from params with documented priority (device_arn, then IBM
    keys, then Azure keys). Unknown/ambiguous params raise ValueError here,
    not at run() time.
    """
```

- [ ] **Step 2: Run full suite + lint one final time**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: all green

- [ ] **Step 3: Commit**

```bash
git add marqov/device.py
git commit -m "docs(device): document provider resolution order"
```

---

### Task 8: CHANGELOG and version bump

This is a public SDK with external contributors building against it. Two observable behavior changes ship in this refactor and must be recorded: (1) sniffing priority now puts `device_arn` above IBM keys, (2) provider-resolution errors raise at `MarqovDevice(...)` construction instead of at first use.

**Files:**
- Create: `CHANGELOG.md`
- Modify: `pyproject.toml:3` (version)
- Modify: `marqov/__init__.py:36` (`__version__`)

- [ ] **Step 1: Create CHANGELOG.md**

```markdown
# Changelog

## 0.3.0 — 2026-06

### Changed
- **Provider resolution is now explicit-first.** `MarqovDevice` resolves its
  provider once at construction via `marqov.backends.detect_provider`. Pass
  `params["provider"]` ("local", "braket", "ibm", "azure") to declare the
  provider directly; sniffing from credential keys remains as a fallback.
- **Sniffing priority changed:** a non-empty `device_arn` now resolves to AWS
  Braket even if IBM/Azure credential keys are also present (previously IBM
  keys won). Mixed-provider params were never produced by marqov tooling, but
  external callers relying on the old accidental priority should pass
  `params["provider"]` explicitly.
- **Errors moved to construction time.** Unknown/undeterminable providers,
  blank `device_arn`, and missing required params for an explicitly declared
  provider now raise `ValueError` from `MarqovDevice(...)` / `get_device(...)`
  instead of from the first `run()` call.

### Fixed
- Circuit format conversion now follows the resolved provider, fixing a crash
  where a local-simulator run with stray IBM credentials in params produced a
  Qiskit circuit that was fed to a Braket `LocalSimulator`.

## 0.2.0 — 2026-06

- IBM SamplerV2 counts extraction fixed (`DataBin` creg fields via `keys()`).
- Vendor-neutral provider dispatch: removed the implicit Braket fallback.
- Version aligned across `pyproject.toml` and `marqov.__version__`.
```

- [ ] **Step 2: Bump both version strings and re-lock**

In `pyproject.toml` line 3: `version = "0.3.0"`
In `marqov/__init__.py` line 36: `__version__ = "0.3.0"`

(`tests/test_version.py` asserts these two stay equal — it will fail if only one is bumped.)

Then run `uv lock` — `uv.lock` records marqov's own version and must be regenerated.

- [ ] **Step 3: Run full suite + lint**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: all green (test_version passes with both strings at 0.3.0)

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md pyproject.toml marqov/__init__.py uv.lock
git commit -m "chore: add CHANGELOG, bump version to 0.3.0 for provider resolution changes"
```

---

## Out of scope (deliberately)

- **Multi-creg IBM counts** (`_extract_ibm_counts` reads only the first creg) — needs a design decision on the IBM workstream, not this refactor.
- **Porting to marqov-platform's vendored copy** (`marqov-platform/sdk/marqov/device.py`) — do as a follow-up sync after this lands; the platform copy currently also carries the pre-fix DataBin bug.
- **A full adapter-class registry** (one class per provider) — YAGNI at 4 providers in one file; the resolved-provider string gives the same correctness with far less churn.
- **Deprecation warnings for sniffed resolution** — consider only when the platform passes `provider` explicitly end-to-end.
