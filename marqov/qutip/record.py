"""Capture a qutip.solver.Result into the script-executor stdout-JSON contract.

Invariants (spec §7): .states NEVER go through stdout (offload or hard-fail);
stochastic (mcsolve) runs MUST carry seeds or they are not Capsule-reproducible.
Verified on qutip 5.3.0 — see plan header for the API facts this relies on.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def _is_stochastic(result: Any) -> bool:
    # mcsolve carries .num_trajectories; mesolve/sesolve do not. stats["solver"]
    # is "Master Equation Evolution" for BOTH, so it cannot discriminate.
    return hasattr(result, "num_trajectories")


def _seed_to_str(s: Any) -> str:
    # result.seeds are numpy SeedSequence; int(s) raises. The reusable value is
    # s.entropy (128-bit int) — emit as a STRING to survive JS/jsonb float64.
    # Verified 5.3.0: seeds=[int(entropy)] reproduces mcsolve exactly.
    entropy = getattr(s, "entropy", None)
    if entropy is None:  # not a SeedSequence -> fail clearly, don't int() it
        raise ValueError(f"cannot serialise seed of type {type(s).__name__}")
    return str(entropy)


def _coerce_series(arr: Any, name: str) -> list[float]:
    a = np.asarray(arr)
    # Reject non-finite FIRST, on the raw array — before the complex gate below.
    # np.isfinite on a complex array is False when EITHER component is non-finite,
    # so this catches a NaN/Inf hiding in the imaginary part, which would otherwise
    # slip through the gate (NaN > tol is False) and be silently dropped by taking
    # a.real. NaN/Inf are also invalid JSON. Fail here, naming the index; never
    # null-substitute — a decay curve with holes reads as data.
    nonfinite = np.where(~np.isfinite(a))[0] if a.size else np.empty(0, dtype=int)
    if nonfinite.size:
        raise ValueError(
            f"observable '{name}' has a non-finite value (NaN/Inf) at index "
            f"{int(nonfinite[0])}; recorded observables must be finite."
        )
    if np.iscomplexobj(a):
        # Tolerance anchor is the SERIES MAXIMUM |real| (global over the series,
        # not per-point), with an absolute 1e-6 floor. For a curve decaying to
        # zero the floor holds the tolerance at 1e-6 across all samples rather
        # than tightening toward the tail and rejecting numerical noise there —
        # a deliberate choice. Hard-fail on a genuinely complex observable.
        scale = max(1.0, float(np.abs(a.real).max())) if a.size else 1.0
        imag_max = float(np.abs(a.imag).max()) if a.size else 0.0
        if imag_max > 1e-6 * scale:
            raise ValueError(
                f"observable '{name}' has genuinely complex expectation values "
                "(non-Hermitian operator). The MVP records real observables only; "
                "use a Hermitian operator, or await the {re, im} schema decision (OD-2)."
            )
        return [float(x.real) for x in a]
    return [float(x) for x in a]


def record(result: Any, observable_names: list[str] | None = None, *,
           states_artifact_path: str | None = None) -> None:
    """Serialise `result` and print one JSON object to stdout.

    Observable names: explicit `observable_names` wins; otherwise the keys of a
    dict `e_ops` (via result.e_data) are used; otherwise obs_0, obs_1, ....
    """
    states = getattr(result, "states", None)
    if states is not None and len(states) > 0 and states_artifact_path is None:
        raise ValueError(
            "result.states are present but no states_artifact_path was given. "
            "Density matrices must be offloaded by reference, never serialised "
            "to stdout. Pass states_artifact_path=... or drop states."
        )

    seeds = getattr(result, "seeds", None)
    if _is_stochastic(result) and not seeds:
        raise ValueError(
            "stochastic solver (mcsolve) result has no seeds; an unseeded run is "
            "not Capsule-reproducible. Reuse a prior result's .seeds."
        )

    times_arr = np.asarray(result.times)
    bad_t = np.where(~np.isfinite(times_arr))[0] if times_arr.size else np.empty(0, dtype=int)
    if bad_t.size:
        raise ValueError(
            f"times has a non-finite value (NaN/Inf) at index {int(bad_t[0])}; "
            "check the tlist passed to the solver."
        )
    times = [float(t) for t in times_arr]
    expect = result.expect if result.expect is not None else []

    if observable_names is not None:
        names = observable_names
    else:
        edata = getattr(result, "e_data", None)
        if isinstance(edata, dict) and len(edata) == len(expect) \
                and all(isinstance(k, str) for k in edata):
            names = list(edata.keys())          # user passed e_ops as a dict
        else:
            names = [f"obs_{i}" for i in range(len(expect))]
    if len(names) != len(expect):
        raise ValueError(
            f"observable_names has {len(names)} entries but result has "
            f"{len(expect)} observables."
        )
    observables = {n: _coerce_series(a, n) for n, a in zip(names, expect)}

    payload: dict[str, Any] = {
        "result_type": "open-system-dynamics",
        "schema_version": 1,
        "times": times,
        "observables": observables,
    }
    if seeds:
        # REPRODUCIBILITY CAVEAT (verified qutip 5.3): these seeds reproduce mcsolve
        # bit-identically ONLY under options={"map": "serial"}. The parallel map does
        # not preserve seed->trajectory assignment, so a re-run with the same seeds
        # produces a DIFFERENT trajectory set (max|Δ| up to 2.0). A Capsule re-run of a
        # stochastic open-system result must force serial to actually reproduce.
        payload["seeds"] = [_seed_to_str(s) for s in seeds]
    if states_artifact_path is not None:
        payload["states_artifact"] = states_artifact_path

    # allow_nan=False: any non-finite that slipped through (e.g. in `times`)
    # raises here rather than emitting invalid JSON.
    print(json.dumps(payload, allow_nan=False))
