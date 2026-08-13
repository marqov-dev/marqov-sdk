import io, json, contextlib
import numpy as np
import pytest
from types import SimpleNamespace
from marqov.qutip.record import record

def _fake(times, expect, states=None, seeds=None, stochastic=False):
    ns = SimpleNamespace(times=np.asarray(times), expect=expect, states=states, seeds=seeds)
    if stochastic:
        ns.num_trajectories = 4
    return ns

def test_record_emits_times_and_observables():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        record(_fake([0.0, 1.0], [np.array([1.0, 0.5])]), observable_names=["sz"])
    out = json.loads(buf.getvalue())
    assert out["result_type"] == "open-system-dynamics"
    assert out["schema_version"] == 1  # required for the platform chart renderer
    assert out["observables"]["sz"] == [1.0, 0.5]

def test_record_rejects_states_without_offload():
    with pytest.raises(ValueError, match="states"):
        record(_fake([0.0], [np.array([1.0])], states=[object()]))

def test_record_rejects_unseeded_stochastic():
    with pytest.raises(ValueError, match="seed"):
        record(_fake([0.0], [np.array([1.0])], seeds=None, stochastic=True))

def test_record_rejects_genuinely_complex_observable():
    with pytest.raises(ValueError, match="complex"):
        record(_fake([0.0], [np.array([0.0 + 1.0j])]), observable_names=["sm"])

def test_record_coerces_near_real_complex():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        record(_fake([0.0], [np.array([1.0 + 1e-9j])]), observable_names=["sz"])
    assert json.loads(buf.getvalue())["observables"]["sz"] == [1.0]

def test_record_rejects_nonfinite_observable():
    # NaN/Inf are invalid JSON; must fail at emit, naming the offending index,
    # never silently substitute null.
    with pytest.raises(ValueError, match="non-finite.*index 1"):
        record(_fake([0.0, 1.0], [np.array([1.0, np.nan])]), observable_names=["sz"])

def test_record_rejects_inf_observable():
    with pytest.raises(ValueError, match="non-finite"):
        record(_fake([0.0], [np.array([np.inf])]), observable_names=["sz"])

def test_record_rejects_nonfinite_times():
    # times gets the same named + indexed validation (a bad tlist/linspace is the
    # likely source, and it isn't covered by the observable coercion path).
    with pytest.raises(ValueError, match="times.*index 1"):
        record(_fake([0.0, np.inf], [np.array([1.0, 0.5])]), observable_names=["sz"])

def test_record_rejects_nonfinite_in_imaginary_part():
    # Regression: a NaN in the IMAGINARY part must be rejected, not slipped
    # through the complex gate (NaN > tol is False) and then silently dropped by
    # taking .real. The finite check runs on the raw complex array first.
    with pytest.raises(ValueError, match="non-finite"):
        record(_fake([0.0, 1.0], [np.array([1.0 + 0j, complex(0.5, float("nan"))])]),
               observable_names=["sz"])

def test_record_real_mesolve_integration():
    pytest.importorskip("qutip")
    from qutip import basis, sigmaz, sigmam, mesolve
    res = mesolve(sigmaz(), basis(2, 0), np.linspace(0, 1, 5),
                  c_ops=[0.1 * sigmam()], e_ops=[sigmaz()])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        record(res, observable_names=["sz"])
    out = json.loads(buf.getvalue())
    assert len(out["times"]) == 5 and len(out["observables"]["sz"]) == 5
    assert "seeds" not in out

def test_record_mcsolve_seeds_are_distinct_per_trajectory():
    """Each trajectory needs its OWN seed. The trajectories of one mcsolve call are
    spawned children of one root SeedSequence: they share `entropy` and differ ONLY
    by `spawn_key`. Serialising `entropy` alone emits the same value ntraj times, so
    a replay collapses the ensemble to one trajectory repeated. This one-line
    distinctness check is the cheapest possible guard against that."""
    pytest.importorskip("qutip")
    from qutip import basis, sigmaz, sigmam, mcsolve
    args = (sigmaz(), basis(2, 0), np.linspace(0, 1, 3))
    kw = dict(c_ops=[0.4 * sigmam()], e_ops=[sigmaz()], ntraj=8, options={"map": "serial"})
    res = mcsolve(*args, **kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        record(res, observable_names=["sz"])
    seeds_json = json.loads(buf.getvalue())["seeds"]
    assert len(seeds_json) == len(res.seeds)
    canonical = [json.dumps(s, sort_keys=True) for s in seeds_json]
    assert len(set(canonical)) == len(canonical), (
        f"seeds must be distinct per trajectory; got {len(set(canonical))} "
        f"distinct of {len(canonical)}"
    )


def test_record_mcsolve_seeds_round_trip_reproduces():
    """Replaying from recorded seeds must reproduce `expect` exactly.

    The collapse rate and ntraj are chosen so that trajectories actually JUMP: with
    a weak c_op nearly every trajectory is jump-free, in which case a degenerate
    (all-identical) seed set coincidentally reproduces and the test proves nothing.
    Measured with 0.1*sigmam/ntraj=4: 57/60 runs jump-free -> passed trivially;
    the 3 that jumped failed 3/3. With 0.4*sigmam/ntraj=8 roughly 80% of runs jump.

    map=serial: seed->trajectory assignment is not stable under the parallel map.
    """
    pytest.importorskip("qutip")
    from qutip import basis, sigmaz, sigmam, mcsolve
    from marqov.qutip.record import seed_from_json
    args = (sigmaz(), basis(2, 0), np.linspace(0, 1, 3))
    kw = dict(c_ops=[0.4 * sigmam()], e_ops=[sigmaz()], ntraj=8, options={"map": "serial"})
    # Loop until we get a run with at least one collapse — a jump-free run does not
    # exercise the seed round-trip at all. Bounded so a pathological RNG can't hang.
    for _ in range(25):
        res = mcsolve(*args, **kw)
        if not np.allclose(res.expect[0], 1.0):
            break
    else:
        pytest.skip("no trajectory collapsed in 25 attempts; nothing to reproduce")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        record(res, observable_names=["sz"])
    seeds_json = json.loads(buf.getvalue())["seeds"]
    res2 = mcsolve(*args, seeds=[seed_from_json(s) for s in seeds_json], **kw)
    np.testing.assert_allclose(res.expect[0], res2.expect[0])
