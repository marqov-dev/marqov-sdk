"""CUNQA executor for running circuits distributed across vQPUs on a Slurm cluster.

CUNQA (CESGA, Apache-2.0) is a distributed quantum-computing emulator that
runs vQPUs as Slurm tasks. This executor launches a family of vQPUs via
``qraise``, splits the requested shots evenly across them, gathers and merges
results, and always tears the family down with ``qdrop`` — including on
failure or cancellation, so a crashed or timed-out run doesn't strand Slurm
allocations.

This targets an early, correctness-only phase of a broader CUNQA integration
effort: N≈4, no failover/reproducibility/scaling logic yet.

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import CUNQAExecutor, CUNQAExecutorConfig
    >>>
    >>> config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00")
    >>> executor = CUNQAExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)

``cunqa`` is an optional dependency, and only importable on a machine with
CUNQA built and on ``sys.path``. CUNQA has no PyPI wheel and is not a
``marqov[...]`` extra — its exact ``qiskit==1.2.4`` pin would downgrade the
whole project's lockfile if declared in ``[project.optional-dependencies]``
(the same class of problem ``qilisdk``'s numpy floor caused). Install
``qiskit==1.2.4`` separately in the environment where CUNQA is built.

API surface verified 2026-08-21 directly against CESGA-Quantum-Spain/cunqa
source (cunqa/qpu.py, cunqa/qjob.py, cunqa/result.py, and the qraise C++
backend on `main`). Confirmed real shape:
- ``from cunqa.qpu import qraise, get_QPUs, run, qdrop``
- ``from cunqa.qjob import gather``
- ``qraise(n, t, *, family=None, co_located=True, mem_per_qpu=None, ...) -> str``
  (returns the family name). ``mem_per_qpu`` is opt-in — if omitted, the
  generated sbatch script has NO memory request at all. Units confirmed GB
  at ``cunqa/qpu.py:382-383`` — Python appends the ``"G"`` suffix itself
  (``f"--mem-per-qpu={mem_per_qpu}G"``) before shelling out. Note this is
  the PYTHON path's units; a raw CLI invocation of the C++ binary directly
  is a DIFFERENT code path with unverified flag spelling/units — check
  ``qraise --help`` on the live cluster before relying on it.
- ``get_QPUs(co_located: bool = False, family: str | None = None) -> list``
- ``run(circuits, qpus, param_values=None, **run_args) -> list`` — no
  documented ``seed`` kwarg (Phase 1 scope, not wired in here)
- ``gather(qjobs) -> list`` of results exposing ``.counts`` as a
  **property**, not a ``.get_counts()`` method. Note this is CUNQA's own
  result serialization — established for CUNQA specifically, not inherited
  from qiskit-aer's key format, even though CUNQA wraps a modified Aer
  internally. If a live run's modal outcome doesn't match the Aer-verified
  expected bin, check the key format first (hex vs binary, single string
  vs space-separated per-register) before suspecting the circuit itself.
- ``qdrop(*families: str, remove_logs: bool = False)``

``Circuit.to_qiskit()`` returns the gate sequence only — Marqov's ``Circuit``
IR has no measurement concept at all (confirmed: ``_SKIP_INSTRUCTIONS``
discards ``measure`` on the way in via ``from_qiskit``, and ``to_qiskit()``
is a bare unitary transpile with no measurement synthesis). Every executor
in this codebase is responsible for adding its own measurements —
``RigettiExecutor._build_measured_program`` is the existing precedent this
follows.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit

logger = logging.getLogger(__name__)


def add_measure_all(qiskit_circuit: Any) -> Any:
    """Return a copy of ``qiskit_circuit`` with a classical register added
    and every qubit measured into it.

    Marqov's ``Circuit.to_qiskit()`` never carries measurements (see module
    docstring) — this is the CUNQA equivalent of
    ``RigettiExecutor._build_measured_program``'s "measure every qubit"
    step. Exposed as a module-level function (not a private method) so
    tests can apply the exact same transformation the executor uses,
    instead of duplicating the logic and risking drift.

    Raises:
        ValueError: if the circuit already has any classical bits.
            ``Circuit.to_qiskit()`` never produces one (it's unitary-only)
            — this is the only input shape this function ever legitimately
            receives, so any pre-existing clbits are rejected outright
            rather than silently tolerated.
    """
    from qiskit import ClassicalRegister

    measured = qiskit_circuit.copy()
    if measured.num_clbits:
        raise ValueError(
            f"expected a measurement-free circuit, got one with {measured.num_clbits} clbits"
        )
    measured.add_register(ClassicalRegister(measured.num_qubits, "c"))
    measured.measure(range(measured.num_qubits), range(measured.num_qubits))
    return measured


class CunqaClient(Protocol):
    """The subset of CUNQA's Python API this executor calls.

    CUNQA itself exposes these as free functions in ``cunqa.qpu`` /
    ``cunqa.qjob``, not methods on an object — ``_CunqaModuleClient`` below
    adapts them to this Protocol so the executor can depend on an injectable
    interface instead of importing the module directly.
    """

    def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str: ...
    def get_QPUs(self, co_located: bool, family: str) -> list[Any]: ...
    def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]: ...
    def gather(self, jobs: list[Any]) -> list[Any]: ...
    def qdrop(self, family: str) -> None: ...


class _CunqaModuleClient:
    """Adapts CUNQA's free functions (cunqa.qpu / cunqa.qjob) to CunqaClient.

    A bare module object doesn't satisfy a self-bearing Protocol, and CUNQA's
    functions live in two different submodules — this class is the seam that
    makes both of those non-issues for the executor.
    """

    def __init__(self) -> None:
        from cunqa.qjob import gather as _gather
        from cunqa.qpu import get_QPUs as _get_QPUs
        from cunqa.qpu import qdrop as _qdrop
        from cunqa.qpu import qraise as _qraise
        from cunqa.qpu import run as _run

        self._qraise = _qraise
        self._get_QPUs = _get_QPUs
        self._run = _run
        self._gather = _gather
        self._qdrop = _qdrop

    def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
        return self._qraise(n_qpus, walltime, **kwargs)

    def get_QPUs(self, co_located: bool, family: str) -> list[Any]:
        return self._get_QPUs(co_located=co_located, family=family)

    def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]:
        return self._run(circuits, qpus, **run_args)

    def gather(self, jobs: list[Any]) -> list[Any]:
        return self._gather(jobs)

    def qdrop(self, family: str) -> None:
        self._qdrop(family)


@dataclass
class CUNQAExecutorConfig:
    """Configuration for the CUNQA distributed-QC executor.

    Attributes:
        n_qpus: Number of vQPUs to raise for the run (Phase 0 target: N≈4).
            Requested shots must be evenly divisible by n_qpus.
        walltime: Slurm walltime for the vQPU allocation, ``HH:MM:SS``.
        simulator: CUNQA backend simulator name (``"Aer"``, per README).
        co_located: Whether vQPUs share a node (CUNQA README default: True).
        classical_comm: Raise vQPUs with classical inter-vQPU communication.
        quantum_comm: Raise vQPUs with quantum inter-vQPU communication.
        mem_per_qpu_gb: Memory (GB) requested per vQPU, forwarded to
            qraise's ``mem_per_qpu`` (confirmed GB units — see module
            docstring). Required for ``EnableMemoryBasedScheduling`` to
            have anything to enforce. Default is a conservative guess for a
            handful of qubits, not CESGA's 15GB default — right-size after
            observing real usage on a live cluster.
        startup_timeout_s: Max seconds to wait for all n_qpus vQPUs to
            register after ``qraise`` before raising TimeoutError.
        poll_interval_s: Seconds between readiness polls. Clamped so the
            final poll never sleeps past startup_timeout_s.
        seed: Reserved for Phase 1 (deterministic seeding). NOT wired into
            ``run()`` yet — CUNQA's run() has no documented seed kwarg.
    """

    n_qpus: int = 4
    walltime: str = "00:10:00"
    simulator: str = "Aer"
    co_located: bool = True
    classical_comm: bool = False
    quantum_comm: bool = False
    mem_per_qpu_gb: int = 4
    startup_timeout_s: float = 60.0
    poll_interval_s: float = 1.0
    seed: int = 0


class CUNQAExecutor(BaseExecutor):
    """Runs circuits distributed across CUNQA vQPUs on a Slurm cluster.

    Shot-parallel semantics: the requested ``shots`` are split evenly across
    ``n_qpus`` vQPUs (matching the CUNQA paper's no-comm scaling benchmark),
    not run independently per vQPU. Counts are merged across all vQPUs.
    """

    def __init__(self, config: CUNQAExecutorConfig, client: CunqaClient | None = None) -> None:
        self._config = config
        self._client = client if client is not None else _CunqaModuleClient()

    async def execute(self, circuit: "Circuit", shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        circuit = self._validate_circuit(circuit)
        cfg = self._config

        if shots % cfg.n_qpus != 0:
            raise ValueError(
                f"shots ({shots}) must be evenly divisible by n_qpus ({cfg.n_qpus}) "
                f"for Phase 0's shot-parallel split — remainder handling is not yet built"
            )
        shots_per_qpu = shots // cfg.n_qpus

        qiskit_circuit = await asyncio.to_thread(circuit.to_qiskit)
        qiskit_circuit = add_measure_all(qiskit_circuit)

        family = await asyncio.to_thread(
            self._client.qraise,
            cfg.n_qpus,
            cfg.walltime,
            simulator=cfg.simulator,
            co_located=cfg.co_located,
            classical_comm=cfg.classical_comm,
            quantum_comm=cfg.quantum_comm,
            mem_per_qpu=cfg.mem_per_qpu_gb,
        )
        start = time.monotonic()
        try:
            qpus = await self._wait_for_qpus_ready(family, cfg)
            circuits = [qiskit_circuit.copy() for _ in qpus]
            jobs = await asyncio.to_thread(self._client.run, circuits, qpus, shots=shots_per_qpu)
            results = await asyncio.to_thread(self._client.gather, jobs)
        finally:
            await self._teardown(family)
        elapsed_ms = (time.monotonic() - start) * 1000

        merged_counts: Counter[str] = Counter()
        per_qpu_counts: list[dict[str, int]] = []
        for result in results:
            merged_counts.update(result.counts)
            per_qpu_counts.append(dict(result.counts))
        counts = dict(merged_counts)

        total = sum(counts.values())
        if total != shots:
            raise RuntimeError(
                f"shot conservation violated: expected {shots} total shots "
                f"({cfg.n_qpus} vQPUs x {shots_per_qpu} each), got {total} (counts={counts})"
            )

        return ExecutionResult(
            counts=counts,
            backend=self.name,
            execution_time_ms=elapsed_ms,
            shots=shots,
            raw_result=results,
            metadata={
                "n_qpus": cfg.n_qpus,
                "family": family,
                "shots_per_qpu": shots_per_qpu,
                "per_qpu_counts": per_qpu_counts,
            },
        )

    async def _teardown(self, family: str) -> None:
        # Teardown must happen on the failure AND cancellation path.
        #
        # In plain asyncio a single cancel() delivers CancelledError once;
        # a bare `await qdrop(...)` here would normally still complete
        # fine. shield exists for the narrower, real case that matters in
        # production: a SECOND cancellation arriving during cleanup itself
        # (Temporal activity timeouts and asyncio.wait_for-style wrappers
        # do exactly this). shield protects the qdrop call from that
        # repeated cancellation, but only guarantees "not cancelled
        # *again*", not "definitely awaited to completion" — so on a
        # CancelledError raised THROUGH the shield, we must still wait for
        # the still-running drop before re-raising, rather than abandoning
        # it as an orphaned, unretrieved-exception task. That follow-up
        # wait is ITSELF shielded too, for the same reason.
        #
        # Known, accepted residual gap: a THIRD cancellation landing on
        # that second shielded await still isn't caught by `except
        # Exception` there — CancelledError would propagate from it and
        # leave `drop` orphaned (though still shielded and running to
        # completion). Three nested cancellations during one teardown is
        # past the point of worthwhile hardening.
        #
        # Also: CancelledError is a BaseException (since Python 3.8), not
        # an Exception — a bare `except Exception` does NOT catch it,
        # which is why it needs its own clause, separate from the generic
        # log-and-swallow handler below.
        drop = asyncio.ensure_future(asyncio.to_thread(self._client.qdrop, family))
        try:
            await asyncio.shield(drop)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(drop)
            except Exception:
                logger.exception(
                    "qdrop failed for family %s; allocation may be stranded", family
                )
            raise
        except Exception:
            logger.exception(
                "qdrop failed for family %s; allocation may be stranded", family
            )

    async def _wait_for_qpus_ready(self, family: str, cfg: CUNQAExecutorConfig) -> list[Any]:
        deadline = time.monotonic() + cfg.startup_timeout_s
        qpus: list[Any] = []
        while time.monotonic() < deadline:
            qpus = await asyncio.to_thread(self._client.get_QPUs, cfg.co_located, family)
            if len(qpus) >= cfg.n_qpus:
                return qpus
            remaining = deadline - time.monotonic()
            await asyncio.sleep(min(cfg.poll_interval_s, max(0.0, remaining)))
        raise TimeoutError(
            f"{len(qpus)}/{cfg.n_qpus} vQPUs registered within {cfg.startup_timeout_s}s "
            f"for family {family!r}"
        )

    async def get_status(self) -> DeviceStatus:
        return DeviceStatus.always_online()
