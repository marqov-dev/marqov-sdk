"""E2e workflow isolation test — SP2 Task 3 (criterion 3).

ONE bell-state @workflow end-to-end through the Temporal test framework:

  WorkflowDispatch._prepare_workflow_input()
    → JobWorkflow (real temporalio Worker)
      → prepare_node_inputs activity (real, in-process)
      → execute_task activity (real, each task body in a scrubbed subprocess)

Keystone assertions (criterion 3):
  a. Results flow back correctly — bell-state tasks produce expected counts.
  b. cloudpickle.loads spy: ZERO in-process calls in the worker/activity process
     during the entire e2e run.  (loads happens only inside the child subprocesses.)
  c. Both task subprocesses' envs lack arbitrary host secrets (HOST_* vars).

Harness measurement:
  - Uses WorkflowEnvironment.start_time_skipping() — temporalio's in-process
    ephemeral test server.  The binary is already cached at
    /var/folders/.../temporal-test-server-sdk-python-1.27.2 (downloaded
    previously; ~61 MB arm64 binary).
  - No Docker, no real Temporal cluster needed.
  - env start < 5 s in CI (measured: sub-second on warm machine).
  - Verdict: LIGHT harness.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cloudpickle
import pytest

from marqov.workflows.activity import execute_task, prepare_node_inputs
from marqov.workflows.temporal_workflow import JobWorkflow
from marqov.workflows.decorators import task, workflow


# ---------------------------------------------------------------------------
# Bell-state tasks (simple noiseless simulation — no hardware required)
# ---------------------------------------------------------------------------
# We use numpy for the simulation so the result is a dict (non-primitive),
# forcing the cloudpickle envelope path through the subprocess boundary.


@task
def run_bell_circuit() -> dict[str, int]:
    """Simulate a bell-state circuit and return measurement counts.

    Returns {'00': N, '11': N} with equal probability.  We use a fixed seed
    so the test is deterministic.
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)
    n_shots = 100
    # Bell state |Φ+⟩: perfect 50/50 between 00 and 11.
    outcomes = rng.choice([0, 1], size=n_shots)  # 0 → '00', 1 → '11'
    counts: dict[str, int] = {"00": int((outcomes == 0).sum()), "11": int((outcomes == 1).sum())}
    return counts


@task
def validate_counts(counts: dict[str, int]) -> dict[str, Any]:
    """Validate that counts are consistent with a bell state.

    Checks:
    - Only '00' and '11' outcomes (no '01' or '10').
    - Total shots = 100.
    - Both '00' and '11' have > 0 counts.
    """
    valid_keys = {"00", "11"}
    unexpected = set(counts.keys()) - valid_keys
    total = sum(counts.values())
    ok = (
        len(unexpected) == 0
        and total == 100
        and counts.get("00", 0) > 0
        and counts.get("11", 0) > 0
    )
    return {
        "valid": ok,
        "counts": counts,
        "total_shots": total,
        "unexpected_outcomes": sorted(unexpected),
    }


@workflow
def bell_state_workflow() -> Any:
    """Bell-state workflow: simulate → validate."""
    counts = run_bell_circuit()
    result = validate_counts(counts)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker(client: Any) -> Any:
    """Build a Worker registered with the real activities and workflow.

    We use UnsandboxedWorkflowRunner to avoid the Temporal sandbox's proxy
    restrictions firing on modules (e.g. numpy) that were already imported in
    the test process.  The sandbox is a production concern; for the spike
    measurement the unsandboxed runner gives a clean read of the two subprocess
    boundaries without sandbox noise.
    """
    from temporalio.worker import Worker, UnsandboxedWorkflowRunner

    return Worker(
        client,
        task_queue="marqov-workflows",
        workflows=[JobWorkflow],
        activities=[execute_task, prepare_node_inputs],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )


async def _run_bell_state_e2e(client: Any) -> dict[str, Any]:
    """Drive the full e2e run and return the final result dict."""
    from marqov.workflows.decorators import WorkflowDispatch

    dispatch: WorkflowDispatch = bell_state_workflow()
    workflow_input = dispatch._prepare_workflow_input()

    import uuid
    workflow_id = f"bell-e2e-{uuid.uuid4().hex[:8]}"

    async with _make_worker(client):
        handle = await client.start_workflow(
            JobWorkflow.run,
            args=[workflow_input],
            id=workflow_id,
            task_queue="marqov-workflows",
        )
        result_json: str = await handle.result()

    parsed = json.loads(result_json)
    # Unwrap the enriched format: {"result": ..., "_workflow_metadata": ...}
    if isinstance(parsed, dict) and "result" in parsed:
        return parsed["result"]
    return parsed


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_bell_state_no_privileged_loads() -> None:
    """E2e bell-state workflow through both boundaries — no privileged loads.

    Criterion 3:
      a. results flow back (bell-state → validate passes).
      b. ZERO cloudpickle.loads calls in the activity/worker process.
      c. child envs lack arbitrary host secrets (HOST_* vars).

    Harness weight: WorkflowEnvironment.start_time_skipping() — light
    (uses the cached temporal-test-server binary; no Docker required).
    """
    from temporalio.testing import WorkflowEnvironment

    # --- spy setup ---
    loads_calls: list[bytes] = []
    real_loads = cloudpickle.loads

    def spy_loads(data: bytes, *args: Any, **kwargs: Any) -> Any:
        loads_calls.append(data[:16])  # record prefix for debugging
        return real_loads(data, *args, **kwargs)

    # Child env capture (assertion c)
    child_envs_captured: list[dict[str, str]] = []

    from marqov.workflows import _child_env as ce

    original_build = ce.build_child_env

    def capturing_build(workdir: Path, provider_env: dict[str, str] | None = None) -> dict[str, str]:
        env = original_build(workdir, provider_env=provider_env)
        child_envs_captured.append(dict(env))
        return env

    # Inject fake host secrets into PARENT env to verify they don't leak.
    fake_secrets = {
        "HOST_SECRET_TOKEN": "fake-e2e-secret",
        "HOST_API_KEY": "fake-e2e-api-key",
        "HOST_NAMESPACE": "example",
    }

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client = env.client

        with patch.object(cloudpickle, "loads", side_effect=spy_loads):
            with patch.dict(os.environ, fake_secrets):
                with patch("marqov.workflows.activity.build_child_env", side_effect=capturing_build):
                    result = await _run_bell_state_e2e(client)

    # --- assertion a: results correct ---
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}: {result!r}"
    assert result.get("valid") is True, (
        f"Bell-state validation failed: {result!r}"
    )
    counts = result.get("counts", {})
    assert "00" in counts and "11" in counts, f"Unexpected counts keys: {counts}"
    assert counts["00"] + counts["11"] == 100, f"Wrong total shots: {counts}"
    assert counts["00"] > 0 and counts["11"] > 0, f"Zero counts in one outcome: {counts}"

    # --- assertion b: ZERO cloudpickle.loads in the worker/activity process ---
    assert len(loads_calls) == 0, (
        f"cloudpickle.loads was called {len(loads_calls)} time(s) IN the worker process "
        f"during the e2e run.  No-privileged-loads invariant VIOLATED.\n"
        f"First call data prefix: {loads_calls[0] if loads_calls else 'n/a'}"
    )

    # --- assertion c: child envs lack host secrets ---
    # Two task nodes → two execute_task calls → two child envs captured.
    assert len(child_envs_captured) >= 1, (
        "No child envs were captured — build_child_env patch not exercised?"
    )
    for i, child_env in enumerate(child_envs_captured):
        assert "HOST_SECRET_TOKEN" not in child_env, (
            f"HOST_SECRET_TOKEN leaked into child env #{i}!"
        )
        host_keys = [k for k in child_env if k.startswith("HOST_")]
        assert host_keys == [], (
            f"HOST_* keys leaked into child env #{i}: {host_keys}"
        )

    # Sanity: we captured exactly 2 child envs (one per task node).
    assert len(child_envs_captured) == 2, (
        f"Expected 2 child env captures (2 task nodes), got {len(child_envs_captured)}"
    )
