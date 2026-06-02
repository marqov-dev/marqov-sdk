"""Temporal workflow for job execution.

IMPORTANT: This module has NO marqov imports whatsoever.

Temporal's workflow sandbox validates all imports at class definition time.
By keeping this module completely free of marqov package imports, we avoid
sandbox validation errors from sympy/quantumflow.

All computation happens in activities (see activity.py), which are not
subject to the same sandbox restrictions. Activities are referenced by
string name, not by importing the function.

Architecture:
    WorkflowDispatch.run() → serializes graph to JSON primitives
                           → pre-computes execution levels
                           → calls JobWorkflow with pure JSON

    JobWorkflow            → orchestrates by node IDs only
                           → schedules activities BY STRING NAME
                           → passes results as JSON between levels

    execute_task           → (in activity.py) does actual computation
                           → imports marqov safely in activity context
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy


@workflow.defn
class JobWorkflow:
    """Generic workflow that executes a job's task graph.

    This workflow receives pre-computed execution data as JSON and
    orchestrates task execution without importing quantum libraries.

    Activities are referenced by string name to avoid any import chain
    that could trigger sandbox validation errors.

    The workflow only works with:
    - Node IDs (strings)
    - Execution levels (list of lists of node IDs)
    - Serialized node data (JSON strings)
    - Serialized results (JSON strings)
    """

    @workflow.run
    async def run(self, workflow_input: dict[str, Any]) -> str:
        """Execute the task graph with parallelization.

        Args:
            workflow_input: Dictionary containing:
                - nodes: Dict of node_id -> serialized node data
                - execution_levels: List of lists of node IDs
                - output_nodes: List of output node IDs

        Returns:
            JSON string with enriched results including workflow metadata.
        """
        nodes = workflow_input["nodes"]
        execution_levels = workflow_input["execution_levels"]
        output_nodes = workflow_input["output_nodes"]

        # Results accumulated across levels
        completed_results: dict[str, Any] = {}
        task_timeline: list[dict[str, Any]] = []

        for level_idx, level in enumerate(execution_levels):
            # Prepare and execute all nodes at this level in parallel
            tasks = []
            task_metas: list[dict[str, Any]] = []

            for node_id in level:
                node_data = nodes[node_id]

                # Resolve dependencies using completed results
                # Activity referenced by string name - no import needed!
                resolved_json = await workflow.execute_activity(
                    "prepare_node_inputs",
                    args=[
                        json.dumps(node_data),
                        json.dumps(completed_results),
                    ],
                    start_to_close_timeout=timedelta(seconds=30),
                )

                resolved = json.loads(resolved_json)

                # Schedule task execution
                retry_policy = RetryPolicy(
                    maximum_attempts=node_data.get("retries", 0) + 1,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=60),
                    backoff_coefficient=2.0,
                )

                meta: dict[str, Any] = {
                    "node_id": node_id,
                    "func_name": node_data.get("func_name", "unknown"),
                    "level": level_idx,
                }

                # Activity referenced by string name - no import needed!
                activity_coro = workflow.execute_activity(
                    "execute_task",
                    args=[
                        resolved["node_id"],
                        resolved["func_ref"],
                        json.dumps(resolved["args"]),
                        json.dumps(resolved["kwargs"]),
                    ],
                    start_to_close_timeout=timedelta(
                        seconds=node_data.get("timeout_seconds", 300)
                    ),
                    # 60s timeout with 10s send interval. Temporal throttles
                    # heartbeat forwarding to 80% of heartbeat_timeout (48s).
                    # 30s was too tight — a single missed forward would exceed
                    # the timeout and falsely mark the activity as failed.
                    heartbeat_timeout=timedelta(seconds=60),
                    retry_policy=retry_policy,
                )
                tasks.append(self._timed_execute(activity_coro, meta))
                task_metas.append(meta)

            # Wait for all activities at this level (parallel execution)
            level_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results, re-raise any exceptions
            for i, result_or_exc in enumerate(level_results):
                if isinstance(result_or_exc, BaseException):
                    task_timeline.extend(task_metas)
                    raise result_or_exc
                result = json.loads(result_or_exc)
                completed_results[result["node_id"]] = result["result"]

            task_timeline.extend(task_metas)

        # Return output node results
        if len(output_nodes) == 1:
            output = completed_results[output_nodes[0]]
        elif len(output_nodes) > 1:
            output = {nid: completed_results[nid] for nid in output_nodes}
        else:
            output = completed_results

        enriched: dict[str, Any] = {
            "result": output,
            "_workflow_metadata": {
                "execution_graph": {
                    "nodes": {
                        nid: {
                            "func_name": nodes[nid].get("func_name", "unknown"),
                            "dependencies": nodes[nid].get("dependencies", []),
                        }
                        for nid in nodes
                    },
                    "execution_levels": execution_levels,
                    "output_nodes": output_nodes,
                },
                "task_timeline": task_timeline,
                "total_tasks": len(nodes),
                "total_levels": len(execution_levels),
            },
        }

        return json.dumps(enriched)

    async def _timed_execute(
        self, task_coro: Any, meta: dict[str, Any]
    ) -> Any:
        """Wrap a task coroutine with start/end timing using Temporal's deterministic clock."""
        meta["started_at"] = workflow.now().isoformat()
        try:
            result = await task_coro
            meta["completed_at"] = workflow.now().isoformat()
            meta["status"] = "completed"
            return result
        except Exception as e:
            meta["completed_at"] = workflow.now().isoformat()
            meta["status"] = "failed"
            meta["error"] = str(e)
            raise
