"""Pre-flight validation for expensive experiment runs.

Client-side advisory layer. Warns but does not block.
Use --force to bypass warnings.

Server-side budget enforcement in the Temporal worker is mandatory
and cannot be bypassed via --force. This utility helps researchers
catch issues before submission — it does not replace backend checks.

Usage in experiment scripts:

    from marqov.experiments.preflight import preflight

    sample_output = {
        "cross_talk": {...},
        "isolated_survival": {...},
        "result_quality": "good",
    }
    check = preflight(
        experiment_config=config.__dict__,
        output_sample=sample_output,
        estimated_cost=340,
        estimated_circuits=1200,
    )
    if check.warnings and not force:
        for w in check.warnings:
            print(w)
        confirm = input("Continue? [y/N] ")
        if confirm.lower() != "y":
            sys.exit(1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of a pre-flight validation check."""

    schema_valid: bool
    estimated_cost: float
    estimated_circuits: int
    budget_check: dict | None
    dry_run_passed: bool
    warnings: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no warnings were raised."""
        return len(self.warnings) == 0


def preflight(
    experiment_config: dict,
    output_sample: dict,
    estimated_cost: float | None = None,
    estimated_circuits: int | None = None,
) -> PreflightResult:
    """Run pre-flight checks before an expensive experiment.

    Validates that the experiment output will preserve raw data,
    include quality assessment, and estimates are provided.

    Args:
        experiment_config: The experiment parameters (for logging).
        output_sample: A sample of what the result JSONB will contain.
            Used to validate that raw data fields are present.
        estimated_cost: Estimated cost in USD.
        estimated_circuits: Number of circuits to submit.

    Returns:
        PreflightResult with warnings and validation status.
    """
    warnings: list[str] = []
    output_fields = list(output_sample.keys()) if output_sample else []

    # Check 1: Raw data fields present
    raw_keywords = ["raw", "counts", "survival", "shots", "measurements"]
    raw_data_fields = [
        f for f in output_fields
        if any(kw in f.lower() for kw in raw_keywords)
    ]
    if not raw_data_fields:
        warnings.append(
            "WARNING: No raw data fields detected in output sample. "
            "Fields checked for keywords: {}. "
            "Derived metrics cannot be recomputed without raw measurements. "
            "Consider adding fields like 'isolated_raw', 'counts', or 'survival'.".format(
                ", ".join(raw_keywords)
            )
        )

    # Check 2: Quality assessment present
    quality_fields = [
        f for f in output_fields
        if any(kw in f.lower() for kw in ["quality", "result_quality"])
    ]
    if not quality_fields:
        warnings.append(
            "WARNING: No quality field in output sample. "
            "Results will be marked 'unknown' in the platform. "
            "Consider adding 'result_quality' to your output."
        )

    # Check 3: Cost estimate provided
    if estimated_cost is None:
        warnings.append(
            "WARNING: No cost estimate provided. "
            "Cannot check against team budget. "
            "Pass estimated_cost to enable budget validation."
        )

    # Check 4: Circuit count provided
    if estimated_circuits is None:
        warnings.append(
            "WARNING: No circuit count estimate provided. "
            "Cannot estimate execution time. "
            "Pass estimated_circuits for time estimation."
        )

    # Check 5: Task ARN / provenance fields
    provenance_fields = [
        f for f in output_fields
        if any(kw in f.lower() for kw in ["task_arn", "provenance", "task_id"])
    ]
    if not provenance_fields:
        warnings.append(
            "WARNING: No provenance fields detected in output sample. "
            "Task ARNs are the recovery key if data is lost. "
            "Consider adding 'task_arns' or 'provenance' to your output."
        )

    # Log results
    if warnings:
        for w in warnings:
            logger.warning(w)
        logger.warning(
            "Pre-flight: %d warning(s). Use --force to bypass.",
            len(warnings),
        )
    else:
        logger.info(
            "Pre-flight passed: %d output fields, ~$%.0f, ~%d circuits",
            len(output_fields),
            estimated_cost or 0,
            estimated_circuits or 0,
        )

    return PreflightResult(
        schema_valid=len(warnings) == 0,
        estimated_cost=estimated_cost or 0,
        estimated_circuits=estimated_circuits or 0,
        budget_check=None,  # populated when running through platform
        dry_run_passed=True,  # set by caller after local sim
        warnings=warnings,
        output_fields=output_fields,
    )
