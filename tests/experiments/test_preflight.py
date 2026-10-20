"""Tests for pre-flight validation utility."""

from __future__ import annotations

import importlib
import inspect
import logging

from marqov.experiments.preflight import preflight

# ``marqov.experiments.preflight`` the attribute is the function; get the module.
preflight_module = importlib.import_module("marqov.experiments.preflight")


class TestPreflightPasses:
    def test_all_fields_present(self) -> None:
        result = preflight(
            experiment_config={"device": "rigetti"},
            output_sample={
                "isolated_survival": {},
                "simultaneous_survival": {},
                "isolated_raw": {},
                "cross_talk": {},
                "result_quality": "good",
                "task_arns": [],
            },
            estimated_cost=340,
            estimated_circuits=1200,
        )
        assert result.passed
        assert result.schema_valid
        assert len(result.warnings) == 0
        assert result.estimated_cost == 340
        assert result.estimated_circuits == 1200

    def test_output_fields_captured(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={
                "counts": {},
                "result_quality": "good",
                "provenance": {},
            },
            estimated_cost=50,
            estimated_circuits=100,
        )
        assert "counts" in result.output_fields
        assert "result_quality" in result.output_fields
        assert "provenance" in result.output_fields


class TestPreflightWarnings:
    def test_warns_no_raw_data(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={
                "cross_talk_delta": 0.01,
                "error_per_clifford": 0.005,
                "result_quality": "good",
                "task_arns": [],
            },
            estimated_cost=100,
            estimated_circuits=50,
        )
        assert not result.passed
        assert any("raw data" in w.lower() for w in result.warnings)

    def test_warns_no_quality(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={
                "isolated_survival": {},
                "counts": {},
                "task_arns": [],
            },
            estimated_cost=100,
            estimated_circuits=50,
        )
        assert any("quality" in w.lower() for w in result.warnings)

    def test_warns_no_cost_estimate(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={
                "counts": {},
                "result_quality": "good",
                "provenance": {},
            },
            estimated_cost=None,
            estimated_circuits=100,
        )
        assert any("cost estimate" in w.lower() for w in result.warnings)

    def test_warns_no_circuit_count(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={
                "counts": {},
                "result_quality": "good",
                "provenance": {},
            },
            estimated_cost=100,
            estimated_circuits=None,
        )
        assert any("circuit count" in w.lower() for w in result.warnings)

    def test_warns_no_provenance(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={
                "counts": {},
                "result_quality": "good",
            },
            estimated_cost=100,
            estimated_circuits=50,
        )
        assert any("provenance" in w.lower() or "task arn" in w.lower() for w in result.warnings)

    def test_empty_output_gets_all_warnings(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={},
        )
        assert not result.passed
        # Should warn about: raw data, quality, cost, circuits, provenance
        assert len(result.warnings) >= 4


class TestPreflightDefaults:
    def test_default_cost_zero(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={"counts": {}, "result_quality": "good", "provenance": {}},
            estimated_cost=None,
        )
        assert result.estimated_cost == 0

    def test_default_circuits_zero(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={"counts": {}, "result_quality": "good", "provenance": {}},
            estimated_circuits=None,
        )
        assert result.estimated_circuits == 0

    def test_budget_check_none_by_default(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample={"counts": {}, "result_quality": "good", "provenance": {}},
            estimated_cost=100,
            estimated_circuits=50,
        )
        assert result.budget_check is None


_GOOD_SAMPLE = {"counts": {"00": 10}, "result_quality": "good", "task_arns": ["a"]}


class TestPreflightReportsOnlyWhatItDoes:
    """No outcome is reported for a check the function never performs."""

    def test_dry_run_not_attempted_is_none(self) -> None:
        result = preflight(
            experiment_config={"backend": "aws/sv1", "shots": 1000},
            output_sample=_GOOD_SAMPLE,
            estimated_cost=50_000,
            estimated_circuits=1_000_000,
        )
        assert result.dry_run_passed is None
        assert result.budget_check is None

    def test_no_hardcoded_true_dry_run(self) -> None:
        assert "dry_run_passed=True" not in inspect.getsource(preflight_module)

    def test_missing_cost_warning_promises_no_budget_validation(self) -> None:
        result = preflight(
            experiment_config={},
            output_sample=_GOOD_SAMPLE,
            estimated_cost=None,
            estimated_circuits=10,
        )
        assert result.warnings == [
            "WARNING: No cost estimate provided. "
            "The cost is recorded as 0 in the result. "
            "This function does not check cost against a budget."
        ]

    def test_budget_check_documented_as_platform_populated(self) -> None:
        doc = inspect.getdoc(preflight_module.PreflightResult) or ""
        assert "budget_check" in doc
        assert "platform-side caller" in doc
        assert "never populated by preflight()" in doc

    def test_config_keys_logged_on_success_path(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=preflight_module.logger.name):
            result = preflight(
                experiment_config={"backend": "aws/sv1", "shots": 1000},
                output_sample=_GOOD_SAMPLE,
                estimated_cost=1,
                estimated_circuits=1,
            )
        assert result.passed
        text = caplog.text
        assert "backend" in text and "shots" in text
        assert "aws/sv1" not in text

    def test_config_keys_logged_on_warning_path(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=preflight_module.logger.name):
            result = preflight(
                experiment_config={"backend": "aws/sv1", "shots": 1000},
                output_sample={},
            )
        assert not result.passed
        assert "backend" in caplog.text and "shots" in caplog.text

    def test_config_values_never_logged(self, caplog) -> None:
        for sample in (_GOOD_SAMPLE, {}):
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger=preflight_module.logger.name):
                preflight(
                    experiment_config={"ibm_token": "secret"},
                    output_sample=sample,
                    estimated_cost=1,
                    estimated_circuits=1,
                )
            assert "ibm_token" in caplog.text
            assert "secret" not in caplog.text
