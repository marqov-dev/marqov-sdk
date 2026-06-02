"""Tests for pre-flight validation utility."""

from __future__ import annotations

from marqov.experiments.preflight import preflight


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
