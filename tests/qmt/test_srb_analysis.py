"""Tests for SRB decay fitting and NoiseProfile generation."""

from __future__ import annotations

import numpy as np
import pytest

from marqov.qmt.characterization.srb import SRBConfig, SRBResult
from marqov.qmt.characterization.srb_analysis import (
    RBFitResult,
    extract_cross_talk,
    fit_rb_decay,
    build_noise_profile_from_srb,
)
from marqov.qmt.models import DeviceModality


def _make_srb_result(
    target: list[int],
    neighbor: list[int],
    iso_survival: dict[int, float],
    sim_survival: dict[int, float],
) -> SRBResult:
    """Helper to build SRBResult with empty raw dicts."""
    return SRBResult(
        config=SRBConfig(target_qubits=target, neighbor_qubits=neighbor),
        isolated_survival=iso_survival,
        simultaneous_survival=sim_survival,
        isolated_raw={m: [v] for m, v in iso_survival.items()},
        simultaneous_raw={m: [v] for m, v in sim_survival.items()},
    )


class TestFitRBDecay:
    def test_perfect_decay(self) -> None:
        p_true = 0.98
        lengths = [1, 2, 4, 8, 16, 32, 64, 128]
        survival = {m: 0.5 * (p_true ** m) + 0.5 for m in lengths}
        fit = fit_rb_decay(survival)
        assert fit.p == pytest.approx(p_true, abs=0.001)
        assert fit.error_per_clifford == pytest.approx((1 - p_true) / 2, abs=0.001)
        assert fit.r_squared > 0.99

    def test_noiseless_decay(self) -> None:
        lengths = [1, 2, 4, 8, 16, 32]
        survival = {m: 1.0 for m in lengths}
        fit = fit_rb_decay(survival)
        assert fit.p == pytest.approx(1.0, abs=0.01)
        assert fit.error_per_clifford == pytest.approx(0.0, abs=0.01)

    def test_auto_truncation(self) -> None:
        p_true = 0.95
        lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        rng = np.random.default_rng(42)
        survival = {}
        for m in lengths:
            val = 0.5 * (p_true ** m) + 0.5
            if val < 0.52:
                val = 0.50 + rng.uniform(-0.02, 0.02)
            survival[m] = val
        fit = fit_rb_decay(survival)
        assert fit.truncated_at is not None
        assert fit.p == pytest.approx(p_true, abs=0.02)

    def test_truncation_oscillating_near_floor(self) -> None:
        """Points that oscillate around noise_floor should truncate at first below-floor point."""
        survival = {
            1: 0.95,
            2: 0.90,
            4: 0.80,
            8: 0.60,
            16: 0.53,   # above floor (0.52)
            32: 0.48,   # below floor — truncation point
            64: 0.55,   # back above, but should be excluded
            128: 0.45,  # below again
        }
        fit = fit_rb_decay(survival)
        assert fit.truncated_at == 32
        assert fit.r_squared > 0.8

    def test_truncation_minimum_points(self) -> None:
        """If truncation would leave < 3 points, return degenerate fit."""
        survival = {
            1: 0.95,
            2: 0.90,
            4: 0.48,   # below floor at index 2 — truncation leaves only 2 points
            8: 0.45,
        }
        fit = fit_rb_decay(survival)
        assert fit.p == pytest.approx(1.0, abs=0.01)
        assert fit.r_squared == 0.0

    def test_truncation_non_contiguous_below_floor(self) -> None:
        """Only the first below-floor point triggers truncation."""
        p_true = 0.96
        survival = {
            1: 0.5 * (p_true ** 1) + 0.5,
            2: 0.5 * (p_true ** 2) + 0.5,
            4: 0.5 * (p_true ** 4) + 0.5,
            8: 0.5 * (p_true ** 8) + 0.5,
            16: 0.5 * (p_true ** 16) + 0.5,
            32: 0.5 * (p_true ** 32) + 0.5,
            64: 0.5 * (p_true ** 64) + 0.5,
            128: 0.5 * (p_true ** 128) + 0.5,
        }
        fit = fit_rb_decay(survival)
        assert fit.truncated_at == 128
        assert fit.p == pytest.approx(p_true, abs=0.02)


class TestExtractCrossTalk:
    def test_cross_talk_from_decay_difference(self) -> None:
        lengths = [1, 2, 4, 8, 16, 32]
        p_iso = 0.98
        p_sim = 0.96
        iso = {m: 0.5 * (p_iso ** m) + 0.5 for m in lengths}
        sim = {m: 0.5 * (p_sim ** m) + 0.5 for m in lengths}
        result = _make_srb_result([0], [2], iso, sim)
        ct = extract_cross_talk(result)
        assert ct["isolated_error_per_clifford"] == pytest.approx(0.01, abs=0.002)
        assert ct["simultaneous_error_per_clifford"] == pytest.approx(0.02, abs=0.002)
        assert ct["cross_talk_delta"] > 0

    def test_no_cross_talk_when_equal(self) -> None:
        lengths = [1, 2, 4, 8, 16, 32]
        p = 0.98
        surv = {m: 0.5 * (p ** m) + 0.5 for m in lengths}
        result = _make_srb_result([0], [2], surv, surv)
        ct = extract_cross_talk(result)
        assert ct["cross_talk_delta"] == pytest.approx(0.0, abs=0.002)


class TestBuildNoiseProfileFromSRB:
    def test_produces_noise_profile(self) -> None:
        lengths = [1, 2, 4, 8, 16, 32]
        p_iso = 0.98
        p_sim = 0.96
        iso = {m: 0.5 * (p_iso ** m) + 0.5 for m in lengths}
        sim = {m: 0.5 * (p_sim ** m) + 0.5 for m in lengths}
        result = _make_srb_result([0], [2], iso, sim)
        profile = build_noise_profile_from_srb(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=4,
            srb_results=[result],
        )
        assert profile.device_name == "test"
        assert profile.cross_talk_between({0}, {2}) > 0
