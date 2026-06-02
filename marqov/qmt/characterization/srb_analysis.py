"""SRB decay curve fitting, error-per-Clifford extraction, and NoiseProfile generation.

Fits the standard RB exponential decay model f(m) = A * p^m + B to survival
probability data, extracts error per Clifford r = (1 - p) / 2, and builds
NoiseProfile objects from isolated vs. simultaneous comparison.

References:
    Magesan et al. 2011, arXiv:1109.6887
    Gambetta et al. 2012, arXiv:1204.6308
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from marqov.qmt.characterization.srb import SRBResult
from marqov.qmt.models import DeviceModality, NoiseProfile

logger = logging.getLogger(__name__)


@dataclass
class RBFitResult:
    """Result of fitting an RB exponential decay curve.

    Attributes:
        p: Depolarizing parameter (0 < p <= 1).
        A: Amplitude of the exponential decay.
        B: Offset / asymptote.
        error_per_clifford: r = (1 - p) / 2.
        r_squared: Coefficient of determination for the fit.
        truncated_at: Sequence length where auto-truncation was applied, or None.
    """

    p: float
    A: float
    B: float
    error_per_clifford: float
    r_squared: float
    truncated_at: int | None = None


def _rb_model(m: np.ndarray, A: float, p: float, B: float) -> np.ndarray:
    """Standard RB decay model: f(m) = A * p^m + B."""
    return A * p ** m + B


def _compute_r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 (coefficient of determination)."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_rb_decay(
    survival: dict[int, float],
    noise_floor: float = 0.52,
) -> RBFitResult:
    """Fit an exponential decay to RB survival probability data.

    Args:
        survival: Mapping of sequence length -> average survival probability.
        noise_floor: Survival values below this are considered noise. Points
            at and beyond the first below-floor value are truncated if we
            still have >= 3 points remaining.

    Returns:
        RBFitResult with fitted parameters and goodness-of-fit.
    """
    # Sort by length
    sorted_items = sorted(survival.items())
    lengths = np.array([m for m, _ in sorted_items], dtype=float)
    probs = np.array([p for _, p in sorted_items], dtype=float)

    # Auto-truncate: find first point below noise_floor
    truncated_at: int | None = None
    below = np.where(probs < noise_floor)[0]
    if len(below) > 0 and below[0] >= 2:
        truncate_idx = below[0]
        truncated_at = int(lengths[truncate_idx])
        lengths = lengths[:truncate_idx]
        probs = probs[:truncate_idx]

    # Need at least 3 data points for a 3-parameter fit
    if len(lengths) < 3:
        return RBFitResult(
            p=1.0, A=0.5, B=0.5,
            error_per_clifford=0.0,
            r_squared=0.0,
            truncated_at=truncated_at,
        )

    # Weighted least-squares: binomial variance proxy with n=100 shots
    n_shots = 100
    variance = probs * (1 - probs) / n_shots
    # Avoid zero variance (for p=1.0 or p=0.0 points)
    variance = np.maximum(variance, 1e-8)
    sigma = np.sqrt(variance)

    try:
        popt, _ = curve_fit(
            _rb_model,
            lengths,
            probs,
            p0=[0.5, 0.99, 0.5],
            bounds=([0, 0, 0], [1, 1, 1]),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=10000,
        )
        A_fit, p_fit, B_fit = popt
        y_pred = _rb_model(lengths, *popt)
        r_squared = _compute_r_squared(probs, y_pred)
    except RuntimeError:
        logger.warning("RB decay fit failed, returning degenerate result")
        return RBFitResult(
            p=1.0, A=0.5, B=0.5,
            error_per_clifford=0.0,
            r_squared=0.0,
            truncated_at=truncated_at,
        )

    error_per_clifford = (1 - p_fit) / 2

    return RBFitResult(
        p=float(p_fit),
        A=float(A_fit),
        B=float(B_fit),
        error_per_clifford=float(error_per_clifford),
        r_squared=float(r_squared),
        truncated_at=truncated_at,
    )


def extract_cross_talk(result: SRBResult) -> dict:
    """Fit isolated and simultaneous decay curves and compare.

    Args:
        result: An SRBResult containing both isolated and simultaneous data.

    Returns:
        Dict with keys: isolated_p, isolated_error_per_clifford, isolated_r_squared,
        simultaneous_p, simultaneous_error_per_clifford, simultaneous_r_squared,
        cross_talk_delta (simultaneous error - isolated error).
    """
    iso_fit = fit_rb_decay(result.isolated_survival)
    sim_fit = fit_rb_decay(result.simultaneous_survival)

    return {
        "isolated_p": iso_fit.p,
        "isolated_error_per_clifford": iso_fit.error_per_clifford,
        "isolated_r_squared": iso_fit.r_squared,
        "simultaneous_p": sim_fit.p,
        "simultaneous_error_per_clifford": sim_fit.error_per_clifford,
        "simultaneous_r_squared": sim_fit.r_squared,
        "cross_talk_delta": sim_fit.error_per_clifford - iso_fit.error_per_clifford,
    }


def build_noise_profile_from_srb(
    device_name: str,
    modality: DeviceModality,
    num_qubits: int,
    srb_results: list[SRBResult],
) -> NoiseProfile:
    """Build a NoiseProfile from a collection of SRB results.

    For each SRBResult, the isolated error per Clifford is assigned to the
    target qubits' error rates, and the cross-talk delta is written into
    the cross-talk matrix between target and neighbor qubits.

    Args:
        device_name: Name of the device.
        modality: Device modality (trapped ion, neutral atom, etc.).
        num_qubits: Total number of qubits on the device.
        srb_results: List of SRB results (one per target/neighbor pair).

    Returns:
        A populated NoiseProfile.
    """
    profile = NoiseProfile(
        device_name=device_name,
        modality=modality,
        num_qubits=num_qubits,
    )

    for result in srb_results:
        ct = extract_cross_talk(result)

        # Set qubit error rates from isolated measurement
        for q in result.config.target_qubits:
            if 0 <= q < num_qubits:
                profile.qubit_error_rates[q] = ct["isolated_error_per_clifford"]

        # Set cross-talk matrix entries between target and neighbor qubits
        delta = ct["cross_talk_delta"]
        for t in result.config.target_qubits:
            for n in result.config.neighbor_qubits:
                if 0 <= t < num_qubits and 0 <= n < num_qubits:
                    profile.cross_talk_matrix[t, n] = delta

    return profile
