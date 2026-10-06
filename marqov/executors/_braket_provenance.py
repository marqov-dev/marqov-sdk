"""Conservative provenance for Braket counts; never infer physical wiring."""

from collections import Counter
from numbers import Integral
from typing import Any

import numpy as np


PROTOCOL = "marqov.braket-measurement-provenance/v1"


def _integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def measurement_provenance(
    result: Any, counts: dict[str, int], requested_shots: int, *, probability_fallback: bool,
) -> dict[str, Any]:
    """Describe observed evidence without changing counts or the provider result.

    ``raw_shot_eligible`` means these counts have raw observation evidence and
    known column order. It does not certify completeness, physical mapping,
    verbatim execution, or the provider's scientific accuracy.
    """
    evidence: dict[str, Any] = {
        "protocol_version": PROTOCOL,
        "count_origin": "unknown",
        "requested_shots": requested_shots,
        "observed_shots": None,
        "provider_reported_shots": None,
        "measured_qubits": None,
        "bitstring_order": "unknown",
        "raw_shot_eligible": False,
        "ineligibility_reason": "unknown_origin",
        "physical_mapping_status": "unqualified",
    }
    try:
        reported = getattr(getattr(result, "task_metadata", None), "shots", None)
        if _integer(reported) and reported >= 0:
            evidence["provider_reported_shots"] = int(reported)
        wires = getattr(result, "measured_qubits", None)
        if isinstance(wires, (list, tuple)) and wires and all(
            _integer(wire) and wire >= 0 for wire in wires
        ) and len(set(wires)) == len(wires):
            evidence["measured_qubits"] = [int(wire) for wire in wires]

        if probability_fallback:
            origin = "probability_derived"
        elif getattr(result, "measurements_copied_from_device", None) is True:
            origin = "provider_measurements"
        elif getattr(result, "measurement_counts_copied_from_device", None) is True:
            origin = "provider_counts"
        elif getattr(result, "measurement_probabilities_copied_from_device", None) is True:
            origin = "probability_derived"
        else:
            origin = "unknown"
        evidence["count_origin"] = origin
        if origin in ("unknown", "probability_derived"):
            evidence["ineligibility_reason"] = (
                "unknown_origin" if origin == "unknown" else "probability_derived"
            )
            return evidence

        valid_counts = all(
            isinstance(key, str) and key and set(key) <= {"0", "1"}
            and _integer(value) and value >= 0
            for key, value in counts.items()
        )
        if not valid_counts:
            evidence["ineligibility_reason"] = "invalid_counts"
            return evidence

        if origin == "provider_measurements":
            rows = getattr(result, "measurements", None)
            if not isinstance(rows, np.ndarray) or rows.ndim != 2 or rows.dtype.kind not in "biu":
                evidence["ineligibility_reason"] = "invalid_measurements"
                return evidence
            if not np.all((rows == 0) | (rows == 1)):
                evidence["ineligibility_reason"] = "invalid_measurements"
                return evidence
            evidence["observed_shots"] = int(rows.shape[0])
            row_counts = Counter("".join(str(int(bit)) for bit in row) for row in rows)
            if row_counts != {key: value for key, value in counts.items() if value}:
                evidence["ineligibility_reason"] = "counts_measurements_mismatch"
                return evidence
            width = rows.shape[1]
        else:
            evidence["observed_shots"] = int(sum(counts.values()))
            width = len(next(iter(counts))) if counts else 0

        measured = evidence["measured_qubits"]
        if measured is None:
            evidence["ineligibility_reason"] = (
                "missing_measured_qubits" if wires is None else "invalid_measured_qubits"
            )
            return evidence
        if width != len(measured) or any(len(key) != width for key in counts):
            evidence["ineligibility_reason"] = "invalid_measured_qubits"
            return evidence
        evidence["bitstring_order"] = "measured_qubits_left_to_right"
        if evidence["observed_shots"] == 0:
            evidence["ineligibility_reason"] = "no_observed_shots"
        elif evidence["observed_shots"] > requested_shots:
            evidence["ineligibility_reason"] = "excess_observed_shots"
        else:
            evidence["raw_shot_eligible"] = True
            evidence["ineligibility_reason"] = None
        return evidence
    except Exception:
        # Missing/malformed optional evidence must not discard a completed result.
        evidence["raw_shot_eligible"] = False
        evidence["ineligibility_reason"] = "invalid_evidence"
        return evidence
