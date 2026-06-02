"""QPU Multi-Tenancy (QMT) — spatial multiplexing of quantum jobs.

Provides two subsystems:
- characterization: cross-talk measurement and noise profiling
- scheduler: job grouping, qubit packing, and result attribution
"""

from marqov.qmt.models import (
    DeviceModality,
    NoiseProfile,
    PackingPlan,
    PackingResult,
    QMTJob,
    QubitMapping,
)

__all__ = [
    "DeviceModality",
    "NoiseProfile",
    "PackingPlan",
    "PackingResult",
    "QMTJob",
    "QubitMapping",
]
