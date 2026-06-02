"""Shared data contract for QPU Multi-Tenancy.

Defines the types that connect the characterization and scheduler workstreams:
- QMTJob: a single tenant's work unit
- NoiseProfile: device characterization data
- PackingPlan: scheduler output (which jobs go where)
- PackingResult: post-execution results per job
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from marqov.circuits import Circuit


class DeviceModality(enum.Enum):
    """Quantum hardware modality — determines packing strategy."""

    TRAPPED_ION = "trapped_ion"
    NEUTRAL_ATOM = "neutral_atom"


@dataclass
class QMTJob:
    """A single tenant's work unit for multi-tenant execution."""

    circuit: Circuit
    submitter: str
    priority: int = 0
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def num_qubits(self) -> int:
        return self.circuit.num_qubits


@dataclass
class NoiseProfile:
    """Device characterization data — produced by characterization, consumed by scheduler."""

    device_name: str
    modality: DeviceModality
    num_qubits: int
    cross_talk_matrix: np.ndarray = field(default=None)  # type: ignore[assignment]
    qubit_error_rates: np.ndarray = field(default=None)  # type: ignore[assignment]
    drift_rate: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_age_hours: float = 24.0

    def __post_init__(self) -> None:
        if self.cross_talk_matrix is None:
            self.cross_talk_matrix = np.zeros((self.num_qubits, self.num_qubits))
        if self.qubit_error_rates is None:
            self.qubit_error_rates = np.zeros(self.num_qubits)

    def cross_talk_between(self, group_a: set[int], group_b: set[int]) -> float:
        """Sum of cross-talk values between two qubit groups."""
        total = 0.0
        for i in group_a:
            for j in group_b:
                total += self.cross_talk_matrix[i, j]
        return total

    @property
    def is_stale(self) -> bool:
        age = datetime.now(timezone.utc) - self.timestamp
        return age.total_seconds() > self.max_age_hours * 3600


@dataclass
class QubitMapping:
    """Maps a job's logical qubits to physical device qubits."""

    job_id: str
    logical_to_physical: dict[int, int]

    @property
    def physical_qubits(self) -> set[int]:
        return set(self.logical_to_physical.values())


@dataclass
class PackingPlan:
    """Scheduler output — describes how jobs are packed onto a device."""

    jobs: list[QMTJob]
    mappings: list[QubitMapping]
    guard_qubits: set[int]
    device_name: str
    total_qubits: int
    expected_fidelity_ratio: float | None = None

    def __post_init__(self) -> None:
        all_physical: list[int] = []
        for mapping in self.mappings:
            all_physical.extend(mapping.physical_qubits)
        if len(all_physical) != len(set(all_physical)):
            raise ValueError(
                "Qubit mappings overlap — each physical qubit can only be assigned to one job"
            )

    @property
    def physical_qubits_used(self) -> set[int]:
        result: set[int] = set()
        for mapping in self.mappings:
            result |= mapping.physical_qubits
        return result


@dataclass
class PackingResult:
    """Post-execution result for a single job within a multi-tenant batch."""

    job_id: str
    counts: dict[str, int]
    shots: int
    single_tenant_fidelity: float | None = None
    multi_tenant_fidelity: float | None = None

    @property
    def fidelity_ratio(self) -> float | None:
        if self.single_tenant_fidelity is None or self.multi_tenant_fidelity is None:
            return None
        return self.multi_tenant_fidelity / self.single_tenant_fidelity
