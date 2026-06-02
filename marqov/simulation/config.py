"""Configuration for simulation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marqov.simulation.noise import NoiseModel


@dataclass
class SimulationConfig:
    """Configuration for a simulation backend."""

    backend_id: str
    backend_type: str
    num_qubits: int = 0
    max_bond_dimension: int | None = None
    svd_cutoff: float | None = None
    rel_svd_cutoff: float | None = None
    seed: int | None = None
    noise_model: NoiseModel | None = None
    extract_state_vector: bool = False

    @classmethod
    def from_backend(
        cls,
        backend_config: dict,
        *,
        noise_model: NoiseModel | None = None,
        extract_state_vector: bool = False,
    ) -> SimulationConfig:
        """Create config from database backend record."""
        slug = backend_config.get("slug", "")
        target_id = backend_config.get("provider_target_id", "qpp")

        type_map = {
            "qb-sim-statevector": "statevector",
            "qb-sim-gpu-statevector": "gpu-statevector",
            "qb-sim-tensor-network": "tensor-network",
            "qb-sim-gpu-tensor-network": "gpu-tensor-network",
            "qb-sim-density-matrix": "density-matrix",
            "qb-sim-noisy-aer": "noisy",
        }

        return cls(
            backend_id=target_id,
            backend_type=type_map.get(slug, "statevector"),
            max_bond_dimension=backend_config.get("max_bond_dimension"),
            svd_cutoff=backend_config.get("svd_cutoff"),
            rel_svd_cutoff=backend_config.get("rel_svd_cutoff"),
            seed=backend_config.get("seed"),
            noise_model=noise_model,
            extract_state_vector=extract_state_vector,
        )
