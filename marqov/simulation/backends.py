"""Backend registry for simulation backends."""

SIMULATION_BACKENDS = {
    "qb-sim-statevector": {
        "slug": "qb-sim-statevector",
        "name": "State Vector Simulator",
        "provider": "Quantum Brilliance",
        "device_type": "simulator",
        "provider_target_id": "qpp",
        "qubit_count": 28,
        "pricing": {"taskFee": 0, "perShot": 0, "minimumCost": 0},
        "description": "Exact state vector simulation. Fast for circuits up to 28 qubits.",
    },
    "qb-sim-tensor-network": {
        "slug": "qb-sim-tensor-network",
        "name": "Tensor Network Simulator",
        "provider": "Quantum Brilliance",
        "device_type": "simulator",
        "provider_target_id": "tnqvm",
        "qubit_count": 100,
        "pricing": {"taskFee": 0, "perShot": 0, "minimumCost": 0},
        "description": "MPS tensor network simulation. Handles 100+ qubits for low-entanglement circuits.",
    },
    "qb-sim-noisy-aer": {
        "slug": "qb-sim-noisy-aer",
        "name": "Noisy Simulator (Aer)",
        "provider": "Quantum Brilliance",
        "device_type": "simulator",
        "provider_target_id": "aer",
        "qubit_count": 28,
        "pricing": {"taskFee": 0, "perShot": 0, "minimumCost": 0},
        "description": "Noisy quantum simulation using Aer backend with configurable noise models.",
    },
}

GPU_SIMULATION_BACKENDS = {
    "qb-sim-gpu-statevector": {
        "slug": "qb-sim-gpu-statevector",
        "name": "GPU State Vector Simulator",
        "provider": "Quantum Brilliance",
        "device_type": "simulator",
        "provider_target_id": "cudaq:custatevec_fp64",
        "qubit_count": 28,
        "pricing": {"taskFee": 0, "perShot": 0, "minimumCost": 0},
        "description": "NVIDIA GPU-accelerated state vector simulation via CuQuantum.",
    },
    "qb-sim-gpu-tensor-network": {
        "slug": "qb-sim-gpu-tensor-network",
        "name": "GPU Tensor Network Simulator",
        "provider": "Quantum Brilliance",
        "device_type": "simulator",
        "provider_target_id": "cudaq:qb_mps",
        "qubit_count": 100,
        "pricing": {"taskFee": 0, "perShot": 0, "minimumCost": 0},
        "description": "NVIDIA GPU-accelerated MPS tensor network simulation.",
    },
    "qb-sim-density-matrix": {
        "slug": "qb-sim-density-matrix",
        "name": "Density Matrix Simulator",
        "provider": "Quantum Brilliance",
        "device_type": "simulator",
        "provider_target_id": "cudaq:dm",
        "qubit_count": 14,
        "pricing": {"taskFee": 0, "perShot": 0, "minimumCost": 0},
        "description": "Full density matrix simulation. Requires CUDA Quantum.",
    },
}
