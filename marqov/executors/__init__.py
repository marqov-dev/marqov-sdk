"""Quantum execution backends.

This module provides executors for running quantum circuits on various backends.

Available executors:
- LocalExecutor: QuantumFlow simulator (no cloud required)
- CudaqExecutor: NVIDIA CUDA-Q (GPU/CPU statevector, direct IQM)
- CUNQAExecutor: CESGA CUNQA distributed-QC emulator (Slurm-based vQPUs)
- BraketExecutor: AWS Braket (simulators and QPUs)
- AzureQuantumExecutor: Azure Quantum (Quantinuum, PASQAL, IonQ, Rigetti)
- QuantinuumExecutor: Quantinuum devices and emulators (via pytket-quantinuum)
- IBMExecutor: IBM Quantum (Heron r2, Eagle, etc. via Qiskit Runtime)
- IonQExecutor: IonQ Direct API (native REST, no Braket intermediary)
- RigettiExecutor: Rigetti QCS QPUs and the local QVM (via pyquil)
- QiliSDKExecutor: Qilimanjaro qilisdk local simulators (QiliSim/QutipBackend)

Example:
    >>> from marqov.executors import LocalExecutor
    >>> from marqov.circuits import bell_state
    >>>
    >>> executor = LocalExecutor()
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)
"""

from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig
from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig
from marqov.executors.cudaq import CudaqExecutor, CudaqExecutorConfig
from marqov.executors.cunqa import CUNQAExecutor, CUNQAExecutorConfig
from marqov.executors.factory import ExecutorFactory
from marqov.executors.ibm import IBMExecutor, IBMExecutorConfig
from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig
from marqov.executors.local import LocalExecutor
from marqov.executors.qilisdk import QiliSDKExecutor, QiliSDKExecutorConfig
from marqov.executors.quantinuum import QuantinuumExecutor, QuantinuumExecutorConfig
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig
from marqov.simulation.executor import SimulationExecutor

__all__ = [
    "AzureQuantumExecutor",
    "AzureQuantumExecutorConfig",
    "BaseExecutor",
    "BraketExecutor",
    "BraketExecutorConfig",
    "CUNQAExecutor",
    "CUNQAExecutorConfig",
    "CudaqExecutor",
    "CudaqExecutorConfig",
    "DeviceStatus",
    "ExecutionResult",
    "ExecutorFactory",
    "IBMExecutor",
    "IBMExecutorConfig",
    "IonQExecutor",
    "IonQExecutorConfig",
    "LocalExecutor",
    "QiliSDKExecutor",
    "QiliSDKExecutorConfig",
    "QuantinuumExecutor",
    "QuantinuumExecutorConfig",
    "RigettiExecutor",
    "RigettiExecutorConfig",
    "SimulationExecutor",
]
