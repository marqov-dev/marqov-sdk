"""Quantum execution backends.

This module provides executors for running quantum circuits on various backends.

Available executors:
- LocalExecutor: QuantumFlow simulator (no cloud required)
- BraketExecutor: AWS Braket (simulators and QPUs)
- AzureQuantumExecutor: Azure Quantum (Quantinuum, PASQAL, IonQ, Rigetti)
- QuantinuumExecutor: Quantinuum devices and emulators (via pytket-quantinuum)
- IBMExecutor: IBM Quantum (Heron r2, Eagle, etc. via Qiskit Runtime)
- IonQExecutor: IonQ Direct API (native REST, no Braket intermediary)
- RigettiExecutor: Rigetti QCS QPUs and the local QVM (via pyquil)

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
from marqov.executors.factory import ExecutorFactory
from marqov.executors.ibm import IBMExecutor, IBMExecutorConfig
from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig
from marqov.executors.local import LocalExecutor
from marqov.executors.quantinuum import QuantinuumExecutor, QuantinuumExecutorConfig
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig
from marqov.simulation.executor import SimulationExecutor

__all__ = [
    "AzureQuantumExecutor",
    "AzureQuantumExecutorConfig",
    "BaseExecutor",
    "BraketExecutor",
    "BraketExecutorConfig",
    "DeviceStatus",
    "ExecutionResult",
    "ExecutorFactory",
    "IBMExecutor",
    "IBMExecutorConfig",
    "IonQExecutor",
    "IonQExecutorConfig",
    "LocalExecutor",
    "QuantinuumExecutor",
    "QuantinuumExecutorConfig",
    "RigettiExecutor",
    "RigettiExecutorConfig",
    "SimulationExecutor",
]
