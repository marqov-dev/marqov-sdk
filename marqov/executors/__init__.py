"""Quantum execution backends.

This module provides executors for running quantum circuits on various backends.

Available executors:
- LocalExecutor: QuantumFlow simulator (no cloud required)
- BraketExecutor: AWS Braket (simulators and QPUs)
- AzureQuantumExecutor: Azure Quantum (Quantinuum, PASQAL, IonQ, Rigetti)
- IBMExecutor: IBM Quantum (Heron r2, Eagle, etc. via Qiskit Runtime)

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
from marqov.executors.local import LocalExecutor
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
    "LocalExecutor",
    "SimulationExecutor",
]
