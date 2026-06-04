"""Rigetti executor for running circuits on QVM/QPU devices via pyQuil.

This module provides RigettiExecutor for executing quantum circuits on Rigetti's QVM and QPU backends.
Supports local QVM execution (Docker qvm + quilc) and Rigetti QCS processors.

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import RigettiExecutor, RigettiExecutorConfig
    >>>
    >>> config = RigettiExecutorConfig(quantum_processor_id="2q-qvm", as_qvm=True)
    >>> executor = RigettiExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}
"""

from __future__ import annotations

import asyncio
import socket
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any
from urllib.parse import urlparse

import numpy as np

from marqov.executors.base import BaseExecutor


@dataclass
class RigettiExecutorConfig:
    """Configuration for RigettiExecutor.

    Attributes:
        quantum_processor_id: Rigetti processor name (e.g., "Ankaa-3", "2q-qvm").
        as_qvm: Force QVM mode when creating pyQuil quantum computer.
            If None, inferred from quantum_processor_id containing "qvm".
        poll_interval_seconds: Polling interval while waiting for run completion.
        timeout_seconds: Maximum wait time for execution completion.
        qvm_url: Optional endpoint used for local QVM status checks.
        quilc_url: Optional endpoint used for local quilc status checks.
    """

    quantum_processor_id: str = "2q-qvm"
    as_qvm: bool | None = None
    poll_interval_seconds: float = 0.2
    timeout_seconds: float | None = 120.0
    qvm_url: str = "http://127.0.0.1:5000"
    quilc_url: str = "tcp://127.0.0.1:5555"


def _is_qvm_target(quantum_processor_id: str, as_qvm: bool | None) -> bool:
    """Infer whether this target should be treated as QVM."""
    if as_qvm is not None:
        return as_qvm
    return "qvm" in quantum_processor_id.lower()


class RigettiExecutor(BaseExecutor):
    """Execute circuits on Rigetti backends using pyQuil."""

    _STATUS_MAP = {
        "online": "online",
        "available": "online",
        "offline": "offline",
        "retired": "offline",
        "maintenance": "maintenance",
        "degraded": "maintenance",
    }

    def __init__(self, config: RigettiExecutorConfig | None = None) -> None:
        self.config = config or RigettiExecutorConfig()
        self._qc: Any = None
        self._current_job_id: str | None = None
        self._active_future: asyncio.Future[Any] | None = None

        try:
            from pyquil import get_qc as _get_qc

            self._get_qc = _get_qc
        except ImportError as exc:
            raise ImportError(
                "pyQuil is required for RigettiExecutor. Install with: pip install marqov[rigetti]"
            ) from exc
