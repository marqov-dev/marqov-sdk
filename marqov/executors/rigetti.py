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
