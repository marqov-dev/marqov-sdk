"""Executor factory for multi-cloud quantum backend support.

This module provides a factory pattern for creating executors based on provider.
Supports AWS Braket, Quantinuum, IBM Quantum, Azure Quantum, IonQ Direct API,
Rigetti QCS, CUDA-Q, Quantum Brilliance, and the Local simulator.

Example:
    >>> backend_config = {
    ...     "provider": "AWS Braket",
    ...     "device_arn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    ...     "s3_bucket": "my-bucket",
    ...     "s3_prefix": "jobs",
    ... }
    >>> executor = ExecutorFactory.create_executor("sv1", backend_config)
    >>> result = await executor.execute(circuit, shots=1000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig
from marqov.executors.base import BaseExecutor
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig
from marqov.executors.cudaq import CudaqExecutor, CudaqExecutorConfig, _SLUG_TO_TARGET
from marqov.executors.ibm import IBMExecutor, IBMExecutorConfig
from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig
from marqov.executors.local import LocalExecutor
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig
from marqov.simulation.config import SimulationConfig
from marqov.simulation.executor import SimulationExecutor
from marqov.executors.quantinuum import QuantinuumExecutor, QuantinuumExecutorConfig

if TYPE_CHECKING:
    pass


class ExecutorFactory:
    """Factory for creating quantum executors based on provider.

    Supports multiple quantum cloud providers through a unified interface.
    Each provider has its own executor implementation that inherits from
    BaseExecutor, ensuring consistent behavior across providers.

    Supported Providers:
        - AWS Braket: Simulators (SV1, DM1, TN1) and QPUs (IonQ, Rigetti, IQM, QuEra)
        - IBM Quantum: Heron r2, Eagle processors via Qiskit Runtime SamplerV2
        - Azure Quantum: Quantinuum, PASQAL, IonQ, Rigetti (Qiskit/Cirq support)
        - Quantinuum: Quantinuum devices and emulators (via pytket-quantinuum)
        - Rigetti QCS: Native pyQuil submission to Rigetti QPUs or the local QVM
        - CUDA-Q: NVIDIA GPU/CPU simulation, or IQM hardware via CUDA-Q's IQM target
        - Quantum Brilliance: routed through the local simulation backend
        - Local: QuantumFlow simulator (no cloud required)
        - IonQ Direct: Native IonQ REST API (no AWS/Braket intermediary)

    Example:
        >>> from marqov.executors.factory import ExecutorFactory
        >>> backend_config = {
        ...     "provider": "AWS Braket",
        ...     "device_arn": "arn:aws:braket:...",
        ...     "s3_bucket": "my-bucket",
        ... }
        >>> executor = ExecutorFactory.create_executor("sv1", backend_config)
        >>> result = await executor.execute(circuit, shots=1000)
    """

    @classmethod
    def create_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> BaseExecutor:
        """Create an executor for the given backend.

        Args:
            backend_slug: Backend identifier (e.g., "sv1", "ibm-kyoto")
            backend_config: Backend configuration from database, must include:
                - provider: Provider name ("AWS Braket", "IBM Quantum", etc.)
                - Provider-specific fields (device_arn for Braket, etc.)

        Returns:
            Configured executor instance ready for circuit execution.

        Raises:
            ValueError: If provider is not supported or config is invalid.

        Example:
            >>> config = {
            ...     "provider": "AWS Braket",
            ...     "device_arn": "arn:aws:braket:...",
            ...     "s3_bucket": "amazon-braket-my-bucket",
            ...     "s3_prefix": "jobs",
            ... }
            >>> executor = ExecutorFactory.create_executor("sv1", config)
        """
        provider = backend_config.get("provider")

        if not provider:
            raise ValueError(f"Backend config missing 'provider' field for {backend_slug}")

        # Handle local simulator
        if backend_slug == "local" or provider == "Local":
            return LocalExecutor()

        # NVIDIA CUDA-Q (GPU/CPU statevector, direct IQM)
        if provider == "CUDA-Q" or backend_slug in _SLUG_TO_TARGET:
            return cls._create_cudaq_executor(backend_slug, backend_config)

        # AWS Braket
        if provider == "AWS Braket":
            return cls._create_braket_executor(backend_slug, backend_config)

        # IBM Quantum
        if provider == "IBM Quantum":
            return cls._create_ibm_executor(backend_slug, backend_config)

        # Quantinuum
        if provider == "Quantinuum":
            return cls._create_quantinuum_executor(backend_slug, backend_config)

        # Azure Quantum
        if provider == "Azure Quantum":
            return cls._create_azure_executor(backend_slug, backend_config)

        # IonQ Direct API
        if provider == "IonQ Direct":
            return cls._create_ionq_executor(backend_slug, backend_config)

        # Rigetti QCS (and local QVM)
        if provider == "Rigetti QCS":
            return cls._create_rigetti_executor(backend_slug, backend_config)

        # C++ simulation backends (qpp, tnqvm, cudaq, aer)
        if provider == "Quantum Brilliance":
            return cls._create_simulation_executor(backend_slug, backend_config)

        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(cls.get_supported_providers())}."
        )

    @classmethod
    def _create_braket_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> BraketExecutor:
        """Create AWS Braket executor from configuration.

        Args:
            backend_slug: Backend slug (e.g., "sv1", "rigetti-ankaa-3")
            backend_config: Configuration with device_arn, s3_bucket, etc.

        Returns:
            Configured BraketExecutor instance.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ["device_arn", "s3_bucket"]
        missing_fields = [f for f in required_fields if f not in backend_config]

        if missing_fields:
            raise ValueError(
                f"BraketExecutor config missing required fields for {backend_slug}: "
                f"{', '.join(missing_fields)}"
            )

        config = BraketExecutorConfig(
            device_arn=backend_config["device_arn"],
            s3_bucket=backend_config["s3_bucket"],
            s3_prefix=backend_config.get("s3_prefix", "marqov"),
            aws_profile=backend_config.get("aws_profile"),
            aws_region=backend_config.get("region"),
            poll_interval_seconds=backend_config.get("poll_interval_seconds", 1.0),
            timeout_seconds=backend_config.get("timeout_seconds"),
        )

        return BraketExecutor(config)

    @classmethod
    def _create_quantinuum_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> QuantinuumExecutor:
        """Create Quantinuum executor from configuration.
        """
        required_fields = [
            "device_name",
        ]
        missing_fields = [f for f in required_fields if f not in backend_config]
        if missing_fields:
            raise ValueError(
                f"QuantinuumExecutor config missing required fields for {backend_slug}: "
                f"{', '.join(missing_fields)}"
            )

        config = QuantinuumExecutorConfig(
            device_name=backend_config["device_name"],
            simulator=backend_config.get("simulator", "state-vector"),
            group=backend_config.get("group"),
            label=backend_config.get("label", "job"),
            provider=backend_config.get("auth_provider"),
            machine_debug=backend_config.get("machine_debug", False),
            api_handler=backend_config.get("api_handler"),
            compilation_config=backend_config.get("compilation_config"),
            options=backend_config.get("options", {}),
            poll_interval_seconds=backend_config.get("poll_interval_seconds", 2.0),
            timeout_seconds=backend_config.get("timeout_seconds", 300.0),
            optimisation_level=backend_config.get("optimisation_level", 2),
        )
        return QuantinuumExecutor(config)

    @classmethod
    def _create_azure_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> AzureQuantumExecutor:
        """Create Azure Quantum executor from configuration.

        Args:
            backend_slug: Backend slug (e.g., "azure-ionq-simulator")
            backend_config: Configuration with Azure workspace details.

        Returns:
            Configured AzureQuantumExecutor instance.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = [
            "subscription_id",
            "resource_group",
            "workspace_name",
            "location",
            "target",
        ]
        missing_fields = [f for f in required_fields if f not in backend_config]

        if missing_fields:
            raise ValueError(
                f"AzureQuantumExecutor config missing required fields for {backend_slug}: "
                f"{', '.join(missing_fields)}"
            )

        config = AzureQuantumExecutorConfig(
            subscription_id=backend_config["subscription_id"],
            resource_group=backend_config["resource_group"],
            workspace_name=backend_config["workspace_name"],
            location=backend_config["location"],
            target=backend_config["target"],
            framework=backend_config.get("framework", "qiskit"),
            timeout_seconds=backend_config.get("timeout_seconds", 300.0),
            poll_interval_seconds=backend_config.get("poll_interval_seconds", 2.0),
        )

        return AzureQuantumExecutor(config)

    @classmethod
    def _create_ibm_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> IBMExecutor:
        """Create IBM Quantum executor from configuration.

        Args:
            backend_slug: Backend slug (e.g., "ibm-kingston", "ibm-brisbane")
            backend_config: Configuration with IBM Quantum credentials and options.

        Returns:
            Configured IBMExecutor instance.

        Raises:
            ValueError: If required fields are missing.
        """
        # backend_name is required — map slug to IBM backend name if needed
        backend_name = backend_config.get(
            "backend_name",
            backend_slug.replace("-", "_"),
        )

        config = IBMExecutorConfig(
            backend_name=backend_name,
            channel=backend_config.get("channel", "ibm_quantum"),
            instance=backend_config.get("instance", "ibm-q/open/main"),
            token=backend_config.get("token"),
            optimization_level=backend_config.get("optimization_level", 1),
            resilience_level=backend_config.get("resilience_level", 1),
            poll_interval_seconds=backend_config.get("poll_interval_seconds", 2.0),
            timeout_seconds=backend_config.get("timeout_seconds"),
        )

        return IBMExecutor(config)

    @classmethod
    def _create_ionq_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> IonQExecutor:
        """Create an IonQ Direct API executor from configuration.

        The IonQ API key may be supplied via config or the IONQ_API_KEY
        environment variable, so no field is strictly required here.

        Args:
            backend_slug: Backend slug used as the IonQ target if not given
                (e.g. "simulator", "qpu.aria-1").
            backend_config: Configuration with optional target, api_key,
                base_url, noise_model, and polling options.

        Returns:
            Configured IonQExecutor instance.
        """
        # Only forward keys present in backend_config and rely on
        # IonQExecutorConfig's own defaults otherwise, so factory and dataclass
        # defaults can't drift apart. The target falls back to the backend slug.
        config_kwargs: dict[str, Any] = {
            "target": backend_config.get("target", backend_slug),
        }
        for key in (
            "api_key",
            "base_url",
            "poll_interval_seconds",
            "timeout_seconds",
            "noise_model",
        ):
            if key in backend_config:
                config_kwargs[key] = backend_config[key]

        config = IonQExecutorConfig(**config_kwargs)

        return IonQExecutor(config)

    @classmethod
    def _create_rigetti_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> RigettiExecutor:
        """Create a Rigetti QCS executor from configuration.

        No field is strictly required: the quantum-processor id falls back to the
        backend slug, so ``{"provider": "Rigetti QCS"}`` with a ``"2q-qvm"`` slug
        runs against a local QVM without any cloud account.

        Args:
            backend_slug: Backend slug used as the quantum-processor id if not
                given (e.g. ``"2q-qvm"``, ``"Ankaa-3"``).
            backend_config: Configuration with optional quantum_processor_id,
                as_qvm, and timeout options.

        Returns:
            Configured RigettiExecutor instance.
        """
        # Only forward keys present in backend_config and rely on
        # RigettiExecutorConfig's own defaults otherwise, so factory and dataclass
        # defaults can't drift apart. The processor id falls back to the slug.
        config_kwargs: dict[str, Any] = {
            "quantum_processor_id": backend_config.get("quantum_processor_id", backend_slug),
        }
        for key in (
            "as_qvm",
            "compiler_timeout_seconds",
            "execution_timeout_seconds",
            "timeout_seconds",
        ):
            if key in backend_config:
                config_kwargs[key] = backend_config[key]

        config = RigettiExecutorConfig(**config_kwargs)

        return RigettiExecutor(config)

    @classmethod
    def _create_cudaq_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> CudaqExecutor:
        """Create an NVIDIA CUDA-Q executor from configuration.

        No field is strictly required for the CPU/GPU targets: the target falls
        back from the backend slug (``cudaq-cpu`` / ``cudaq-gpu`` / ``cudaq-iqm``)
        or an explicit ``target`` in the config. The ``iqm`` target additionally
        needs ``iqm_url`` (and a token via config or the ``IQM_TOKEN`` env var).

        Args:
            backend_slug: Backend slug; maps to a CUDA-Q target when it is one of
                ``cudaq-cpu`` / ``cudaq-gpu`` / ``cudaq-iqm``.
            backend_config: Optional ``target``, ``iqm_url``, ``iqm_token``,
                ``gpu_fallback_to_cpu``, ``seed``, ``target_options``.

        Returns:
            Configured CudaqExecutor instance.
        """
        target = backend_config.get("target") or _SLUG_TO_TARGET.get(backend_slug, "qpp-cpu")

        config_kwargs: dict[str, Any] = {"target": target}
        for key in ("iqm_url", "iqm_token", "gpu_fallback_to_cpu", "seed", "target_options"):
            if key in backend_config:
                config_kwargs[key] = backend_config[key]

        return CudaqExecutor(CudaqExecutorConfig(**config_kwargs))

    @classmethod
    def _create_simulation_executor(
        cls,
        backend_slug: str,
        backend_config: dict[str, Any],
    ) -> SimulationExecutor:
        """Create simulation executor from configuration.

        Args:
            backend_slug: Backend slug (e.g., "qb-sim-statevector")
            backend_config: Configuration with provider_target_id and optional params.

        Returns:
            Configured SimulationExecutor instance.
        """
        config = SimulationConfig.from_backend(backend_config)
        return SimulationExecutor(config)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported providers.

        Returns:
            List of provider names.

        Example:
            >>> ExecutorFactory.get_supported_providers()
            ['AWS Braket', 'IBM Quantum', 'Azure Quantum', 'IonQ Direct', 'Rigetti QCS', 'Quantum Brilliance', 'CUDA-Q', 'Local', 'Quantinuum']
        """
        return [
            "AWS Braket",
            "IBM Quantum",
            "Azure Quantum",
            "IonQ Direct",
            "Rigetti QCS",
            "Quantum Brilliance",
            "CUDA-Q",
            "Local",
            "Quantinuum",
        ]

    @classmethod
    def is_provider_supported(cls, provider: str) -> bool:
        """Check if a provider is currently supported.

        Args:
            provider: Provider name to check.

        Returns:
            True if provider is supported, False otherwise.

        Example:
            >>> ExecutorFactory.is_provider_supported("AWS Braket")
            True
            >>> ExecutorFactory.is_provider_supported("IBM Quantum")
            True
        """
        return provider in cls.get_supported_providers()
