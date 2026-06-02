"""Executor factory for multi-cloud quantum backend support.

This module provides a factory pattern for creating executors based on provider.
Supports AWS Braket, IBM Quantum, Azure Quantum, and IonQ Direct API.

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
from marqov.executors.ibm import IBMExecutor, IBMExecutorConfig
from marqov.executors.local import LocalExecutor
from marqov.simulation.config import SimulationConfig
from marqov.simulation.executor import SimulationExecutor

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
        - Local: QuantumFlow simulator (no cloud required)
        - IonQ Direct API: Coming soon

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

        # AWS Braket
        if provider == "AWS Braket":
            return cls._create_braket_executor(backend_slug, backend_config)

        # IBM Quantum
        if provider == "IBM Quantum":
            return cls._create_ibm_executor(backend_slug, backend_config)

        # Azure Quantum
        if provider == "Azure Quantum":
            return cls._create_azure_executor(backend_slug, backend_config)

        # IonQ Direct API (future)
        if provider == "IonQ Direct":
            raise NotImplementedError(
                f"IonQ Direct API support coming soon. "
                f"See docs/MULTI_CLOUD_EXECUTOR_DESIGN.md for roadmap."
            )

        # C++ simulation backends (qpp, tnqvm, cudaq, aer)
        if provider == "Quantum Brilliance":
            return cls._create_simulation_executor(backend_slug, backend_config)

        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: AWS Braket, IBM Quantum, Azure Quantum, Quantum Brilliance, Local. "
            f"Coming soon: IonQ Direct."
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
            ['AWS Braket', 'IBM Quantum', 'Azure Quantum', 'Local']
        """
        return [
            "AWS Braket",
            "IBM Quantum",
            "Azure Quantum",
            "Quantum Brilliance",
            "Local",
            # "IonQ Direct",     # Coming soon
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
