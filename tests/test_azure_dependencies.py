"""Credential-free import and native-job checks for the Azure adapter extras."""

from unittest.mock import Mock

import pytest


def test_vendor_framework_adapters_import():
    # Deliberately do not skip missing vendor dependencies: the full-suite and
    # installed-wheel jobs install [all], which promises these adapter imports.
    from azure.quantum.cirq import AzureQuantumService
    from azure.quantum.qiskit import AzureQuantumJob, AzureQuantumProvider

    assert callable(AzureQuantumProvider)
    assert callable(AzureQuantumService)
    assert callable(AzureQuantumJob)


def test_native_qiskit_job_timeout_and_cancel_contract():
    from azure.quantum.qiskit import AzureQuantumJob

    # Use the real vendor job with injected workspace/job objects. Neither its
    # constructor nor these timeout/cancel branches contact Azure.
    workspace = Mock()
    backend = Mock()
    backend.provider.get_workspace.return_value = workspace
    native_job = Mock()
    native_job.id = "offline-native-job"
    native_job.wait_until_completed.side_effect = TimeoutError("offline poll budget")
    job = AzureQuantumJob(backend, azure_job=native_job)

    assert job.job_id() == native_job.id
    with pytest.raises(TimeoutError, match="offline poll budget"):
        job.result(timeout=0.03)
    native_job.wait_until_completed.assert_called_once_with(timeout_secs=0.03)
    job.cancel()
    workspace.cancel_job.assert_called_once_with(native_job)
