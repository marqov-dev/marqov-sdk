"""marqov.platform — public API for the Marqov managed-quantum platform client.

This package exposes the complete surface needed to submit quantum jobs to the
Marqov Platform, poll for results, and handle any errors that may arise.  All
names below are importable directly from ``marqov.platform``::

    from marqov.platform import MarqovClient, Job, JobStatus, PlatformResult
    from marqov.platform import MarqovPlatformError, JobFailed

**Laziness guarantee:** importing ``marqov`` (the public SDK) does **not**
import this sub-package.  ``marqov.platform`` is only loaded when explicitly
imported by the caller.
"""

from marqov.platform._models import (
    Backend,
    JobStatus,
    PlatformInfo,
    PlatformResult,
    is_terminal,
)
from marqov.platform.client import MarqovClient
from marqov.platform.errors import (
    AuthenticationError,
    BackendUnavailable,
    InvalidProgram,
    JobFailed,
    MarqovPlatformError,
    PaidBackendNotSupportedYet,
    PermissionTierError,
    RateLimited,
    TransportError,
)
from marqov.platform.job import Job

__all__ = [
    # Client
    "MarqovClient",
    # Job handle
    "Job",
    # Models
    "JobStatus",
    "Backend",
    "PlatformResult",
    "PlatformInfo",
    "is_terminal",
    # Errors
    "MarqovPlatformError",
    "AuthenticationError",
    "PermissionTierError",
    "BackendUnavailable",
    "PaidBackendNotSupportedYet",
    "InvalidProgram",
    "JobFailed",
    "RateLimited",
    "TransportError",
]
