"""One reading of the backend parameter names MarqovDevice and ExecutorFactory share.

:class:`~marqov.device.MarqovDevice` and
:class:`~marqov.executors.factory.ExecutorFactory` take the same logical
backend configuration as a plain dict. This module is the single place that
decides what each key is called, so one dict means the same thing on both
paths (marqov-sdk#162).

The unprefixed spellings are canonical, matching the executor configs the
factory builds. The older ``ibm_``/``azure_`` spellings that ``MarqovDevice``
used to read are still accepted with a :class:`DeprecationWarning`.

Private on purpose: it can change without a deprecation cycle of its own.
It must not import anything from ``marqov.platform``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

#: Deprecated spelling -> canonical spelling.
DEPRECATED_ALIASES: dict[str, str] = {
    "ibm_channel": "channel",
    "ibm_instance": "instance",
    "ibm_token": "token",
    "ibm_optimization_level": "optimization_level",
    "azure_subscription_id": "subscription_id",
    "azure_resource_group": "resource_group",
    "azure_workspace_name": "workspace_name",
    "azure_location": "location",
}

#: Every canonical backend key, including the ones that never diverged
#: (``device_arn``, ``backend_name``, ``target``, the S3 keys), which pass
#: through unchanged.
CANONICAL_KEYS: tuple[str, ...] = (
    "channel",
    "instance",
    "token",
    "optimization_level",
    "subscription_id",
    "resource_group",
    "workspace_name",
    "location",
    "s3_bucket",
    "s3_prefix",
    "device_arn",
    "backend_name",
    "target",
)

#: Defaults both paths apply when the key is absent.
DEFAULT_IBM_OPTIMIZATION_LEVEL = 1
DEFAULT_AZURE_LOCATION = "eastus"
DEFAULT_S3_PREFIX = "marqov"


def normalize_backend_params(params: Mapping[str, Any], *, stacklevel: int = 2) -> dict[str, Any]:
    """Return a copy of ``params`` with every deprecated key renamed to its canonical one.

    Keys that are not deprecated aliases are copied through untouched. A
    value of ``None`` or ``""`` under either spelling of a renamed key counts
    as absent: a blank deprecated key is dropped, and a blank canonical key
    is filled from its deprecated spelling, so neither raises as a conflict.

    Args:
        params: The backend configuration as supplied by the caller.
        stacklevel: Passed to :func:`warnings.warn` unchanged. The default,
            2, attributes the warning to this function's caller; callers one
            or more frames removed from user code pass a larger value.

    Returns:
        A new dict carrying only canonical spellings for the renamed keys.

    Warns:
        DeprecationWarning: When a deprecated spelling is what supplied a
            value, naming the canonical key to use instead.

    Raises:
        ValueError: When a deprecated and a canonical spelling are both
            present with different values, naming both keys.
    """
    normalized = dict(params)
    for deprecated, canonical in DEPRECATED_ALIASES.items():
        if deprecated not in normalized:
            continue
        value = normalized.pop(deprecated)
        if _is_blank(value):
            continue
        if not _is_blank(normalized.get(canonical)):
            if normalized[canonical] != value:
                raise ValueError(
                    f"Backend params set both {deprecated!r} ({value!r}) and "
                    f"{canonical!r} ({normalized[canonical]!r}) to different values; "
                    f"keep only {canonical!r}."
                )
            continue
        warnings.warn(
            f"Backend param {deprecated!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        normalized[canonical] = value
    return normalized


def _is_blank(value: Any) -> bool:
    """``None`` and ``""`` both mean "not supplied" (``0`` and ``False`` do not)."""
    return value is None or (isinstance(value, str) and value == "")
