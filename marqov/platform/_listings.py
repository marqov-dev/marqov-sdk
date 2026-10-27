"""Parsers for the platform's read-only job and project listings.

Every parser fails closed with ``code="invalid_response"`` when the shape is
not the documented one, rather than returning a partial listing.  Job and
document IDs read from a response are validated as UUIDs before they are
used in a request path.
"""

from __future__ import annotations

import math
from typing import Any
from uuid import UUID

from ._models import JobSummary, Project
from .errors import MarqovPlatformError


def invalid(what: str) -> MarqovPlatformError:
    return MarqovPlatformError(f"The platform returned an unexpected {what}", code="invalid_response")


def rows(response: Any, key: str, what: str) -> list[dict[str, Any]]:
    """Return ``response[key]`` when it is a list of objects."""
    value = response.get(key) if isinstance(response, dict) else None
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise invalid(what)
    return value


def uuid_of(value: Any, what: str) -> str:
    """Return ``value`` as a canonical UUID string, or raise ``invalid_response``."""
    if not isinstance(value, str):
        raise invalid(what)
    try:
        return str(UUID(value))
    except ValueError:
        raise invalid(what) from None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _usd(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def job_summary(row: dict[str, Any]) -> JobSummary:
    """Build a :class:`JobSummary` from a job-list row or a status response."""
    return JobSummary(
        id=uuid_of(row.get("id"), "job identity"),
        status=_text(row.get("status")) or "unknown",
        backend=_text(row.get("backend")) or "",
        created_at=_text(row.get("created_at")),
        updated_at=_text(row.get("updated_at")),
        estimated_cost_usd=_usd(row.get("estimated_cost_usd")),
        actual_cost_usd=_usd(row.get("actual_cost_usd")),
        hold_reason=_text(row.get("hold_reason")),
    )


def project(row: dict[str, Any]) -> Project:
    """Build a :class:`Project` from a ``GET /api/projects`` row."""
    name = row.get("name")
    if not isinstance(name, str):
        raise invalid("project")
    return Project(
        id=uuid_of(row.get("id"), "project identity"),
        name=name,
        description=_text(row.get("description")) or "",
        created_at=_text(row.get("created_at")),
        updated_at=_text(row.get("updated_at")),
    )


def linked_job_ids(document: Any, project_id: str) -> list[str]:
    """Job IDs of a document's result blocks, in block order.

    Raises ``invalid_response`` if the document belongs to another project.
    """
    meta = document.get("document") if isinstance(document, dict) else None
    if not isinstance(meta, dict) or meta.get("project_id") != project_id:
        raise invalid("project document")
    return [
        uuid_of(block["job_run_id"], "linked job identity")
        for block in rows(document, "blocks", "project document")
        if block.get("block_type") == "result" and block.get("job_run_id") is not None
    ]
