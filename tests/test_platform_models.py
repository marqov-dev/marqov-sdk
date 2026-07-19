"""Tests for marqov.platform error hierarchy and dataclass models (Task 2)."""

import pytest

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
from marqov.platform._models import (
    Backend,
    JobStatus,
    PlatformInfo,
    PlatformResult,
    is_terminal,
)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestMarqovPlatformError:
    def test_stores_message(self):
        e = MarqovPlatformError("something broke")
        assert e.message == "something broke"

    def test_stores_code(self):
        e = MarqovPlatformError("msg", code="err/code")
        assert e.code == "err/code"

    def test_stores_status(self):
        e = MarqovPlatformError("msg", status=500)
        assert e.status == 500

    def test_str_without_code(self):
        e = MarqovPlatformError("plain message")
        assert str(e) == "plain message"

    def test_str_with_code(self):
        e = MarqovPlatformError("bad key", code="auth/invalid-key")
        assert "bad key" in str(e)
        assert "auth/invalid-key" in str(e)

    def test_code_defaults_to_none(self):
        e = MarqovPlatformError("msg")
        assert e.code is None

    def test_status_defaults_to_none(self):
        e = MarqovPlatformError("msg")
        assert e.status is None

    def test_is_exception(self):
        with pytest.raises(MarqovPlatformError):
            raise MarqovPlatformError("err")


class TestErrorSubclasses:
    @pytest.mark.parametrize(
        "cls",
        [
            AuthenticationError,
            PermissionTierError,
            BackendUnavailable,
            PaidBackendNotSupportedYet,
            InvalidProgram,
            JobFailed,
            RateLimited,
            TransportError,
        ],
    )
    def test_inherits_base(self, cls):
        e = cls("test")
        assert isinstance(e, MarqovPlatformError)

    @pytest.mark.parametrize(
        "cls",
        [
            AuthenticationError,
            PermissionTierError,
            BackendUnavailable,
            PaidBackendNotSupportedYet,
            InvalidProgram,
            JobFailed,
            RateLimited,
            TransportError,
        ],
    )
    def test_passes_code_and_status(self, cls):
        e = cls("msg", code="x", status=400)
        assert e.code == "x"
        assert e.status == 400

    def test_authentication_error_catchable(self):
        with pytest.raises(MarqovPlatformError):
            raise AuthenticationError("no key")

    def test_paid_backend_not_supported_yet_has_docstring(self):
        assert PaidBackendNotSupportedYet.__doc__ is not None
        doc = PaidBackendNotSupportedYet.__doc__
        assert "deprecated" in doc.lower() or "retain" in doc.lower()


# ---------------------------------------------------------------------------
# PlatformResult
# ---------------------------------------------------------------------------


class TestPlatformResult:
    def test_counts_present(self):
        r = PlatformResult(raw={"counts": {"00": 5, "11": 3}})
        assert r.counts == {"00": 5, "11": 3}

    def test_counts_absent_is_none(self):
        r = PlatformResult(raw={})
        assert r.counts is None

    def test_probabilities_from_counts(self):
        r = PlatformResult(raw={"counts": {"00": 3, "11": 1}})
        probs = r.probabilities
        assert probs == pytest.approx({"00": 0.75, "11": 0.25})

    def test_probabilities_empty_when_counts_absent(self):
        r = PlatformResult(raw={})
        assert r.probabilities == {}

    def test_probabilities_empty_when_counts_all_zero(self):
        r = PlatformResult(raw={"counts": {"00": 0}})
        assert r.probabilities == {}

    def test_probabilities_sums_to_one(self):
        r = PlatformResult(raw={"counts": {"00": 512, "11": 488}})
        total = sum(r.probabilities.values())
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------


class TestIsTerminal:
    def test_completed_is_terminal(self):
        assert is_terminal("completed") is True

    def test_failed_is_terminal(self):
        assert is_terminal("failed") is True

    def test_cancelled_is_terminal(self):
        assert is_terminal("cancelled") is True

    def test_running_is_not_terminal(self):
        assert is_terminal("running") is False

    def test_pending_is_not_terminal(self):
        assert is_terminal("pending") is False

    def test_cancelling_is_not_terminal(self):
        assert is_terminal("cancelling") is False

    def test_unknown_status_returns_false_no_raise(self):
        # Future server-side statuses must never crash the client.
        assert is_terminal("archived") is False
        assert is_terminal("COMPLETED") is False  # case-sensitive
        assert is_terminal("") is False


# ---------------------------------------------------------------------------
# JobStatus
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_completed_equals_string(self):
        assert JobStatus.COMPLETED == "completed"

    def test_failed_equals_string(self):
        assert JobStatus.FAILED == "failed"

    def test_pending_equals_string(self):
        assert JobStatus.PENDING == "pending"

    def test_running_equals_string(self):
        assert JobStatus.RUNNING == "running"

    def test_cancelling_equals_string(self):
        assert JobStatus.CANCELLING == "cancelling"

    def test_cancelled_equals_string(self):
        assert JobStatus.CANCELLED == "cancelled"

    def test_is_str_subclass(self):
        assert isinstance(JobStatus.COMPLETED, str)

    def test_unknown_value_raises(self):
        # Confirms NEVER-call-JobStatus(raw) policy: it raises on unknowns.
        with pytest.raises(ValueError):
            JobStatus("archived")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TestBackend:
    def test_construction(self):
        b = Backend(
            slug="sv1",
            name="StateVec Simulator",
            provider="marqov",
            device_type="simulator",
            status="online",
            is_available=True,
            pricing={"type": "free"},
            supported_program_types=["qasm3"],
        )
        assert b.slug == "sv1"
        assert b.supported_program_types == ["qasm3"]
        assert b.extra == {}

    def test_extra_field(self):
        b = Backend(
            slug="qpu1",
            name="QPU",
            provider="ibm",
            device_type="qpu",
            status="online",
            is_available=False,
            pricing={},
            supported_program_types=[],
            extra={"region": "us-east-1"},
        )
        assert b.extra["region"] == "us-east-1"


# ---------------------------------------------------------------------------
# PlatformInfo
# ---------------------------------------------------------------------------


class TestPlatformInfo:
    def test_construction(self):
        info = PlatformInfo(sdk_version="0.2.0", api_version="v1")
        assert info.sdk_version == "0.2.0"
        assert info.api_version == "v1"
