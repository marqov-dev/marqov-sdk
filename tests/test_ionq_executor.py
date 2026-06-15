"""Tests for the IonQ Direct API executor.

All tests use an injected stub HTTP session, so they never touch the network or
require IonQ credentials. A commented integration template at the bottom shows how
to run against the real API via the IONQ_API_KEY environment variable.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from marqov.circuits import Circuit, bell_state
from marqov.executors import ExecutionResult, IonQExecutor
from marqov.executors.ionq import IonQExecutorConfig


def _make_session(router) -> MagicMock:
    """Build a stub HTTP session whose .request() is driven by ``router``.

    Mirrors ``requests.Session.request(method, url, **kwargs) -> Response``.

    Args:
        router: Callable (method, url, **kwargs) -> dict returning the JSON body
            for a given request. Raising propagates to the executor.

    Returns:
        A MagicMock session compatible with IonQExecutor's request path.
    """
    session = MagicMock()

    def request(method, url, **kwargs):  # noqa: ANN001, ANN202
        response = MagicMock()
        response.json.return_value = router(method, url, **kwargs)
        response.raise_for_status.return_value = None
        return response

    session.request.side_effect = request
    return session


def _patch_qasm(num_qubits: int = 2):
    """Patch _circuit_to_qasm so execute tests don't need a real conversion.

    The real to_qiskit() -> QASM conversion is covered by TestCircuitToQasm.
    """
    return patch.object(
        IonQExecutor,
        "_circuit_to_qasm",
        return_value=("OPENQASM 2.0;\n", num_qubits),
    )


class TestIonQExecutorConfig:
    """Tests for IonQExecutorConfig."""

    def test_defaults(self) -> None:
        """Config has sensible defaults."""
        config = IonQExecutorConfig()
        assert config.target == "simulator"
        assert config.api_key is None
        assert config.base_url == "https://api.ionq.co/v0.3"
        assert config.poll_interval_seconds == 1.0
        assert config.timeout_seconds is None
        assert config.noise_model is None

    def test_custom_values(self) -> None:
        """Config stores custom values."""
        config = IonQExecutorConfig(
            target="qpu.aria-1",
            api_key="secret",
            base_url="https://example.test/v1",
            poll_interval_seconds=0.5,
            timeout_seconds=120.0,
            noise_model="aria-1",
        )
        assert config.target == "qpu.aria-1"
        assert config.api_key == "secret"
        assert config.base_url == "https://example.test/v1"
        assert config.poll_interval_seconds == 0.5
        assert config.timeout_seconds == 120.0
        assert config.noise_model == "aria-1"


class TestHistogramToCounts:
    """Tests for the histogram -> counts conversion."""

    def test_basic_conversion(self) -> None:
        """State indices are zero-padded and scaled by shots."""
        counts = IonQExecutor._histogram_to_counts({"0": 0.5, "3": 0.5}, 1000, 2)
        assert counts == {"00": 500, "11": 500}

    def test_zero_padding(self) -> None:
        """Indices are padded to the qubit width."""
        counts = IonQExecutor._histogram_to_counts({"1": 1.0}, 100, 3)
        assert counts == {"001": 100}

    def test_rounding(self) -> None:
        """Probabilities are rounded to the nearest integer count."""
        counts = IonQExecutor._histogram_to_counts({"0": 0.333, "1": 0.667}, 9, 1)
        assert counts == {"0": 3, "1": 6}

    def test_counts_sum_to_shots(self) -> None:
        """Largest-remainder allocation keeps the total exactly equal to shots."""
        # Three equal thirds would each round to 33 (sum 99) with naive rounding.
        histogram = {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3}
        counts = IonQExecutor._histogram_to_counts(histogram, 100, 2)
        assert sum(counts.values()) == 100
        # The leftover shot goes to a single bin: counts are 34/33/33 in some order.
        assert sorted(counts.values()) == [33, 33, 34]

    def test_counts_sum_to_shots_many_states(self) -> None:
        """Totals stay exact even when many bins each carry a fractional part."""
        histogram = {str(i): 1 / 7 for i in range(7)}
        counts = IonQExecutor._histogram_to_counts(histogram, 1000, 3)
        assert sum(counts.values()) == 1000

    def test_empty_histogram(self) -> None:
        """An empty histogram yields no counts."""
        assert IonQExecutor._histogram_to_counts({}, 100, 2) == {}


class TestExtractHistogram:
    """Tests for normalizing the results-endpoint response shape."""

    def test_direct_mapping(self) -> None:
        """A bare histogram mapping is returned as-is."""
        assert IonQExecutor._extract_histogram({"0": 0.5, "1": 0.5}) == {"0": 0.5, "1": 0.5}

    def test_histogram_wrapped(self) -> None:
        """A {'histogram': {...}} shape is unwrapped."""
        assert IonQExecutor._extract_histogram({"histogram": {"0": 1.0}}) == {"0": 1.0}

    def test_data_histogram_wrapped(self) -> None:
        """A {'data': {'histogram': {...}}} shape is unwrapped."""
        payload = {"data": {"histogram": {"3": 1.0}}}
        assert IonQExecutor._extract_histogram(payload) == {"3": 1.0}

    def test_probabilities_key(self) -> None:
        """A {'probabilities': {...}} shape is unwrapped."""
        assert IonQExecutor._extract_histogram({"probabilities": {"2": 1.0}}) == {"2": 1.0}


class TestCircuitToQasm:
    """Tests for the to_qiskit() -> QASM conversion path."""

    def test_uses_to_qiskit_and_dumps_qasm(self) -> None:
        """Conversion goes through to_qiskit() and returns QASM + qubit count."""
        circuit = bell_state()
        qasm, num_qubits = IonQExecutor._circuit_to_qasm(circuit)
        assert "OPENQASM" in qasm
        assert num_qubits == 2


class TestAuthHeaders:
    """Tests for API key resolution."""

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No config key and no env var raises ValueError."""
        monkeypatch.delenv("IONQ_API_KEY", raising=False)
        executor = IonQExecutor(IonQExecutorConfig())
        with pytest.raises(ValueError, match="IonQ API key not found"):
            executor._auth_headers()

    def test_config_key_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config key takes effect."""
        monkeypatch.delenv("IONQ_API_KEY", raising=False)
        executor = IonQExecutor(IonQExecutorConfig(api_key="cfg-key"))
        assert executor._auth_headers() == {"Authorization": "apiKey cfg-key"}

    def test_env_key_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variable is used when config key is absent."""
        monkeypatch.setenv("IONQ_API_KEY", "env-key")
        executor = IonQExecutor(IonQExecutorConfig())
        assert executor._auth_headers() == {"Authorization": "apiKey env-key"}


class TestIonQExecutorExecute:
    """Tests for IonQExecutor.execute with a stubbed session."""

    @pytest.mark.asyncio
    async def test_execute_returns_result(self) -> None:
        """A completed job yields counts from the histogram."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST" and url.endswith("/jobs"):
                return {"id": "job-123", "status": "submitted"}
            if method == "GET" and url.endswith("/jobs/job-123"):
                return {
                    "status": "completed",
                    "data": {"histogram": {"0": 0.5, "3": 0.5}},
                    "execution_time": 42,
                }
            raise AssertionError(f"unexpected request: {method} {url}")

        config = IonQExecutorConfig(target="simulator", api_key="k")
        executor = IonQExecutor(config, session=_make_session(router))
        circuit = bell_state()

        with _patch_qasm(num_qubits=2):
            result = await executor.execute(circuit, shots=1000)

        assert isinstance(result, ExecutionResult)
        assert result.counts == {"00": 500, "11": 500}
        assert result.shots == 1000
        assert result.backend == "simulator"
        assert result.metadata["job_id"] == "job-123"
        assert result.metadata["provider"] == "ionq"
        assert result.probabilities == {"00": 0.5, "11": 0.5}

    @pytest.mark.asyncio
    async def test_execute_submits_qasm_input(self) -> None:
        """The submission payload carries the QASM input format."""
        captured: dict = {}

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                captured.update(kwargs.get("json", {}))
                return {"id": "job-1", "status": "submitted"}
            return {"status": "completed", "data": {"histogram": {"0": 1.0}}}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        circuit = Circuit().h(0)
        with _patch_qasm(num_qubits=1):
            await executor.execute(circuit, shots=10)

        assert captured["input"]["format"] == "qasm"
        assert "data" in captured["input"]
        assert captured["target"] == "simulator"
        assert captured["shots"] == 10

    @pytest.mark.asyncio
    async def test_execute_uses_results_endpoint_when_no_inline_histogram(self) -> None:
        """Falls back to /results and unwraps a wrapped histogram shape."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                return {"id": "job-9", "status": "submitted"}
            if url.endswith("/jobs/job-9/results"):
                # Wrapped shape — must be unwrapped, not passed through verbatim.
                return {"data": {"histogram": {"0": 0.5, "3": 0.5}}}
            return {"status": "completed"}  # job object has no inline histogram

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        circuit = bell_state()
        with _patch_qasm(num_qubits=2):
            result = await executor.execute(circuit, shots=1000)

        assert result.counts == {"00": 500, "11": 500}

    @pytest.mark.asyncio
    async def test_execute_preserves_zero_execution_time(self) -> None:
        """A reported execution_time of 0 is preserved, not treated as missing."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                return {"id": "job-0", "status": "submitted"}
            return {
                "status": "completed",
                "data": {"histogram": {"0": 1.0}},
                "execution_time": 0,
            }

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        circuit = Circuit().h(0)
        with _patch_qasm(num_qubits=1):
            result = await executor.execute(circuit, shots=10)

        assert result.execution_time_ms == 0

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_wall_time_when_unreported(self) -> None:
        """When IonQ omits execution_time, wall time (a positive float) is used."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                return {"id": "job-w", "status": "submitted"}
            return {"status": "completed", "data": {"histogram": {"0": 1.0}}}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        circuit = Circuit().h(0)
        with _patch_qasm(num_qubits=1):
            result = await executor.execute(circuit, shots=10)

        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_polls_until_complete(self) -> None:
        """Execute keeps polling while the job is still running."""
        calls = {"n": 0}

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                return {"id": "job-xyz", "status": "submitted"}
            calls["n"] += 1
            if calls["n"] < 3:
                return {"status": "running"}
            return {"status": "completed", "data": {"histogram": {"0": 1.0}}}

        config = IonQExecutorConfig(api_key="k", poll_interval_seconds=0.0)
        executor = IonQExecutor(config, session=_make_session(router))
        circuit = Circuit().h(0)

        with _patch_qasm(num_qubits=1):
            result = await executor.execute(circuit, shots=100)

        assert calls["n"] == 3
        assert result.counts == {"0": 100}

    @pytest.mark.asyncio
    async def test_execute_records_job_id(self) -> None:
        """The submitted job id is tracked for cancellation."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                return {"id": "track-me", "status": "submitted"}
            return {"status": "completed", "data": {"histogram": {"0": 1.0}}}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        circuit = Circuit().h(0)
        with _patch_qasm(num_qubits=1):
            await executor.execute(circuit, shots=10)

        assert executor._current_job_id == "track-me"

    @pytest.mark.asyncio
    async def test_execute_applies_noise_model(self) -> None:
        """A simulator noise model is included in the submission payload."""
        captured: dict = {}

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                captured.update(kwargs.get("json", {}))
                return {"id": "job-1", "status": "submitted"}
            return {"status": "completed", "data": {"histogram": {"0": 1.0}}}

        config = IonQExecutorConfig(api_key="k", target="simulator", noise_model="aria-1")
        executor = IonQExecutor(config, session=_make_session(router))
        circuit = Circuit().h(0)
        with _patch_qasm(num_qubits=1):
            await executor.execute(circuit, shots=10)

        assert captured["noise"] == {"model": "aria-1"}

    @pytest.mark.asyncio
    async def test_execute_times_out(self) -> None:
        """A perpetually-running job raises TimeoutError when timeout_seconds is set."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                return {"id": "job-t", "status": "submitted"}
            return {"status": "running"}  # never reaches a terminal state

        config = IonQExecutorConfig(
            api_key="k", poll_interval_seconds=0.0, timeout_seconds=0.05
        )
        executor = IonQExecutor(config, session=_make_session(router))
        circuit = Circuit().h(0)
        with _patch_qasm(num_qubits=1):
            with pytest.raises(asyncio.TimeoutError):
                await executor.execute(circuit, shots=10)

    @pytest.mark.asyncio
    async def test_execute_failed_job_raises(self) -> None:
        """A failed job raises RuntimeError with the IonQ message."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            if method == "POST":
                return {"id": "bad-job", "status": "submitted"}
            return {"status": "failed", "failure": {"error": "boom"}}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        circuit = Circuit().h(0)
        with _patch_qasm(num_qubits=1):
            with pytest.raises(RuntimeError, match="boom"):
                await executor.execute(circuit, shots=10)


class TestRequestHeaders:
    """Tests for header handling in the internal request helper."""

    @pytest.mark.asyncio
    async def test_caller_headers_merged_with_auth(self) -> None:
        """Caller-supplied headers merge with auth headers (no TypeError)."""
        captured: dict = {}

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            captured.update(kwargs)
            return {"ok": True}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        await executor._request("GET", "/jobs", headers={"X-Custom": "1"})

        assert captured["headers"]["Authorization"] == "apiKey k"
        assert captured["headers"]["X-Custom"] == "1"


class TestIonQExecutorCancel:
    """Tests for IonQExecutor.cancel."""

    @pytest.mark.asyncio
    async def test_cancel_success(self) -> None:
        """Cancel returns True on success."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            assert method == "PUT"
            assert url.endswith("/jobs/job-1/status/cancel")
            return {}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        assert await executor.cancel("job-1") is True

    @pytest.mark.asyncio
    async def test_cancel_failure(self) -> None:
        """Cancel returns False when the request fails."""
        session = MagicMock()
        session.request.side_effect = Exception("network down")
        executor = IonQExecutor(IonQExecutorConfig(api_key="k"), session=session)
        assert await executor.cancel("job-1") is False


class TestIonQExecutorGetStatus:
    """Tests for IonQExecutor.get_status."""

    @pytest.mark.asyncio
    async def test_available_maps_to_online(self) -> None:
        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            return {"status": "available", "jobs_queued": 4, "average_queue_time": 120}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        status = await executor.get_status()
        assert status.status == "online"
        assert status.queue_depth == 4
        assert status.queue_time_seconds == 120

    @pytest.mark.asyncio
    async def test_zero_queue_depth_preserved(self) -> None:
        """A queue_depth of 0 (empty queue) is reported, not treated as missing."""

        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            return {"status": "available", "queue_depth": 0, "jobs_queued": 5}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        status = await executor.get_status()
        assert status.queue_depth == 0

    @pytest.mark.asyncio
    async def test_unavailable_maps_to_offline(self) -> None:
        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            return {"status": "unavailable"}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        status = await executor.get_status()
        assert status.status == "offline"

    @pytest.mark.asyncio
    async def test_calibrating_maps_to_maintenance(self) -> None:
        def router(method, url, **kwargs):  # noqa: ANN001, ANN202
            return {"status": "calibrating"}

        executor = IonQExecutor(
            IonQExecutorConfig(api_key="k"), session=_make_session(router)
        )
        status = await executor.get_status()
        assert status.status == "maintenance"

    @pytest.mark.asyncio
    async def test_error_returns_maintenance(self) -> None:
        session = MagicMock()
        session.request.side_effect = Exception("API error")
        executor = IonQExecutor(IonQExecutorConfig(api_key="k"), session=session)
        status = await executor.get_status()
        assert status.status == "maintenance"
        assert status.queue_depth is None


class TestIonQExecutorMisc:
    """Miscellaneous IonQExecutor behavior."""

    def test_executor_name(self) -> None:
        """Executor name property works."""
        executor = IonQExecutor(IonQExecutorConfig(api_key="k"))
        assert executor.name == "IonQExecutor"

    @pytest.mark.asyncio
    async def test_non_circuit_raises_type_error(self) -> None:
        """Passing a non-Circuit reuses the base validation error."""
        executor = IonQExecutor(IonQExecutorConfig(api_key="k"))
        with pytest.raises(TypeError, match="Expected a Marqov Circuit"):
            await executor.execute("not a circuit")


# Integration test template (requires real IonQ credentials, not run in CI):
#
# @pytest.mark.integration
# @pytest.mark.asyncio
# async def test_execute_on_ionq_simulator() -> None:
#     """Execute on the IonQ cloud simulator (requires IONQ_API_KEY)."""
#     config = IonQExecutorConfig(target="simulator")  # key from IONQ_API_KEY
#     executor = IonQExecutor(config)
#     result = await executor.execute(bell_state(), shots=100)
#     assert set(result.counts).issubset({"00", "11"})
