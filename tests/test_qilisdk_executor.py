"""Tests for marqov.executors.qilisdk (Qilimanjaro qilisdk local simulators)."""

import pytest

qilisdk = pytest.importorskip("qilisdk")

from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.executors import ExecutionResult, ExecutorFactory, QiliSDKExecutor
from marqov.executors.qilisdk import QiliSDKExecutorConfig


class TestQiliSDKExecutor:
    """Tests for QiliSDKExecutor."""

    @pytest.mark.asyncio
    async def test_execute_returns_result(self) -> None:
        """Execute returns an ExecutionResult."""
        executor = QiliSDKExecutor()
        circuit = Circuit().h(0)
        result = await executor.execute(circuit, shots=100)

        assert isinstance(result, ExecutionResult)
        assert result.backend == "qilisdk-qilisim"
        assert result.shots == 100

    @pytest.mark.asyncio
    async def test_execute_bell_state(self) -> None:
        """Bell state produces expected measurement distribution."""
        executor = QiliSDKExecutor()
        circuit = bell_state()
        result = await executor.execute(circuit, shots=1000)

        assert set(result.counts.keys()).issubset({"00", "11"})
        assert sum(result.counts.values()) == 1000
        for count in result.counts.values():
            assert 400 < count < 600

    @pytest.mark.asyncio
    async def test_execute_ghz_state(self) -> None:
        """GHZ state on 3 qubits only produces all-0s or all-1s."""
        executor = QiliSDKExecutor()
        circuit = ghz_state(3)
        result = await executor.execute(circuit, shots=500)

        assert set(result.counts.keys()).issubset({"000", "111"})
        assert sum(result.counts.values()) == 500

    @pytest.mark.asyncio
    async def test_execute_all_supported_gates(self) -> None:
        """A circuit touching every canonical gate runs without error."""
        executor = QiliSDKExecutor()
        circuit = (
            Circuit()
            .h(0)
            .x(1)
            .y(0)
            .z(1)
            .s(0)
            .t(1)
            .rx(0.3, 0)
            .ry(0.6, 1)
            .rz(0.9, 0)
            .cnot(0, 1)
            .cz(0, 1)
            .swap(0, 1)
        )
        result = await executor.execute(circuit, shots=50)

        assert sum(result.counts.values()) == 50

    @pytest.mark.asyncio
    async def test_execute_metadata(self) -> None:
        """Execution includes simulator metadata."""
        executor = QiliSDKExecutor()
        circuit = Circuit().x(0)
        result = await executor.execute(circuit, shots=10)

        assert result.metadata["simulator"] == "qilisim"
        assert result.metadata["vendor"] == "Qilimanjaro"
        assert result.metadata["framework"] == "QiliSDK"
        assert result.metadata["access_path"] == "local"
        assert result.metadata["reproducibility"]["shots"] == 10

    @pytest.mark.asyncio
    async def test_execute_qutip_backend(self) -> None:
        """The qutip reference simulator produces the same-shaped result."""
        pytest.importorskip("qutip")

        executor = QiliSDKExecutor(QiliSDKExecutorConfig(simulator="qutip"))
        circuit = bell_state()
        result = await executor.execute(circuit, shots=200)

        assert result.backend == "qilisdk-qutip"
        assert set(result.counts.keys()).issubset({"00", "11"})
        assert sum(result.counts.values()) == 200

    @pytest.mark.asyncio
    @pytest.mark.parametrize("simulator", ["qilisim", "qutip"])
    async def test_provenance_matches_actual_engine_and_counts(self, simulator: str) -> None:
        import hashlib
        from importlib.metadata import version
        import json

        executor = QiliSDKExecutor(QiliSDKExecutorConfig(
            simulator=simulator, compute_provider="isolated-test-runtime"
        ))
        result = await executor.execute(Circuit().x(0).cz(0, 1), shots=32)
        assert result.counts == {"10": 32}
        assert result.metadata["compute_provider"] == "isolated-test-runtime"
        record = result.metadata["reproducibility"]
        assert record["packages"]["qilisdk"] == version("qilisdk")
        assert result.metadata["engine"] == (
            f"{type(executor._backend).__module__}.{type(executor._backend).__qualname__}"
        )
        assert record["counts_sha256"] == hashlib.sha256(
            json.dumps(result.counts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert record["seed"] is None
        assert "no hardware execution" in record["scope"]

    def test_unsupported_simulator_raises(self) -> None:
        """An unknown simulator name raises immediately, not a silent fallback."""
        with pytest.raises(ValueError, match="Unknown qilisdk simulator"):
            QiliSDKExecutor(QiliSDKExecutorConfig(simulator="not-a-real-backend"))  # type: ignore[arg-type]


class TestQiliSDKExecutorFactory:
    """Tests for creating QiliSDKExecutor via ExecutorFactory."""

    def test_create_qilisdk_executor_default(self) -> None:
        """Factory creates a QiliSim-backed executor by default."""
        executor = ExecutorFactory.create_executor(
            "qilisdk-qilisim", {"provider": "Qilimanjaro"}
        )
        assert isinstance(executor, QiliSDKExecutor)
        assert executor.config.simulator == "qilisim"

    def test_create_qilisdk_executor_qutip(self) -> None:
        """Factory honors an explicit qutip simulator selection."""
        pytest.importorskip("qutip")

        executor = ExecutorFactory.create_executor(
            "qilisdk-qutip", {"provider": "Qilimanjaro", "simulator": "qutip"}
        )
        assert isinstance(executor, QiliSDKExecutor)
        assert executor.config.simulator == "qutip"

    def test_qilimanjaro_in_supported_providers(self) -> None:
        """Qilimanjaro appears in the supported-providers list."""
        assert "Qilimanjaro" in ExecutorFactory.get_supported_providers()
        assert ExecutorFactory.is_provider_supported("Qilimanjaro")

# Direct SpeQtrum transport fixtures never contact a provider or use a keyring.
class TestSpeQtrumDirect:
    @staticmethod
    def config(**overrides):
        from marqov.executors.speqtrum import SpeQtrumExecutorConfig
        return SpeQtrumExecutorConfig(device_code="fixture-device", username="fixture-account",
                                     api_key="fixture-secret", **overrides)

    @staticmethod
    def wire(monkeypatch, *, fail=None, terminal="completed"):
        import base64
        import json

        from qilisdk.backends import QiliSim
        from qilisdk.functionals import DigitalPropagation
        from qilisdk.readout import Readout
        from qilisdk.utils.serialization import serialize

        from marqov.executors.speqtrum import SpeQtrumHTTPClient
        calls, saved = [], {}
        # Actual vendor typed sampling result fixes canonical asymmetric bit order.
        result = QiliSim().execute(DigitalPropagation(
            circuit=QiliSDKExecutor._to_qilisdk_circuit(Circuit().x(0).x(1).x(1))),
            Readout().with_sampling(nshots=32))

        class Response:
            status_code = 200
            def __init__(self, payload, status=200):
                self.payload, self.status_code = payload, status
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def iter_content(self, chunk_size):
                yield json.dumps(self.payload).encode()

        class Session:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def mount(self, prefix, adapter):
                assert adapter.max_retries.total == 0
            def request(self, method, url, **kwargs):
                assert self.trust_env is False
                assert kwargs['allow_redirects'] is False and kwargs['timeout'] > 0
                path = url.rsplit('/api/v1', 1)[1]
                calls.append((method, path))
                if path == '/authorisation-tokens':
                    assertion = json.loads(base64.urlsafe_b64decode(kwargs['json']['assertion']))
                    assert assertion['username'] == 'fixture-account'
                    return Response({'accessToken': 'fixture-token'})
                assert kwargs['headers']['Authorization'] == 'Bearer fixture-token'
                if path == '/devices':
                    return Response({'items': [{'code': 'fixture-device', 'nqubits': 2,
                                                'type': 'simulator', 'status': 'online'}]})
                if path == '/execute':
                    if fail == 'timeout':
                        import requests
                        raise requests.Timeout('provider secret/raw detail')
                    saved['payload'] = kwargs['json']['payload']
                    saved['kind'] = json.loads(saved['payload'])['type']
                    return Response({'id': 'bad' if fail == 'malformed' else 42}, status=401 if fail == '401' else 200)
                assert path == '/jobs/42'
                poll = sum(path == '/jobs/42' for _, path in calls)
                status = ('pending' if poll == 1 else 'completed') if terminal == 'device-change' else terminal
                device_id = 8 if terminal == 'device-change' and poll > 1 else 7
                return Response({'id': 42, 'device_id': device_id, 'status': status, 'payload': saved['payload'],
                                 'result': base64.b64encode(json.dumps({
                                     'type': saved['kind'], 'functional_result': serialize(result)
                                 }).encode()).decode()})
        monkeypatch.setattr('marqov.executors.speqtrum.requests.Session', Session)
        assert SpeQtrumHTTPClient.max_response_bytes > 0
        return calls

    @pytest.mark.asyncio
    async def test_direct_known_id_readback_and_asymmetric_counts(self, monkeypatch):
        from marqov.executors.speqtrum import SpeQtrumExecutor
        calls = self.wire(monkeypatch)
        executor = SpeQtrumExecutor(self.config())
        result = await executor.execute(Circuit().x(0).x(1).x(1), shots=32)
        assert result.counts == {'10': 32} and result.metadata['job_id'] == 42
        assert result.metadata['access_path'] == 'direct'
        assert 'fixture-secret' not in repr(executor.config)
        assert executor.last_job.job_id == 42
        replay = await executor.readback(executor.last_job)
        assert replay.counts == result.counts
        assert calls.count(('POST', '/execute')) == 1
        assert await executor.cancel('42') is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize('failure', ['401', 'timeout', 'malformed'])
    async def test_uncertain_send_is_not_replayed(self, monkeypatch, failure):
        from marqov.executors.speqtrum import SpeQtrumExecutionError, SpeQtrumExecutor
        calls = self.wire(monkeypatch, fail=failure)
        executor = SpeQtrumExecutor(self.config())
        with pytest.raises(SpeQtrumExecutionError) as error:
            await executor.execute(Circuit().x(0).x(1).x(1), shots=32)
        assert error.value.acceptance_unknown and error.value.job_id is None
        assert error.value.__cause__ is None
        assert 'provider secret' not in str(error.value)
        assert calls.count(('POST', '/execute')) == 1
        assert calls.count(('POST', '/authorisation-tokens')) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", ["timeout", "device-change"])
    async def test_terminal_failure_or_device_change_retains_known_job(self, monkeypatch, terminal):
        from marqov.executors.speqtrum import SpeQtrumExecutionError, SpeQtrumExecutor
        calls = self.wire(monkeypatch, terminal=terminal)
        executor = SpeQtrumExecutor(self.config(poll_interval_seconds=0.001))
        with pytest.raises(SpeQtrumExecutionError) as error:
            await executor.execute(Circuit().x(0).x(1).x(1), shots=32)
        assert error.value.job_id == 42 and not error.value.acceptance_unknown
        assert calls.count(('GET', '/jobs/42')) == (2 if terminal == 'device-change' else 1)

    @pytest.mark.parametrize('field', ['device_code', 'username', 'api_key'])
    def test_factory_remote_requires_explicit_identity(self, field):
        config = {'provider': 'Qilimanjaro', 'access_path': 'speqtrum',
                  'device_code': 'fixture-device', 'username': 'fixture-account', 'api_key': 'fixture-secret'}
        del config[field]
        with pytest.raises(ValueError):
            ExecutorFactory.create_executor('explicit-speqtrum', config)

    def test_remote_config_rejects_unsafe_endpoint_and_deadline(self):
        with pytest.raises(ValueError):
            self.config(api_url='http://example.com')
        with pytest.raises(ValueError):
            self.config(timeout_seconds=float('inf'))
        with pytest.raises(ValueError):
            ExecutorFactory.create_executor('explicit-speqtrum', {'provider': 'Qilimanjaro', 'access_path': 'remote'})

    @pytest.mark.parametrize('document', [
        '!function ZmFrZQ==',
        '!!python/object/apply:os.system [echo unsafe]',
        '!FunctionalResult &a {_readout_results: *a}',
        '!FunctionalResult\n_execution_time: 0\n_execution_time: 1',
        '[' * 20 + '0' + ']' * 20,
        'x' * (1024 * 1024 + 1),
    ])
    def test_provider_sampling_parser_refuses_executable_or_unbounded_yaml(self, document):
        from marqov.executors.speqtrum import _sampling_counts
        with pytest.raises(ValueError):
            _sampling_counts(document)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('override', [{'api_url': 'https://other.example/api/v1'}, {'audience': 'other'}])
    async def test_readback_is_bound_to_api_and_audience(self, monkeypatch, override):
        from marqov.executors.speqtrum import SpeQtrumExecutor
        calls = self.wire(monkeypatch)
        first = SpeQtrumExecutor(self.config())
        await first.execute(Circuit().x(0).x(1).x(1), shots=32)
        before = len(calls)
        with pytest.raises(ValueError):
            await SpeQtrumExecutor(self.config(**override)).readback(first.last_job)
        assert len(calls) == before

    @pytest.mark.asyncio
    async def test_analog_uses_exact_vendor_wire_and_known_result(self, monkeypatch):
        from qilisdk.analog import Hamiltonian, PauliX, Schedule

        from marqov.executors.speqtrum import SpeQtrumExecutor
        calls = self.wire(monkeypatch)
        h = Hamiltonian({(PauliX(0),): 1.0, (PauliX(1),): 1.0})
        schedule = Schedule.linear(h, h, total_time=1, dt=0.1)
        executor = SpeQtrumExecutor(self.config())
        result = await executor.execute_analog(schedule, shots=32)
        assert result.counts == {'10': 32}
        assert executor.last_job.execute_type == 'analog_evolution'
        assert calls.count(('POST', '/execute')) == 1
