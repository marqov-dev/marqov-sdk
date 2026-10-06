"""Benchmark reporting distinguishes measured zero time from missing timing."""

import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture(params=["harness.py", "raw_sdk/harness.py"])
def harness(request, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / request.param
    name = "benchmark_harness_" + request.param.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("times, expected", [
    ([0.0, 2.0, 4.0, None], ("0.000", "4.000", "2.000")),
    ([0.0, 0.0, 0.0], ("0.000", "0.000", "0.000")),
    ([None, None], None),
])
def test_quantum_time_summary(harness, capsys, times, expected):
    results = [harness.BenchmarkResult(
        name="timing", run_number=i + 1, wall_time_seconds=1.0,
        quantum_time_seconds=timing, result_data={}, timestamp="unused", backend="local",
    ) for i, timing in enumerate(times)]
    harness.print_summary(results)
    output = capsys.readouterr().out
    header = "Quantum time (seconds):"
    if expected is None:
        assert header not in output
    else:
        assert header in output
        quantum = output.split(header, 1)[1]
        for label, value in zip(("Min", "Max", "Mean"), expected):
            assert f"{label}: {value}" in quantum


def test_run_and_save_results_record_utc(harness, tmp_path):
    results = harness.run_benchmark("timing", lambda: ({"answer": 42}, 0.0), runs=1,
                                    backend="local")
    timestamp = datetime.fromisoformat(results[0].timestamp)
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)
    path = harness.save_results(results, tmp_path / "results", suffix="_test")
    assert re.fullmatch(r"timing_\d{8}_\d{6}_test\.json", path.name)
    saved = json.loads(path.read_text())
    assert saved[0]["timestamp"] == results[0].timestamp
    assert saved[0]["quantum_time_seconds"] == 0.0
    assert saved[0]["result_data"] == {"answer": 42}
