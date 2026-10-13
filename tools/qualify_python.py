"""Qualify one integration per clean environment; retain logs and exact resolutions.

Run from a checkout with a built wheel. No credentials or live provider jobs.
Output directories must be new; scratch environments are deleted after each leg.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

PROFILES = {
    "core": [
        "tests/test_local_sampling.py",
        "tests/test_workflow_outputs.py",
        "tests/test_transport_graph_integrity.py",
    ],
    "braket": ["tests/test_braket_modifiers.py", "tests/test_circuit_serialization.py"],
    "qiskit": [],
    "cirq": [],
    "pennylane": ["tests/test_lightning_executor.py"],
    "qutip": ["tests/qutip"],
    "pytket": [],
    "quantinuum": [],
    "openqasm": [],
    "ibm": ["tests/test_ibm_channel.py"],
    "azure": [
        "tests/test_azure_dependencies.py",
        "tests/test_azure_bitorder.py",
        "tests/test_azure_timeout.py",
        "tests/test_azure_cirq_timeout.py",
    ],
    "ionq": ["tests/test_ionq_v04.py", "tests/test_ionq_submission.py"],
    "pyquil": ["tests/test_pyquil_modifiers.py"],
    "rigetti": [
        "tests/test_rigetti_executor.py",
        "tests/test_rigetti_readout.py",
        "tests/test_rigetti_timeout.py",
    ],
    "lightning-kokkos": ["tests/test_lightning_executor.py"],
    "cudaq": ["tests/test_cudaq_executor.py"],
    "qilisdk": [
        "tests/test_qilisdk_executor.py",
        "tests/test_qilisdk_async.py",
        "tests/test_qilisdk_options.py",
        "tests/test_qilisdk_seed.py",
        "tests/test_qilisdk_analog_executor.py",
    ],
    "all": [],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", default="3.13")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = args.wheel.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "PYTHONSAFEPATH": "1",
        "MARQOV_REQUIRE_INSTALLED": "1",
        "PYTHONUNBUFFERED": "1",
    }
    report = {
        "record_version": 1,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "sdk_source_commit": args.source_commit,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "os": platform.system(),
        "machine": platform.machine(),
        "requested_python": args.python,
        "live_provider_execution": False,
        "results": [],
    }

    def command(argv, log, cwd=root):
        with log.open("w") as stream:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
        return completed.returncode

    for profile in args.profiles:
        leg = output / profile
        leg.mkdir()
        venv = leg / "venv"
        result = {"profile": profile, "install": "not_run", "probe": "not_run", "tests": "not_run"}
        report["results"].append(result)
        print(f"Qualifying {profile}", flush=True)
        try:
            code = command(["uv", "venv", "--python", args.python, str(venv)], leg / "venv.log")
            if code:
                raise RuntimeError(f"venv creation failed: {code}")
            python = str(venv / "bin/python")
            packages = [str(wheel) if profile in {"core", "qilisdk"} else f"{wheel}[{profile}]"]
            if profile == "qilisdk":
                packages += ["qilisdk==0.3.0", "qutip-qip>=0.4.0"]
            # Minimal test tools, deliberately not [dev] (it installs other integrations).
            packages += ["pytest>=8", "pytest-asyncio>=0.23", "pyyaml>=6"]
            code = command(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--index-url",
                    "https://pypi.org/simple",
                    *packages,
                ],
                leg / "install.log",
            )
            result["install"] = "passed" if code == 0 else "failed"
            if code == 0:
                command(
                    [
                        python,
                        "-c",
                        'import importlib.metadata as m, json; print(json.dumps(sorted([{"name": d.metadata["Name"], "version": d.version} for d in m.distributions()], key=lambda d: d["name"].lower())))',
                    ],
                    leg / "packages.json",
                    leg,
                )
                result["packages"] = json.loads((leg / "packages.json").read_text())
                command([python, "-c", "import sys; print(sys.version)"], leg / "python.log", leg)
                result["python"] = (leg / "python.log").read_text().strip()
                check = command(
                    ["uv", "pip", "check", "--python", python], leg / "dependency-check.log"
                )
                result["dependency_check"] = "passed" if check == 0 else "failed"
                if profile != "all":
                    code = command(
                        [python, str(root / "tools/python_compatibility_probe.py"), profile],
                        leg / "probe.log",
                        leg,
                    )
                    result["probe"] = "passed" if code == 0 else "failed"
                    if code != 0:
                        result["probe_error"] = (leg / "probe.log").read_text()[-4000:]
                if PROFILES[profile]:
                    code = command(
                        [
                            python,
                            "-m",
                            "pytest",
                            *PROFILES[profile],
                            "-q",
                            "--import-mode=importlib",
                            f"--junitxml={leg / 'tests.xml'}",
                        ],
                        leg / "tests.log",
                    )
                    result["tests"] = "passed" if code == 0 else "failed"
                    if (leg / "tests.xml").exists():
                        suites = ET.parse(leg / "tests.xml").getroot().iter("testsuite")
                        totals = {key: 0 for key in ["tests", "failures", "errors", "skipped"]}
                        for suite in suites:
                            for key in totals:
                                totals[key] += int(suite.get(key, "0"))
                        result["test_totals"] = totals
                    result["test_summary"] = (leg / "tests.log").read_text()[-1500:]
            else:
                result["install_error"] = (leg / "install.log").read_text()[-5000:]
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            result["runner_error"] = str(error)
        finally:
            if venv.exists():
                shutil.rmtree(venv)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{profile}: install={result['install']}, probe={result['probe']}, tests={result['tests']}",
            flush=True,
        )

    failed = any(
        row["install"] != "passed"
        or row["probe"] == "failed"
        or row["tests"] == "failed"
        or row.get("runner_error")
        or row.get("dependency_check") == "failed"
        for row in report["results"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
