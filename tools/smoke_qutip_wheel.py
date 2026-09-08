"""Check a built wheel in a fresh environment (requires uv on PATH).

Run: .venv/bin/python tools/smoke_qutip_wheel.py /absolute/path/marqov-X.Y.Z-py3-none-any.whl
Dependencies resolve only from real PyPI. Nothing is published.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from email.parser import Parser
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv must be installed and available on PATH")
    with ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            parser.error("expected one distribution metadata file")
        version = Parser().parsestr(archive.read(metadata[0]).decode())["Version"]
    example = Path(__file__).resolve().parents[1] / "examples" / "qutip_decay.py"
    with tempfile.TemporaryDirectory(prefix="marqov-qutip-smoke-") as directory:
        root = Path(directory)
        env = root / "venv"
        subprocess.run([uv, "venv", "--python", sys.executable, str(env)], check=True)
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        def install(requirement):
            subprocess.run(
                [uv, "--no-config", "pip", "install", "--python", str(python),
                 "--default-index", "https://pypi.org/simple", requirement],
                check=True, cwd=root,
                env={k: v for k, v in os.environ.items()
                     if not k.startswith(("UV_", "PIP_"))},
            )

        install(str(wheel))
        subprocess.run(
            [str(python), "-I", "-c",
             "import importlib.util; assert importlib.util.find_spec('qutip') is None"],
            check=True, cwd=root,
        )
        install(str(wheel) + "[qutip]")
        subprocess.run(
            [str(python), "-I", "-c",
             "import marqov, sys; from pathlib import Path; "
             "assert marqov.__version__ == sys.argv[1]; "
             "assert Path(marqov.__file__).is_relative_to(Path(sys.prefix)); "
             "import qutip", version], check=True, cwd=root,
        )
        script = root / "qutip_decay.py"
        shutil.copyfile(example, script)
        output = subprocess.check_output([str(python), "-I", str(script)], text=True, cwd=root)
        payload = json.loads(output)
        assert payload["result_type"] == "open-system-dynamics"
        assert payload["schema_version"] == 1
        assert payload["times"] == [i / 2 for i in range(11)]
        values = payload["observables"]["sigma_z"]
        assert len(values) == 11
        for time, value in zip(payload["times"], values):
            assert math.isclose(value, 2 * math.exp(-0.4 * time) - 1, abs_tol=1e-5)
    print(f"marqov {version}: base excludes QuTiP; extra installs it; wheel example matches analytic decay.")


if __name__ == "__main__":
    main()
