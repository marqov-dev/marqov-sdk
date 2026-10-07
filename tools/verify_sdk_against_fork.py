"""Verify installed SDK/fork metadata and RECORD ownership from a neutral cwd."""
import importlib.metadata as md
import sysconfig
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def verify_installed() -> None:
    sdk = md.distribution("marqov")
    requirements = [Requirement(r) for r in sdk.requires or []]
    pins = [r for r in requirements if canonicalize_name(r.name) == "marqov-quantumflow"]
    assert len(pins) == 1, f"expected one fork requirement, got {pins}"
    specs = list(pins[0].specifier)
    assert len(specs) == 1 and specs[0].operator == "==" and "*" not in specs[0].version, \
        f"expected exact fork pin, got {pins[0]}"
    fork = md.distribution("marqov-quantumflow")
    assert fork.version == specs[0].version, \
        f"expected marqov-quantumflow {specs[0].version}, got {fork.version}"
    try:
        upstream = md.version("quantumflow")
    except md.PackageNotFoundError:
        pass
    else:
        raise AssertionError(f"upstream quantumflow is also installed ({upstream})")

    import quantumflow

    print("Checking installed SDK and QuantumFlow imports...")
    import marqov

    sites = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
    for module, dist in ((marqov, sdk), (quantumflow, fork)):
        path = Path(module.__file__).resolve()
        assert any(path.is_relative_to(site) for site in sites), \
            f"{module.__name__} not from site-packages: {path}"
        owned = {Path(dist.locate_file(file)).resolve() for file in dist.files or []}
        assert path in owned, f"{path} is not owned by {dist.metadata['Name']} RECORD"
        print(f"OK: {module.__name__} -> {path}")
    from marqov.circuits import Circuit, bell_state
    Circuit().h(0).cnot(0, 1)
    bell_state()
    print(f"OK: installed SDK uses marqov-quantumflow {fork.version}")


if __name__ == "__main__":
    verify_installed()
