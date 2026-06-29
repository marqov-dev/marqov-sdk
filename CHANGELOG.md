# Changelog

All notable changes to the `marqov` SDK are documented here. This project follows
[Semantic Versioning](https://semver.org/). While on `0.x`, the public API may
still change between minor versions; `1.0.0` is reserved for the first API-stable
release.

## [0.2.0] — 2026-06-29

First public release on PyPI.

### Added
- Public release of the `marqov` SDK: build quantum circuits and run them across
  multiple hardware backends (AWS Braket, IBM Quantum, Azure Quantum, IonQ,
  Rigetti, Quantinuum, and local simulation) behind one API.
- Circuit interop helpers (import/export with Qiskit, Cirq, PennyLane, pytket,
  and pyQuil) and workflow/task decorators for composing multi-step programs.

### Changed
- The circuit IR / transpilation dependency is now the published
  [`marqov-quantumflow`](https://pypi.org/project/marqov-quantumflow/) package
  instead of a git URL, which is what makes `pip install marqov` possible.
- The package version is single-sourced from `marqov/__init__.py` (`__version__`)
  via hatchling, so the distribution metadata, `marqov.__version__`, and
  `marqov --version` always agree.

### Known limitations
- `@task`/`@workflow` serialize the decorated function with `cloudpickle` at
  decoration time. When the function is a local/closure (not module-level) and a
  large set of heavy backends is imported in the same process, this can overflow
  into a `RecursionError`. Module-level task functions are unaffected. A fix that
  defers/removes the function serialization is planned.

[0.2.0]: https://pypi.org/project/marqov/0.2.0/
