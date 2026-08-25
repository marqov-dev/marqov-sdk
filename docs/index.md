# Marqov SDK documentation

Marqov is a Python SDK for building quantum circuits, running them on local or cloud
backends, and composing hybrid quantum-classical workflows.

This documentation starts with a local Bell-state example that runs without cloud
credentials, then exposes the public Python API from the package docstrings.

## Build the docs

```bash
pip install -e .
pip install -r docs/requirements.txt
mkdocs serve
```

## What to read first

- [Bell state with local execution](examples/bell-state-local.md) shows the smallest
  end-to-end circuit workflow.
- [Circuit API](api/circuits.md) documents the backend-agnostic circuit builder.
- [Executor API](api/executors.md) documents local and backend executor interfaces.
