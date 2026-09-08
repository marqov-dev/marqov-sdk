# Workflow capture for execution adapters

Calling an SDK `@workflow` executes its Python body to build a graph. Task calls
record nodes and serialize their functions; they normally do not execute task
bodies. Importing source, constructing the graph and serializing arbitrary Python
objects can all execute user code. Hosted adapters must perform these operations
inside their isolated execution environment, not in a privileged controller.

`WorkflowDispatch.capture()` exports an already-built graph and its return
template without rerunning the workflow or submitting tasks:

```python
from marqov import task, workflow

@task
def double(value):
    return value * 2

@workflow
def calculate(value):
    result = double(value)
    return {"answer": result, "again": (result, "units"), "constant": 0.25}

captured = calculate(3).capture()
node_id = next(iter(captured.graph.nodes))
# After an adapter has actually executed the task and obtained its result:
assert captured.resolve_output({node_id: 6}) == {
    "answer": 6, "again": (6, "units"), "constant": 0.25,
}
```

`WorkflowCapture` and `TaskResultReference` are exported from `marqov.workflows`.
The capture contains the existing `TransportGraph`, including task identity,
function bytes, positional/keyword arguments, dependencies and **requested**
configuration. Its `output` replaces task proxies with `TaskResultReference` and
preserves plain lists, tuples, dictionaries, constants and repeated references.
`resolve_output` substitutes completed node results; missing results raise
`KeyError`. A constant-only return, including `None`, is retained.

This is an in-process representation, **not a wire format**. The graph is mutable;
opaque leaf values are shared rather than copied, and calling capture later reads
the retained result object at that time. Adapters must select a versioned artifact
codec and compatible runtime, validate dependencies and bounds, and retain their
own stable execution identity. Random SDK node IDs identify this capture; running
the workflow again is a new capture, not recovery of an accepted task attempt.
Graph/function/argument/result data must not be assumed safe to deserialize in a
trusted process. Neither requested configuration nor a capture grants authority.

Argument dependency extraction now traverses list/tuple/dictionary values as
recursively as SDK argument serialization. Repeated predecessors appear once,
in first-occurrence order. Cyclic argument containers raise `ValueError`.

Capture supports references in plain lists, tuples and dictionary values.
Container subclasses, cyclic return containers, task references used directly as
dictionary keys, nested workflow-dispatch outputs and awaitable outputs are
explicitly rejected. Other Python objects, including numerical arrays, are opaque
leaves; references hidden inside those objects are unsupported. Adapters must
choose which opaque types their artifact codec accepts. Capture of an async
workflow body is unsupported; ordinary async **task** functions can remain
executable material for an adapter that supports them.

An unresolved task proxy now raises `TypeError` if used as a boolean (for example
`if task_result:`). Its truth value cannot select a branch before the task runs.
This does not implement dynamic graph execution or all Python operations on
unresolved results.

The existing `run()`, `start()` and private legacy Temporal input builder retain
their transport/result behavior. They do not consume the new return template;
the shape-preserving contract is opt-in through `capture()`. A manually constructed
`WorkflowDispatch` without a retained result refuses capture. SDK core remains
independent of the optional hosted API client and requires no platform account to
construct a capture.
