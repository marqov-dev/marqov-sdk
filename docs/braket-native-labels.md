# Native Braket labels and submission intent

`Circuit.to_braket_native()` returns a Braket circuit in the existing Rigetti
Rx/Rz/CZ/XY basis. It preserves source labels, each gate's target order and
operation order. It does not translate gates, compact labels or insert idle-wire
padding. An Rx on label 7 remains an Rx on label 7. The normal `to_braket()`
conversion remains unchanged and compacts logical labels.

Only exact QuantumFlow Rx/Rz/CZ/XY gates are accepted. Qubit labels must be
nonnegative integers (excluding booleans); two-qubit gates require distinct
labels. Angles must be finite real numeric values representable as float;
symbolic parameters, unsupported gates and empty circuits raise `ValueError`.
Rx/Rz angles use radians. QuantumFlow XY(t) is exported with Braket angle
`-2*pi*t`, matching the published marqov-quantumflow 1.0.1 convention. Use the SDK's
required dependency (==1.0.1); older QuantumFlow XY import bugs are not repaired
by this export. Importing external circuits still relies on their importer.

## Explicit submission

Both public submission methods accept keyword-only `preserve_qubit_labels=False`:

```python
native = Circuit().rx(0.3, 7).rz(-0.7, 1).cz(7, 1)
result = await executor.execute(
    native, shots=1000, preserve_qubit_labels=True,
    verbatim=True, disable_qubit_rewiring=True,
)
# MarqovDevice.run accepts the same options and returns counts as before.
```

The opt-in selects native export and requires both `verbatim=True` and
`disable_qubit_rewiring=True`. The configured ARN must structurally identify an
AWS Braket Rigetti QPU. Other providers, simulators and non-Braket paths refuse
this opt-in. ARN checking establishes configured provider/resource intent only;
it does not discover a device or certify availability, topology, native gate
capability, allowed angles or physical addressing. This basis is Rigetti scoped,
not a vendor-generic native exporter. All recognized flags must be Python bools;
invalid options and native circuits refuse before provider initialization or
submission. No provider discovery is performed by the validator.

Both entry points share native preparation and verbatim gate validation. A
verbatim box surrounds the native instructions; explicit measurement is not
accepted, and Braket measurement remains implicit through shots. Both forward an
explicit `disable_qubit_rewiring` value unchanged. Without that option the
executor continues omitting it (provider default), while `MarqovDevice.run`
continues forwarding False. With `preserve_qubit_labels=False`, ordinary logical
conversion, ordinary verbatim validation and result return types remain
unchanged. Non-Braket legacy backend kwargs handling is unchanged; physical
intent can no longer be silently ignored there. Unrecognized kwargs remain the
separate marqov-sdk#77 audit.

Preserved source/provider labels and submitted directives are not executed
hardware mapping evidence. The result provenance introduced in marqov-sdk#206
continues to report `physical_mapping_status="unqualified"`. Capability checks,
retained executed mapping, physical timing and pulse overlap need separate
qualification evidence. `MarqovDevice.run` still returns counts alone.

## Offline qualification seam

Run `python -m pytest -q tests/test_braket_native.py` in the SDK environment with
published marqov-quantumflow 1.0.1. Tests use real Braket circuits and fake
provider tasks: replace `BraketExecutor._get_device` with an `AsyncMock` and
`MarqovDevice._get_provider_device` with a fake provider, then inspect
`device.run` arguments. No AWS/provider call is needed. Sparse [7,1] targets,
complex native matrices, verbatim directives, forwarding, default compatibility
and rejection before initialization are covered. This is offline source
qualification, not a hosted or hardware result.

Global logical-register width and sparse counts policy in marqov-sdk#154 remain
separate. This change contributes to the Braket parity audit marqov-sdk#66 but
does not close its broader behavior audit.
