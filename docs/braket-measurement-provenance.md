# Braket measurement provenance

`BraketExecutor.execute()` adds a JSON-compatible record at
`result.metadata["measurement_provenance"]`. It describes evidence retained in
`result.raw_result`; it does not change counts, resample results, or submit an
additional provider request. `MarqovDevice.run()` still returns counts alone and
cannot supply this provenance contract.

| Field | Type and meaning |
|---|---|
| `protocol_version` | Literal `marqov.braket-measurement-provenance/v1` |
| `count_origin` | `provider_measurements`, `provider_counts`, `probability_derived`, or `unknown` |
| `requested_shots` | Integer requested in this execution |
| `observed_shots` | Nonnegative integer from genuine measurement rows or explicitly provider-origin counts; `null` for synthetic/unknown evidence |
| `provider_reported_shots` | Nonnegative integer from Braket task metadata, or `null`; this may describe requested shots, not actual returned observations |
| `measured_qubits` | Ordered list of unique nonnegative provider wire IDs, or `null` |
| `bitstring_order` | `measured_qubits_left_to_right` when validated; otherwise `unknown` |
| `raw_shot_eligible` | Boolean: trustworthy raw count evidence with consistent measured-wire order and a positive observed total no greater than requested |
| `ineligibility_reason` | `null` if eligible; otherwise a reason listed below |
| `physical_mapping_status` | Literal `unqualified` |

For example, `measured_qubits=[7,1]` and count key `"10"` means the first
character measures provider wire 7 and the second measures provider wire 1.
There is no sorting, reversal, padding or inference of physical hardware labels.
This does not prove a submitted SDK wire retained its identity through conversion
or provider rewiring. Physical mapping needs a separate qualified mapping and
retained evidence reference before physical-qubit inference.

## Evidence rules

Braket's `measurements_copied_from_device=True` identifies actual provider rows.
Those rows must be a binary integer/boolean matrix whose histogram agrees with
returned counts and whose column count matches the measured-wire order.
`measurement_counts_copied_from_device=True` can identify explicit provider
counts; keys, values and measured-wire width are validated. Origin flags are
provider SDK evidence, not an independent cryptographic attestation.

Counts derived from genuine provider rows are valid raw observations even when
`measurement_counts_copied_from_device=False`: that flag can mean Braket merely
aggregated the rows. Conversely, Braket can create nonempty counts from provider
probabilities, with `measurement_probabilities_copied_from_device=True` and no
copied measurement rows. Those counts, and Marqov's own probability allocation,
are `probability_derived` and never raw-shot eligible. Missing origin evidence is
`unknown`; count presence or a matching sum does not establish observations.

Genuine partial returns retain their rows and actual observed total. They may be
raw-shot eligible but do not fulfill the requested acquisition. Consumers must
apply their own completeness rule using requested and observed shots. The legacy
`ExecutionResult.shots` continues to contain requested shots for compatibility;
use the provenance record for the observed total. QMT may reject partial returns
in addition to refusing synthetic/unknown counts.

Refusal reasons are `unknown_origin`, `probability_derived`, `invalid_counts`,
`invalid_measurements`, `counts_measurements_mismatch`, `missing_measured_qubits`,
`invalid_measured_qubits`, `no_observed_shots`, `excess_observed_shots`, and
`invalid_evidence`. A malformed optional evidence field disables eligibility
without discarding the completed result. Consumers must reject unknown protocol
versions or malformed records rather than treating the flag alone as sufficient.

This record does not qualify physical addressing, verbatim compilation, hardware
availability, pulse timing or overlap. Wire/export and option parity remain
separate work in marqov-sdk#154, marqov-sdk#66 and marqov-sdk#77. Hosted transport
must preserve and verify this evidence separately; this local executor addition
is not hosted or hardware qualification.
