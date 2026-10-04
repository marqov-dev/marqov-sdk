# IBM executor qualification

Marqov reuses its IBMExecutor and the vendor's SamplerV2 execution path. A job ID
is captured for each submission before waiting for results. Result retrieval
errors and timeouts raise IBMExecutionError with that ID; they never trigger an
automatic resubmission. A timeout does not cancel the provider's running job.

Results use the shared canonical count decoder and identify IBM, Qiskit Runtime,
SamplerV2, local versus direct access, submitted job ID, exact package versions,
shots and transpiled-circuit/count hashes. The token is excluded from configuration
repr. No provider credential is added to a hosted runner or result record.

The exact executor was exercised offline with a deliberately supplied AerSimulator
backend, through the real vendor SamplerV2 and transpiler. Both asymmetric basis
states return the expected canonical `10` and `01` counts. This local qualification
is not an IBM cloud/QPU execution claim. The normal public executor configuration
still selects the requested IBM service backend; it has no local fallback.

Live direct qualification requires an existing IBM account, token, instance and
available backend, with an agreed budget. Hosted execution additionally requires
the credential-isolated gateway integration and normal admission/accounting
checks. Neither is claimed complete by the offline result.
