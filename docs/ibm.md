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

## Isolated Linux qualification

`qualification/ibm-python312.json` records the current installed SDK on Linux
amd64/Python3.12, Qiskit2.4.1, IBM Runtime0.50.0 and Aer0.17.2. The existing
asymmetric basis workloads returned `10:32` and `01:32` through the real
SamplerV2. The container had no network or credentials, a read-only root,
non-root user and dropped capabilities. These are offline results.

The qualification starts from the retained compiler base and preserves its
70 non-SDK package versions; only published Marqov0.7.0 is replaced with the
current source SDK0.8.1. Requirements carry distribution hashes. This image
is an isolated qualification candidate, not a minimal hosted runtime.
Alice & Bob1.2 requires Qiskit1.x, while this IBM Runtime requires Qiskit2.x;
this image must not replace the existing shared partner environment. Hosted
adoption needs a separately reviewed immutable image/lock profile and the
credential-isolated gateway, not a switch on the legacy Python runner.

To repeat using the exact base image identity recorded in the receipt:

```bash
uv pip compile docs/qualification/ibm-python312.in -c docs/qualification/ibm-python312-constraints.txt --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes -o docs/qualification/ibm-python312-requirements.txt
docker buildx build --platform linux/amd64 --load --build-context compiler-runtime=docker-image://marqov-lightning-cpu-trixie:full -f docs/qualification/ibm-python312.Dockerfile -t marqov-ibm-current-sdk:qualification .
docker run --rm --platform linux/amd64 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --memory 2g --cpus 2 --tmpfs /work:rw,nosuid,nodev,size=128m,uid=10001,gid=10001 -v "$PWD/docs/qualification/ibm-python312.py:/opt/compiler/ibm-qualification.py:ro" --entrypoint /opt/venv/bin/python marqov-ibm-current-sdk:qualification /opt/compiler/ibm-qualification.py
```

The mutable local tag must first match the receipt's base image ID. Rebuilding
may produce a different candidate digest; the receipt identifies the actual
qualified image. No service deployment or cloud provider run is implied.
