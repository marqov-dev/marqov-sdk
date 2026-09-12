# Optional Braket installation for managed compiler compatibility

The hosted compiler/task contract pins cloudpickle 3.1.2 and uses released SDK
capture across separate processes. Braket's patched SDK versions currently pin
cloudpickle 2.2.1. Keeping Braket in every SDK installation forces the hosted lock
resolver back to an older Braket release. Both the first patched release 1.117.0
and the available patched range through 1.127.0 were tested and conflict with the
hosted serialization pin. The [vendor advisory](https://github.com/amazon-braket/amazon-braket-sdk-python/security/advisories/GHSA-g697-2xrc-gc46)
identifies the affected job-result/checkpoint deserialization path and the fix.

Braket now belongs to the `braket` and `all` extras, with minimum version 1.117.0.
Core workflow serialization declares cloudpickle directly. Platform runtime
profiles retain their exact serialization pin; the SDK does not override
Braket's dependency metadata or downgrade the platform contract. Existing locked
SDK development/provider dependency versions remain unchanged.

This is an installation compatibility change. Existing Braket users select
`marqov[braket]` or `marqov[all]`. `MarqovDevice("local", ...)` and `marqov-sim`
use Braket's local simulator and also need the extra. `LocalExecutor`,
`Circuit.simulate()` and core workflow capture remain independent. Missing
provider/conversion dependencies give an actionable installation error; unrelated
broken transitive imports still propagate. Simulator semantics are unchanged.

## Local qualification

- Full SDK suite: 780 passed, 20 skipped, 13 existing expected-failure tests passed;
  zero failures. No new skip/xfail marks were introduced. The initial sandbox run
  could not download/start the official Temporal test server; the permitted run
  passed the real workflow isolation test.
- `tests/test_optional_braket.py` covers core capture without Braket, explicit
  provider/local-device/conversion refusal and an unrelated broken dependency.
- A built candidate wheel was installed in a fresh environment with
  cloudpickle 3.1.2 and no amazon-braket-sdk distribution. Metadata exposes Braket
  only under its two extras. Actual LocalExecutor Bell execution produced 100
  shots; a captured function executed in a separate child process and reconstructed
  its structured answer 42.
- Changed Python files pass Ruff; the candidate diff passes the secret scan.

The local wheel retains the checkout's 0.6.1 version only as an unpublished test
artifact (SHA256 321360d291c311ec7dd1bfd12bccd42e10d1aef650d099145883e768df3c1d2c).
It must not replace the released 0.6.1 artifact. A reviewed new release/version,
published wheel hash, refreshed frozen platform profiles and exact container
qualification are required before runtime adoption. No package was published and
no hosted runtime was changed by this qualification.

## Release candidate

The candidate is version0.7.0 because default installation no longer supplies
Braket execution, circuit conversion or the MarqovDevice local simulator.
Existing users of those paths must install `marqov[braket]` (or `[all]`).
Independent review found no blocking code issue. The earlier0.6.1 local wheel
was qualification evidence only; published0.6.1 remains immutable. The0.7.0
artifact must be reviewed, published, pinned and qualified by the platform before
adoption. No publication is implied by this version change.

The built0.7.0 local candidate wheel passed the same installed-core proof with
cloudpickle3.1.2, Braket absent,100-shot local Bell simulation and a captured
function executed in a fresh process returning the structured answer42.
Candidate SHA256: `8843a6d43201ef219aba06ca35d2ee02326a83ff216cec618aedd50b1c8e24c7`.
This local build hash is not a claim about a future published artifact.

Final0.7.0 candidate full suite:780 passed,20 skipped and13 pre-existing XPASS,
with no new skips or expected failures. Log: `/private/tmp/marqov-sdk-070-full-tests.log`.
