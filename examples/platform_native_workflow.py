"""Run a saved @workflow on Marqov's managed native runtime.

Discover the team's runtime -> submit -> wait -> read the result.

Needs a Marqov API key with ``simulate`` permission or higher, the UUID of the
team that pays for the run, and a script saved in that team whose source
defines the workflow named below. Set:

    MARQOV_PLATFORM_KEY   API key (marqey_live_... / marqey_test_...)
    MARQOV_TEAM_ID        team UUID
    MARQOV_SCRIPT_ID      saved script UUID
"""

import os
import uuid

from marqov.platform import MarqovClient

client = MarqovClient()  # reads MARQOV_PLATFORM_KEY
team_id = os.environ["MARQOV_TEAM_ID"]
script_id = os.environ["MARQOV_SCRIPT_ID"]

# 1. Discover. Only runtimes enabled for this team are listed; an empty list
#    means managed native execution is not enabled for it.
runtime = next(
    (r for r in client.managed_runtimes(team_id) if r["programming_model"] == "native_workflow"),
    None,
)
if runtime is None:
    raise SystemExit("Managed native workflows are not enabled for this team.")
print(f"{runtime['backend']}: cap between {runtime['min_cap_cents']} and "
      f"{runtime['max_cap_cents']} cents")

# 2. Submit. The cap is your decision: the most this run may spend, in cents,
#    covering compilation and tasks. Keep the idempotency key: resubmitting with
#    the same key returns the original admission instead of running twice.
cap_cents = 100
idempotency_key = str(uuid.uuid4())
job = client.submit_native(
    team_id=team_id,
    script_id=script_id,
    entrypoint="native_canary",   # the @workflow function in the saved script
    kwargs={"seed": 7},
    cap_cents=cap_cents,
    idempotency_key=idempotency_key,
)
print("Admitted job", job.id)

# 3. Wait. Raises JobFailed if the job fails and TimeoutError if it is still
#    running when the timeout expires (it keeps running server-side).
result = job.result(timeout=600.0)

# 4. Read. A workflow result is the platform's managed-result projection: one
#    entry per declared output, with its JSON value when it can be displayed.
for output in result.raw["outputs"]:
    display = output["display"]
    if display["state"] == "available":
        print(output["task_key"], display["value"])
    else:
        print(output["task_key"], "value not displayable:", display["reason"])
