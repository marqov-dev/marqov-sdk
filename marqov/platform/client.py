"""Public client for the Marqov managed-quantum platform.

Usage::

    from marqov.platform import MarqovClient

    client = MarqovClient(api_key="marqey_live_...")
    job    = client.submit("from marqov import task; ...", backend="sv1", framework="marqov")
    result = job.result(timeout=120.0)
    print(result.counts)

Observable API contract notes:

- Submit request body fields: ``inline_code``, ``framework``, ``backend``,
  and ``params`` (with ``shots``).  Note: the server-side field is ``params``,
  **not** ``parameters``.

- Submit response shape: ``{ "job_id": "<uuid>" }``.

- Backends endpoint response shape: ``{ "backends": [...], "updatedAt": "..." }``.

- Backends item fields (camelCase from server): ``slug``, ``name``,
  ``provider``, ``deviceType``, ``status``, ``isAvailable``, ``pricing``,
  ``supportedProgramTypes``.

- ``platform_info()`` — §11 TBC assumption:
      No ``/api/meta``, ``/api/version``, or ``/api/health`` route was found.
      ``platform_info()`` is implemented against the **mocked path** ``/api/meta``
      and marked as a §11 TBC assumption.  When the platform ships a real endpoint,
      update the path and the response mapping here (and remove the §11 TBC warning
      in the docstring).

- ``sdk_version``: sourced from ``marqov.__version__`` and sent for
  forward-compat; not yet consumed server-side (the server strips unknown
  keys from the submit body — reconciliation item).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import marqov
from marqov.circuits import Circuit

from ._models import Backend, PlatformInfo
from ._native import build_public_submission, check_receipt, parse_runtimes, require_uuid
from ._transport import Transport
from .errors import MarqovPlatformError
from .job import Job


class MarqovClient:
    """High-level client for the Marqov Platform API.

    Manages a single :class:`~marqov.platform._transport.Transport` instance
    shared across all method calls.  Key resolution, Bearer-token injection,
    and all retry / idempotency logic are delegated to the transport.

    Args:
        api_key:  Marqov Platform API key (``marqey_live_…`` or
                  ``marqey_test_…``).  Falls back to the
                  ``MARQOV_PLATFORM_KEY`` environment variable when ``None``.
        base_url: Override the default production endpoint.  Falls back to the
                  ``MARQOV_PLATFORM_URL`` env var, then the built-in default
                  (``https://app.marqov.ai``).
        timeout:  Per-request HTTP timeout in seconds.  Default is 30 seconds.

    Raises:
        :class:`~marqov.platform.errors.AuthenticationError`: If no API key is
            resolved from the argument or environment.

    Example::

        client = MarqovClient(api_key="marqey_live_abc123")
        job    = client.submit(circuit, backend="sv1")
        result = job.result()
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._transport = Transport(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        program: str | Circuit,
        *,
        backend: str,
        shots: int = 1000,
        framework: str | None = None,
    ) -> Job:
        """Submit a quantum program to the platform.

        Args:
            program:   The quantum program to run.  Accepts either:

                       * A ``str`` — treated as ``inline_code``.  The
                         ``framework`` argument is **required** when passing a
                         string program; omitting it raises :class:`ValueError`.
                       * A :class:`marqov.Circuit` — serialised as a separate
                         ``circuit`` body field: ``{"format": "qasm3",
                         "payload": <OpenQASM 3 string>}``.  ``inline_code``
                         is **not** set for a ``Circuit`` submission.  Passing
                         ``framework`` with a ``Circuit`` raises
                         :class:`ValueError` (the circuit self-describes its
                         format; a framework override would be incorrect).

                         .. warning::
                             **PROVISIONAL — §11 reconciliation item.**
                             The exact circuit-submission wire contract (the
                             ``circuit`` body field) is pending the platform's
                             circuit-submission variant (spec §8.6 #1), which is
                             unbuilt at this revision.  This path is verified —
                             or marked **BLOCKED** — by the Task 5b staging
                             smoke test.  Do not assume it works against the
                             current server.

            backend:   Backend slug to run on (e.g. ``"sv1"``,
                       ``"dwave-sim"``).
            shots:     Number of measurement shots.  Default is 1 000.
            framework: Framework identifier for string programs (e.g.
                       ``"marqov"``).  **Required** when *program* is a
                       ``str``; ignored — and an error — when *program* is a
                       :class:`~marqov.Circuit`.

        Returns:
            :class:`~marqov.platform.job.Job` handle for the submitted job.

        Raises:
            :class:`ValueError`:
                - ``program`` is a ``str`` but ``framework`` was not supplied.
                - ``program`` is a :class:`~marqov.Circuit` and ``framework``
                  was supplied (the circuit self-describes its format).
            :class:`TypeError`:
                ``program`` is neither ``str`` nor :class:`~marqov.Circuit`.
            :class:`~marqov.platform.errors.PaidBackendNotSupportedYet`:
                The server returned ``analysis_required`` (422) because the
                requested backend requires a pre-run analysis (paid backends).
                In v1.0 this propagates from the transport without special-
                casing; the caller should choose a free backend.
            :class:`~marqov.platform.errors.AuthenticationError`:
                HTTP 401 from the server.
            :class:`~marqov.platform.errors.MarqovPlatformError`:
                Any other non-2xx platform error.

        Wire contract:
            - Submit body fields: ``inline_code``, ``framework``, ``backend``,
              ``params`` (with ``shots``).
            - Submit response: ``{ "job_id": "<uuid>" }``.
        """
        # --- Validate program type and build body --------------------------
        if isinstance(program, str):
            if framework is None:
                raise ValueError(
                    "framework is required when submitting a string program. "
                    "Pass framework='marqov' (or the appropriate framework for your code)."
                )
            inline_code: str | None = program
            req_framework: str | None = framework
            circuit_payload: dict | None = None

        elif isinstance(program, Circuit):
            if framework is not None:
                raise ValueError(
                    "Do not pass framework= when submitting a marqov.Circuit. "
                    "The circuit self-describes its format as QASM 3."
                )
            # PROVISIONAL — §11 reconciliation item pending the platform's
            # circuit-submission variant (spec §8.6 #1).
            #
            # The circuit rides as a SEPARATE body field ("circuit"), NOT inside
            # inline_code.  inline_code is executable server-side code and must
            # not carry a JSON envelope.  The exact wire contract for "circuit"
            # is unbuilt on the platform at this revision; this implementation
            # will be verified — or marked BLOCKED — by the Task 5b staging
            # smoke test.  Do NOT claim this path works against the current server.
            qasm3_str = program.to_openqasm(version=3)
            circuit_payload = {"format": "qasm3", "payload": qasm3_str}
            inline_code = None
            req_framework = None

        else:
            raise TypeError(
                f"program must be a str or marqov.Circuit, got {type(program).__name__!r}."
            )

        # --- Build request body --------------------------------------------
        # Body field names follow the server's submit schema.
        # Note: the server-side field is "params" (not "parameters").
        body: dict = {
            "backend": backend,
            "params": {"shots": shots},
            # sdk_version is sent for forward-compat; not yet consumed server-side
            # (server's Zod schema strips unknown keys — reconciliation item).
            "sdk_version": marqov.__version__,
        }
        if inline_code is not None:
            # str-program path: inline_code carries the executable source.
            body["inline_code"] = inline_code
        if req_framework is not None:
            body["framework"] = req_framework
        if circuit_payload is not None:
            # Circuit path: "circuit" is a separate field, NOT inside inline_code.
            # PROVISIONAL — see docstring §11 reconciliation note above.
            body["circuit"] = circuit_payload

        # --- POST to /api/jobs/submit (idempotent write — safe to retry) ---
        resp = self._transport.request(
            "POST",
            "/api/jobs/submit",
            json=body,
            idempotent_write=True,
        )

        # Response shape: { "job_id": "<uuid>" }
        job_id: str = resp["job_id"]
        return Job(self._transport, job_id)

    def managed_runtimes(self, team_id: str) -> list[dict[str, Any]]:
        """List the managed native runtimes enabled for a team.

        Calls ``GET /api/jobs/managed-runtimes?team_id=…``.  Each entry is
        ``{"backend", "programming_model", "min_cap_cents", "max_cap_cents"}``.
        An empty list means managed native execution is not enabled for the
        team.  Discovery is advisory: it reserves and authorises nothing, and
        the platform re-checks everything at submission.

        Args:
            team_id: The team's UUID.  An API key may only query its own team.

        Raises:
            ValueError: ``team_id`` is not a UUID (no request is sent).
            MarqovPlatformError: The response has an unexpected shape, or any
                platform error (e.g. 404 for a team the key cannot see).
        """
        require_uuid(team_id, "team_id")
        resp = self._transport.request(
            "GET", "/api/jobs/managed-runtimes", params={"team_id": team_id}
        )
        return parse_runtimes(resp)

    def submit_native(
        self,
        *,
        team_id: str,
        entrypoint: str,
        cap_cents: int,
        script_id: str | None = None,
        source: str | None = None,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        programming_model: str = "native_workflow",
        backend: str = "marqov-sim",
        source_sha256: str | None = None,
        idempotency_key: str | None = None,
    ) -> Job:
        """Submit a Python ``@task``/``@workflow`` program to a managed runtime.

        Uses the platform's versioned managed-native submission
        (``marqov.public-submission/v1``).  The program runs on the platform's
        own pinned SDK, not the version installed locally.  The returned
        :class:`~marqov.platform.job.Job` is polled and read exactly like any
        other job; a completed workflow's ``result().raw`` is the platform's
        ``marqov.managed-result/v1`` projection.

        Steps, in order (the first failure stops everything after it):

        1. Validate and serialise the request locally.  Nothing is sent if
           this fails.
        2. Discover the team's runtimes and require one matching ``backend``
           **and** ``programming_model`` exactly, with ``cap_cents`` inside
           its range.  No other backend or runtime is ever substituted.
        3. ``POST /api/jobs/submit`` with the ``Idempotency-Key``.
        4. Verify the admission receipt refers to this source and input.

        Args:
            team_id:           Team UUID that owns and pays for the job.  Required.
            entrypoint:        Name of the workflow (or task) function to call.
            cap_cents:         Spending cap in cents covering compilation and
                               tasks.  Required; there is no default.
            script_id:         UUID of a script saved on the platform, **or**
            source:            the Python source to submit inline.  Pass exactly one.
            args, kwargs:      Arguments for ``entrypoint``.  Must be JSON
                               values (dict keys must be ``str``; numbers finite).
                               Integers are sent exactly, however large.
            programming_model: ``"native_workflow"`` (default) or ``"single_task"``.
            backend:           Runtime backend; must match a discovered runtime.
            source_sha256:     Optional lowercase hex SHA-256 of the saved
                               script's content.  The platform refuses the
                               submission (409) if the saved script changed.
            idempotency_key:   Optional UUID.  Reuse the same key to retry the
                               same submission safely, including from another
                               process; the platform returns the original
                               admission instead of admitting twice.  A fresh
                               key is generated when omitted.

        Returns:
            :class:`~marqov.platform.job.Job` for the admitted job.

        Raises:
            ValueError / TypeError: The request is invalid (nothing is sent),
                or ``cap_cents`` is outside the discovered runtime's range.
            MarqovPlatformError: ``code="runtime_not_enabled"`` when no
                matching runtime is enabled for the team; the server's code for
                a refusal (e.g. ``spend_limit_exceeded`` 402,
                ``idempotency_conflict`` 409, ``managed_execution_unavailable``
                503 — retry with the same ``idempotency_key``);
                ``invalid_receipt`` / ``receipt_mismatch`` if the response
                cannot be confirmed as this submission's admission.
        """
        body, input_text = build_public_submission(
            team_id=team_id, entrypoint=entrypoint, cap_cents=cap_cents,
            script_id=script_id, source=source, args=args, kwargs=kwargs,
            programming_model=programming_model, backend=backend,
            source_sha256=source_sha256,
        )
        if idempotency_key is not None:
            require_uuid(idempotency_key, "idempotency_key")

        runtimes = self.managed_runtimes(team_id)
        runtime = next(
            (r for r in runtimes
             if r["backend"] == backend and r["programming_model"] == programming_model),
            None,
        )
        if runtime is None:
            enabled = [f"{r['backend']}/{r['programming_model']}" for r in runtimes] or ["none"]
            raise MarqovPlatformError(
                f"No managed runtime {backend}/{programming_model} is enabled for this team "
                f"(enabled: {', '.join(enabled)}).",
                code="runtime_not_enabled",
            )
        if not runtime["min_cap_cents"] <= cap_cents <= runtime["max_cap_cents"]:
            raise ValueError(
                f"cap_cents must be between {runtime['min_cap_cents']} and "
                f"{runtime['max_cap_cents']} for this runtime, got {cap_cents}"
            )

        resp = self._transport.request(
            "POST",
            "/api/jobs/submit",
            json=body,
            idempotent_write=True,
            idempotency_key=idempotency_key,
        )
        job_id = check_receipt(
            resp, input_text=input_text, source=source, source_sha256=source_sha256
        )
        return Job(self._transport, job_id)

    def job(self, job_id: str) -> Job:
        """Reconnect to an existing job by ID.

        Creates a :class:`~marqov.platform.job.Job` handle for a job that was
        previously submitted (e.g. from a different process or session).

        Args:
            job_id: The UUID string of the existing job.

        Returns:
            :class:`~marqov.platform.job.Job` handle that can be used to poll
            status, retrieve results, or request cancellation.

        Example::

            job = client.job("550e8400-e29b-41d4-a716-446655440000")
            result = job.result(timeout=60.0)
        """
        return Job(self._transport, job_id)

    def backends(self) -> list[Backend]:
        """Fetch the list of available quantum backends from the platform.

        Calls GET ``/api/backends`` and maps each item in the ``backends``
        array to a :class:`~marqov.platform._models.Backend` dataclass.

        The server returns camelCase field names; this method translates them to
        the snake_case names used by :class:`~marqov.platform._models.Backend`.

        Returns:
            List of :class:`~marqov.platform._models.Backend` instances, ordered
            by the server's ``displayOrder`` field.

        Raises:
            :class:`~marqov.platform.errors.MarqovPlatformError`: Any non-2xx
                platform error.

        Wire contract:
            Response: ``{ "backends": [...], "updatedAt": "..." }``
        """
        resp = self._transport.request("GET", "/api/backends")

        # Response shape: { "backends": [...], "updatedAt": "..." }
        raw_backends: list[dict] = resp.get("backends", [])

        result: list[Backend] = []
        for raw in raw_backends:
            # Map camelCase server fields to snake_case Backend dataclass.
            # Build extra dict from all remaining camelCase keys not mapped
            extra: dict = {}
            for k, v in raw.items():
                if k not in (
                    "slug", "name", "provider", "deviceType",
                    "status", "isAvailable", "pricing", "supportedProgramTypes",
                ):
                    extra[k] = v

            backend = Backend(
                slug=raw.get("slug", ""),
                name=raw.get("name", ""),
                provider=raw.get("provider", ""),
                device_type=raw.get("deviceType", ""),
                status=raw.get("status", ""),
                is_available=bool(raw.get("isAvailable", False)),
                pricing=raw.get("pricing") or {},
                supported_program_types=raw.get("supportedProgramTypes") or [],
                extra=extra,
            )
            result.append(backend)

        return result

    def platform_info(self) -> PlatformInfo:
        """Return version metadata about the SDK and the platform API.

        .. warning::
            **§11 TBC assumption**: No ``/api/meta``, ``/api/version``,
            or ``/api/health`` endpoint has been confirmed in the platform API.
            This method is implemented against the **mocked path** ``/api/meta``.
            When the platform ships a real endpoint, update the path and the
            response-field mapping here (and remove this warning).

        Returns:
            :class:`~marqov.platform._models.PlatformInfo` with:

            * ``sdk_version``: the installed ``marqov`` SDK version
              (from ``marqov.__version__``).
            * ``api_version``: the platform API version string, read from the
              ``api_version`` field of the ``/api/meta`` response (TBC).

        Raises:
            :class:`~marqov.platform.errors.MarqovPlatformError`: Any non-2xx
                platform error.
        """
        # §11 TBC assumption: path "/api/meta" is mocked — no real endpoint exists.
        resp = self._transport.request("GET", "/api/meta")
        api_version: str = resp.get("api_version", "unknown")
        return PlatformInfo(
            sdk_version=marqov.__version__,
            api_version=api_version,
        )

