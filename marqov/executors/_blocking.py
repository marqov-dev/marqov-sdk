"""Per-execution blocking calls with bounded, best-effort interruption cleanup."""

import asyncio
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Self

logger = logging.getLogger(__name__)
_CANCEL_WAIT_SECONDS = 1.0


class BlockingCalls:
    """Own threads separately from the event loop's default executor.

    Shutdown stops queued work without joining active calls. Python interpreter
    shutdown may still join those threads; vendor I/O limits remain necessary.
    Two workers allow cleanup to start while a result call is still blocked.
    """

    def __enter__(self) -> Self:
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="marqov-remote")
        return self

    def __exit__(self, *args: object) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    async def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            self._pool, partial(function, *args, **kwargs)
        )

    async def submit_owned(
        self,
        submit: Callable[..., Any],
        *args: Any,
        label: str,
        job_id_of: Callable[[Any], object],
    ) -> Any:
        """Run a one-send submission whose reply survives caller interruption.

        ``submit(*args, claim)`` must call ``claim()`` immediately before its
        irreversible send. Cancellation before the claim sends nothing. After it,
        wait a bounded interval for the reply, then re-raise the interruption with
        a note naming the accepted job, or with ``acceptance_unknown = True``. A
        reply arriving after that is logged with its job ID. Nothing is replayed.
        """
        lock = threading.Lock()
        state: dict[str, Any] = {"interrupted": False, "sending": False, "abandoned": False}
        replies: list[Any] = []

        def claim() -> None:
            with lock:
                if state["interrupted"]:
                    raise RuntimeError(f"{label} submission interrupted before send")
                state["sending"] = True

        def owned() -> Any:
            reply = submit(*args, claim)
            with lock:
                replies.append(reply)
                late = state["abandoned"]
            if late:
                logger.warning(
                    "%s %s was accepted after the interrupted caller stopped waiting; "
                    "reconcile it and do not resubmit",
                    label,
                    job_id_of(reply),
                )
            return reply

        post = asyncio.get_running_loop().run_in_executor(self._pool, owned)
        post.add_done_callback(lambda done: done.cancelled() or done.exception())
        try:
            return await asyncio.shield(post)
        except asyncio.CancelledError as interruption:
            with lock:
                state["interrupted"] = True
                sending = state["sending"]
            if not sending:
                post.cancel()
                raise
            try:
                await asyncio.wait_for(asyncio.shield(post), _CANCEL_WAIT_SECONDS)
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 - reply still unknown
                pass
            with lock:
                state["abandoned"] = True
                reply = replies[0] if replies else None
            post.cancel()
            if reply is None:
                interruption.acceptance_unknown = True  # type: ignore[attr-defined]
                interruption.add_note(f"{label} submission acceptance unknown; do not resubmit.")
            else:
                interruption.add_note(
                    f"{label} {job_id_of(reply)} was accepted before the interruption; "
                    "it is retained in last_job for readback and must not be resubmitted."
                )
            raise

    async def wait(
        self,
        result: Callable[[], Any],
        cancel: Callable[[], Any],
        *,
        job_id: str,
        timeout: float | None,
    ) -> Any:
        try:
            if timeout is None:
                return await self.call(result)
            return await asyncio.wait_for(self.call(result), timeout)
        except (TimeoutError, asyncio.CancelledError) as interruption:
            # asyncio.run() turns a Ctrl+C CancelledError into a bare
            # KeyboardInterrupt, dropping the note below; log the id first.
            logger.warning(
                "Remote task %r interrupted; cancellation requested, provider state unconfirmed",
                job_id,
            )
            try:
                await asyncio.wait_for(self.call(cancel), _CANCEL_WAIT_SECONDS)
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 - preserve original interruption
                logger.warning(
                    "Cancellation request for interrupted task %r was not confirmed; "
                    "the task may still be running",
                    job_id,
                )
            interruption.add_note(
                f"Remote task {job_id!r} was interrupted; cancellation was best effort "
                "and the provider task may still be running."
            )
            raise
