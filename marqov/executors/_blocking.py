"""Per-execution blocking calls with bounded, best-effort interruption cleanup."""

import asyncio
import logging
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
