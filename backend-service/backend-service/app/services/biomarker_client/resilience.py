from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass


class CircuitBreakerOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    success_count: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_successes: int = 2,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_successes = half_open_successes
        self.state = CircuitBreakerState()

    def _is_open(self) -> bool:
        if self.state.opened_at is None:
            return False
        return (time.monotonic() - self.state.opened_at) < self.recovery_timeout

    def can_attempt(self) -> bool:
        if self.state.opened_at is None:
            return True
        if self._is_open():
            return False
        return True

    def on_success(self) -> None:
        if self.state.opened_at is not None:
            self.state.success_count += 1
            if self.state.success_count >= self.half_open_successes:
                self.state.opened_at = None
                self.state.failure_count = 0
                self.state.success_count = 0
        else:
            self.state.failure_count = 0

    def on_failure(self) -> None:
        if self.state.opened_at is not None:
            self.state.opened_at = time.monotonic()
            self.state.success_count = 0
            return

        self.state.failure_count += 1
        if self.state.failure_count >= self.failure_threshold:
            self.state.opened_at = time.monotonic()
            self.state.success_count = 0

    def current_state(self) -> str:
        if self.state.opened_at is None:
            return "closed"
        if self._is_open():
            return "open"
        return "half_open"


async def retry_with_backoff(
    func,
    *,
    attempts: int,
    base_delay: float,
    multiplier: float,
    max_delay: float,
    jitter: float,
    on_retry=None,
):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break

            delay = min(base_delay * (multiplier ** (attempt - 1)), max_delay)
            if jitter:
                delay *= 1 + random.uniform(-jitter, jitter)
            if on_retry:
                on_retry(exc, attempt, delay)
            await asyncio.sleep(max(0.0, delay))

    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_with_backoff failed without exception")
