from __future__ import annotations

import pytest

from app.services.biomarker_client.resilience import CircuitBreaker, retry_with_backoff


def test_circuit_breaker_opens_after_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

    assert breaker.current_state() == "closed"
    breaker.on_failure()
    assert breaker.current_state() == "closed"
    breaker.on_failure()
    assert breaker.current_state() == "open"


@pytest.mark.asyncio
async def test_retry_with_backoff_retries_until_success():
    calls = {"count": 0}

    async def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("fail")
        return "ok"

    result = await retry_with_backoff(
        flaky,
        attempts=3,
        base_delay=0.0,
        multiplier=2.0,
        max_delay=1.0,
        jitter=0.0,
    )

    assert result == "ok"
    assert calls["count"] == 2
