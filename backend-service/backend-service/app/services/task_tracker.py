from __future__ import annotations

import asyncio
from asyncio import Task
from typing import Any

from loguru import logger


class BackgroundTaskRegistry:
    def __init__(self) -> None:
        self._tasks: set[Task[Any]] = set()

    def create_task(self, coro: Any, *, name: str = "") -> Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self, timeout: float = 5.0) -> None:
        if not self._tasks:
            return
        logger.info("draining_background_tasks", count=len(self._tasks), timeout=timeout)
        done, pending = await asyncio.wait(
            self._tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
        )
        for task in pending:
            task.cancel()
        self._tasks.clear()
        logger.info("background_tasks_drained", done=len(done), cancelled=len(pending))


bg_tasks = BackgroundTaskRegistry()
