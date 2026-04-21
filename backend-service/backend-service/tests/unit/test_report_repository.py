from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.reports.repository import ReportRepository


class DummyScalarResult:
    def __init__(self, items=None, count: int = 0):
        self._items = items or []
        self._count = count

    def scalars(self):
        return self

    def all(self):
        return list(self._items)

    def scalar_one(self):
        return self._count


class DummyDB:
    def __init__(self):
        self.calls = []

    async def execute(self, stmt):
        self.calls.append(stmt)
        return DummyScalarResult(items=[SimpleNamespace(id="r1")], count=7)


@pytest.mark.asyncio
async def test_report_repository_get_all_and_count_all():
    repo = ReportRepository(DummyDB())

    reports = await repo.get_all(skip=0, limit=10)
    total = await repo.count_all()

    assert len(reports) == 1
    assert total == 7
