from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.biomarkers.repository import BiomarkerRepository


class DummyScalarResult:
    def __init__(self, item=None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


class DummyDB:
    def __init__(self, item=None):
        self.item = item
        self.calls = []

    async def execute(self, stmt):
        self.calls.append(stmt)
        return DummyScalarResult(self.item)

    def add(self, obj):
        self.calls.append(("add", obj))

    async def flush(self):
        self.calls.append("flush")

    async def refresh(self, obj):
        self.calls.append(("refresh", obj))


@pytest.mark.asyncio
async def test_biomarker_repository_get_by_prediction_id_returns_item():
    repo = BiomarkerRepository(DummyDB(item=SimpleNamespace(id="b1")))

    biomarker = await repo.get_by_prediction_id(uuid.uuid4())

    assert biomarker is not None
