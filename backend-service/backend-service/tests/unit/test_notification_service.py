from __future__ import annotations

import uuid

import pytest

from app.notifications.service import NotificationService


class DummyDBNotification:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True


@pytest.mark.asyncio
async def test_notification_service_creates_direct_notification() -> None:
    db = DummyDBNotification()
    service = NotificationService(db)
    await service.process_event_notification(
        "notification",
        {"type": "alert", "title": "Test", "message": "Hello"},
    )
    assert db.committed
    assert len(db.added) == 1
    n = db.added[0]
    assert n.type == "alert"
    assert n.title == "Test"
    assert n.message == "Hello"


@pytest.mark.asyncio
async def test_notification_service_training_completed() -> None:
    db = DummyDBNotification()
    service = NotificationService(db)
    await service.process_event_notification(
        "training_stage",
        {
            "stage": "pipeline",
            "status": "completed",
            "message": "Training finished",
            "pipeline": "imaging",
        },
    )
    assert db.committed
    assert any(n.title == "Training Completed" for n in db.added)


@pytest.mark.asyncio
async def test_notification_service_skips_training_progress() -> None:
    db = DummyDBNotification()
    service = NotificationService(db)
    await service.process_event_notification(
        "training_stage",
        {"stage": "epoch_3", "status": "running", "message": "Epoch 3/10"},
    )
    assert not db.added


@pytest.mark.asyncio
async def test_notification_service_xai_completed() -> None:
    db = DummyDBNotification()
    service = NotificationService(db)
    await service.process_event_notification(
        "xai.explanation_ready",
        {"stage": "shap", "status": "completed", "message": "XAI done"},
    )
    assert db.committed
    assert any(n.title == "XAI Completed" for n in db.added)


@pytest.mark.asyncio
async def test_notification_service_llmops_completed() -> None:
    db = DummyDBNotification()
    service = NotificationService(db)
    await service.process_event_notification(
        "report_generation",
        {"status": "completed", "message": "Report generated"},
    )
    assert db.committed
    assert any(n.title == "LLM Operation Complete" for n in db.added)
