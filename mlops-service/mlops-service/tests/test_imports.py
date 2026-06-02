from __future__ import annotations


def test_app_imports() -> None:
    from app.api.app import app

    assert app.title == "RetinaXAI MLOps Service"
