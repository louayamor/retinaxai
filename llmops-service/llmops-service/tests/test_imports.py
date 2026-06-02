from __future__ import annotations


def test_app_imports() -> None:
    from app.main import app

    assert app.title == "RetinaXAI LLMOps Service"
