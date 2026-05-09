from __future__ import annotations


from app.main import app, create_app


def test_create_app_returns_fastapi_instance():
    assert app is not None
    assert app.title == "RetinaXAI LLMOps Service"


def test_create_app_has_no_cli_functions():
    import app.main as main_module

    assert not hasattr(main_module, "run_serve")
    assert not hasattr(main_module, "run_reindex")
    assert not hasattr(main_module, "main")
