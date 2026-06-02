from __future__ import annotations

from pathlib import Path


def test_app_imports() -> None:
    # StaticFiles mount requires uploads dir to exist
    from app.core.config import settings

    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    from app.main import app

    assert app.title == "retinaxai-backend"
