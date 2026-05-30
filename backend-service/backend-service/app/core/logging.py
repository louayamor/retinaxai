from __future__ import annotations

import contextvars
import logging
import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings

LOG_DIR = Path("logs")

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def _inject_request_id(record: dict) -> None:
    record["extra"]["request_id"] = request_id_var.get() or record["extra"].get("request_id", "")


def _json_sink(message) -> None:
    record = message.record
    _inject_request_id(record)
    import json
    sys.stdout.write(json.dumps({
        "time": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "request_id": record["extra"]["request_id"],
        "message": record["message"],
        "extra": {k: v for k, v in record["extra"].items() if k != "request_id"},
    }, default=str) + "\n")


_CONSOLE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function}:{line} - {message}\n"


def _console_sink(message) -> None:
    record = message.record
    _inject_request_id(record)
    sys.stdout.write(_CONSOLE_FORMAT.format_map(record))


def _file_sink(message) -> None:
    record = message.record
    _inject_request_id(record)
    msg = _CONSOLE_FORMAT.format_map(record)
    try:
        with open(str(LOG_DIR / "system" / "app.log"), "a") as f:
            f.write(msg)
    except OSError:
        pass


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "system").mkdir(exist_ok=True)
    (LOG_DIR / "auth").mkdir(exist_ok=True)
    (LOG_DIR / "requests").mkdir(exist_ok=True)

    log_level = settings.LOG_LEVEL.upper()

    logger.remove()
    if settings.LOG_FORMAT == "json":
        logger.add(_json_sink, level=log_level)
    else:
        logger.add(_console_sink, level=log_level)
    logger.add(_file_sink, level=log_level)

    logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
