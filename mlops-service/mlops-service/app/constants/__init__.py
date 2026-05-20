from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_FILE_PATH = _SERVICE_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = _SERVICE_ROOT / "config" / "params.yaml"
SCHEMA_FILE_PATH = _SERVICE_ROOT / "config" / "schema.yaml"