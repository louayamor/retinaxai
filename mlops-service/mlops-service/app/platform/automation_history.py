import json
from datetime import datetime
from pathlib import Path


class AutomationHistory:
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.history_file.exists():
            with open(self.history_file) as f:
                self._history = json.load(f)

    def _save(self) -> None:
        with open(self.history_file, "w") as f:
            json.dump(self._history, f, indent=2)

    def record(self, event_type: str, data: dict) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        self._history.append(entry)
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self._save()

    def latest(self) -> dict | None:
        if not self._history:
            return None
        return self._history[-1]
