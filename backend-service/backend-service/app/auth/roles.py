from __future__ import annotations

import enum


class Role(str, enum.Enum):
    DOCTOR = "doctor"
    ENGINEER = "engineer"
    ADMIN = "admin"
