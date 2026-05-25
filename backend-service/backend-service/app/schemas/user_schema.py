from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator

from app.auth.roles import Role
from app.schemas.common import BaseResponse


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: Role = Role.DOCTOR

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer")
        return v


class UserUpdate(BaseModel):
    username: str | None = None
    is_active: bool | None = None
    role: Role | None = None


class UserRead(BaseResponse):
    id: uuid.UUID
    email: str
    username: str
    is_active: bool
    role: Role
    created_at: datetime
