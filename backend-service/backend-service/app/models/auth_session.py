from __future__ import annotations
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class AuthSession(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "auth_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    access_token_jti: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    refresh_token: Mapped[str] = mapped_column(
        String(512), nullable=False, unique=True, index=True
    )
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    user = relationship("User", back_populates="auth_sessions")
