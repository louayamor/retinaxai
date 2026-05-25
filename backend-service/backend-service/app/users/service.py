from __future__ import annotations
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ConflictException,
    ForbiddenException,
    NotFoundException,
    UnauthorizedException,
)
from app.core.security import hash_password, verify_password
from app.models.user import User
from app.schemas.user_schema import UserCreate, UserUpdate
from app.users.repository import UserRepository


class UserService:
    def __init__(self, db: AsyncSession):
        self.repo = UserRepository(db)

    async def create(self, data: UserCreate) -> User:
        existing = await self.repo.get_by_email(data.email)
        if existing:
            raise ConflictException(f"User with email '{data.email}' already exists.")
        user = User(
            email=data.email,
            username=data.username,
            hashed_password=hash_password(data.password),
        )
        return await self.repo.create(user)

    async def get_by_id(self, user_id: uuid.UUID) -> User:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise NotFoundException("User", user_id)  # type: ignore[reportArgumentType]
        return user

    async def get_by_email(self, email: str) -> User:
        user = await self.repo.get_by_email(email)
        if not user:
            raise NotFoundException("User", email)
        return user

    async def authenticate(self, email: str, password: str) -> User:
        user = await self.repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise UnauthorizedException("Invalid email or password.")
        if not user.is_active:
            raise ForbiddenException("Account is inactive.")
        return user

    async def update(self, user_id: uuid.UUID, data: UserUpdate) -> User:
        user = await self.get_by_id(user_id)
        for field, value in data.model_dump(exclude_none=True).items():
            setattr(user, field, value)
        return await self.repo.update(user)
