from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.auth.roles import Role
from app.core.security import hash_password
from app.models.user import User

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://louay:louay@localhost:5432/retinaxai_db",
)


async def main() -> None:
    engine = create_async_engine(DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        result = await session.execute(select(User).where(User.email == "admin@retinaxai.com"))
        existing = result.scalar_one_or_none()

        if existing:
            print(f"Admin already exists: {existing.username} (role={existing.role})")
            return

        user = User(
            email="admin@retinaxai.com",
            username="admin",
            hashed_password=hash_password("admin123"),
            role=Role.ADMIN,
            is_active=True,
        )
        session.add(user)
        await session.commit()
        print(f"Created admin: {user.email} / admin123")


if __name__ == "__main__":
    asyncio.run(main())
