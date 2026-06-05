from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.auth.dependencies import CurrentUser
from app.auth.roles import Role
from app.core.exceptions import ForbiddenException
from app.models.user import User


def require_role(allowed: Role | list[Role]):
    allowed_list = [allowed] if isinstance(allowed, Role) else allowed

    async def _check(user: CurrentUser) -> User:
        if user.role not in allowed_list:
            raise ForbiddenException(
                f"Role '{user.role}' is not authorized for this endpoint."
            )
        return user

    return _check


EngineerUser = Annotated[User, Depends(require_role(Role.ENGINEER))]
AdminUser = Annotated[User, Depends(require_role(Role.ADMIN))]
