from __future__ import annotations
from fastapi import HTTPException, status


class AppException(HTTPException):
    def __init__(self, status_code: int, detail: str, error_code: str | None = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class NotFoundException(AppException):
    def __init__(self, resource: str, identifier: str | int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with identifier '{identifier}' not found.",
            error_code="NOT_FOUND",
        )


class ConflictException(AppException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="CONFLICT",
        )


class ForbiddenException(AppException):
    def __init__(self, detail: str = "Insufficient permissions."):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="FORBIDDEN",
        )


class UnauthorizedException(AppException):
    def __init__(self, detail: str = "Could not validate credentials."):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="UNAUTHORIZED",
        )


class ServiceUnavailableException(AppException):
    def __init__(self, service: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Upstream service '{service}' is unavailable.",
            error_code="SERVICE_UNAVAILABLE",
        )


class UnprocessableEntityException(AppException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="UNPROCESSABLE_ENTITY",
        )
