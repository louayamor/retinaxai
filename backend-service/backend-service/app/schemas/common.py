from __future__ import annotations
from pydantic import BaseModel


class BaseResponse(BaseModel):
    model_config = {"from_attributes": True}


class PaginatedResponse(BaseModel):
    total: int
    page: int
    size: int
    pages: int


class MessageResponse(BaseModel):
    message: str
