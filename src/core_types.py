import datetime
from enum import StrEnum
from typing import Annotated
from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class Task(BaseModel):
    id: int
    strategy: str
    document_id: int
    status: TaskStatus = TaskStatus.PENDING
    args: str
    parent_id: int | None = None
    created_at: Annotated[
        datetime.datetime, Field(default_factory=datetime.datetime.now)
    ]


class PartialTask(BaseModel):
    strategy: str
    document_id: int
    args: str = ""
    status: TaskStatus = TaskStatus.PENDING
    parent_id: int | None = None


class Chunk(BaseModel):
    id: int
    document_id: int
    document_order: int
    content: str
    created_at: Annotated[
        datetime.datetime, Field(default_factory=datetime.datetime.now)
    ]


class PartialChunk(BaseModel):
    document_id: int
    document_order: int
    content: str


class Document(BaseModel):
    id: int
    urn: str
    source_id: str
    created_at: Annotated[
        datetime.datetime, Field(default_factory=datetime.datetime.now)
    ]


class PartialDocument(BaseModel):
    urn: str
    source_id: str
