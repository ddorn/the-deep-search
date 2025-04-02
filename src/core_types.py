import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated
from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class DBModel(BaseModel):
    id: int
    created_at: Annotated[
        datetime.datetime, Field(default_factory=datetime.datetime.now)
    ]


class PartialTask(BaseModel):
    strategy: str
    document_id: int
    args: str = ""
    status: TaskStatus = TaskStatus.PENDING
    parent_id: int | None = None


class Task(PartialTask, DBModel):
    pass

class PartialChunk(BaseModel):
    document_id: int
    document_order: int
    content: str

class Chunk(PartialChunk, DBModel):
    pass



class PartialDocument(BaseModel):
    urn: str
    source_id: str

class Document(PartialDocument, DBModel):
    pass


class PartialByproduct(BaseModel):
    document_id: int
    path: Path

class Byproduct(PartialByproduct, DBModel):
    pass