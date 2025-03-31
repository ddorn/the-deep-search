from enum import StrEnum
from openai import BaseModel


class Status(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class Task(BaseModel):
    strategy: str
    document_id: str
    chunk_id: str | None
    child: str | None
    status: Status
