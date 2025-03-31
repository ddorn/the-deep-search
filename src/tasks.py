import datetime
from enum import StrEnum
from pydantic import BaseModel


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class Task(BaseModel):
    id: int
    strategy: str
    document_id: str
    status: TaskStatus = TaskStatus.PENDING
    args: str
    child_id: str | None
    created_at: datetime.datetime


class PartialTask(BaseModel):
    strategy: str
    document_id: str
    args: str = ""
    status: TaskStatus = TaskStatus.PENDING
    child: str | None = None

