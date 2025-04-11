import datetime
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class DBModel(BaseModel):
    id: int
    created_at: Annotated[datetime.datetime, Field(default_factory=datetime.datetime.now)]


class PartialTask(BaseModel):
    strategy: str
    document_id: int
    input_asset_id: int
    status: TaskStatus = TaskStatus.PENDING
    one_shot: bool = False


class Task(PartialTask, DBModel):
    pass


class PartialChunk(BaseModel):
    document_id: int
    document_order: int
    content: str


class Chunk(PartialChunk, DBModel):
    pass


class PartialDocument(BaseModel):
    source_urn: str
    source_id: str
    title: str
    url: str | None = None


class Document(PartialDocument, DBModel):
    pass


class AssetType(StrEnum):
    GOOGLE_DOC = "google_doc"
    AUDIO_TO_DL = "audio_to_dl"
    AUDIO_FILE = "audio_file"
    TRANSCRIPT = "transcript"
    GENERIC_FILE = "generic_file"
    TEXT_FILE = "text_file"
    SYNCED_TEXT_FILE = "synced_text_file"
    CHUNK_ID = "chunk_id"
    EMBEDDING_ID = "embedding_id"


class PartialAsset(BaseModel):
    document_id: int
    created_by_task_id: int | None
    next_steps_created: bool = False
    type: str
    content: str | None = None
    path: Path | None = None


class Asset(PartialAsset, DBModel):
    pass


class Rule(BaseModel):
    strategy: str
    asset_type: str | None = None
    source: str | None = None

    def matches(self, asset: Asset, document: Document) -> bool:
        if self.asset_type is not None and re.match(self.asset_type, asset.type) is None:
            return False

        if self.source is not None and re.match(self.source, document.source_id) is None:
            return False

        return True
