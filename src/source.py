import abc
from pathlib import Path
from pydantic import BaseModel

from tasks import Task


# How to be up to date?
class Source(BaseModel):
    @abc.abstractmethod
    async def prepare_source(self, document_id: str) -> Path: ...

    @abc.abstractmethod
    async def get_changes(self) -> list[Task]: ...
