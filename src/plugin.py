
import abc
from pathlib import Path
from openai import BaseModel

from tasks import Task


class Plugin(BaseModel):

    @abc.abstractmethod
    async def prepare_source(self, document_id: str) -> Path:
        ...

    @abc.abstractmethod
    async def get_changes(self) -> list[Task]:
        ...
