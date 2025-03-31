import abc
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel

from config import SourceConfig
from constants import DIRS


class ExtraConfig(BaseModel):
    pass

class Source[ExtraConfigType: BaseModel](abc.ABC):
    """
    Sources of data for the search engine.

    The main responsability of this class is to generate tasks
    to keep up to date with the changes in the source.
    """

    NAME: str
    EXTRA_CONFIG: type[ExtraConfigType]

    def __init__(self, title: str, config: SourceConfig):
        self.title = title
        """The title of the source of data. Each type of source can exist multiple times with different titles."""

        assert config.type == self.NAME
        self.config = self.EXTRA_CONFIG.model_validate(config.args)

    @abc.abstractmethod
    async def add_tasks_from_changes(self):
        ...

    @cached_property
    def data_folder(self) -> Path:
        folder = DIRS.user_data_path / "sources" / self.title
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @cached_property
    def cache_folder(self) -> Path:
        folder = DIRS.user_cache_path / "sources" / self.title
        folder.mkdir(parents=True, exist_ok=True)
        return folder
