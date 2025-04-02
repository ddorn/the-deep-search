from contextlib import contextmanager
from functools import cached_property
from importlib import import_module
import importlib
import importlib.util
from pathlib import Path
import abc
from typing import ClassVar

from openai import BaseModel

from constants import DIRS
from core_types import PartialByproduct, Task
from storage import get_db

NOT_GIVEN = object()

class NoConfig(BaseModel):
    pass


class Module[ConfigType: BaseModel](abc.ABC):
    NAME: ClassVar[str]
    CONFIG_TYPE: type[ConfigType] = NoConfig  # noqa: ignore[assignment]

    def __init__(self, config: ConfigType):
        self.config = config

    def data_folder_name(self) -> str:
        """
        The name of the folder where the module stores its data.
        """
        return self.NAME

    @cached_property
    def data_folder(self) -> Path:
        folder = DIRS.user_data_path / "modules" / self.data_folder_name()
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def byproduct_path(self, kind: str, name: str) -> Path:
        folder = self.data_folder / kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    # @contextmanager
    # def open_byproduct(self, kind: str, name: str, mode: str, **kwargs):
    #     byproduct_path = self.byproduct_path(kind, name)
    #     with byproduct_path.open(mode, **kwargs) as f:
    #         yield f

    def write_byproduct(self, kind: str, name: str, document_id: int, content: str):
        byproduct_path = self.byproduct_path(kind, name)

        get_db().create_byproduct(
            PartialByproduct(
                document_id=document_id,
                path=byproduct_path,
            )
        )

        with byproduct_path.open("w") as f:
            f.write(content)

    def read_byproduct(self, kind: str, name: str, default=NOT_GIVEN):
        byproduct_path = self.byproduct_path(kind, name)

        try:
            return byproduct_path.read_text()
        except FileNotFoundError:
            if default is NOT_GIVEN:
                raise
            return default

    @contextmanager
    def mount(self):
        """
        Context manager to mount and unmount the module on exit.
        """

        self.on_mount()
        try:
            yield
        finally:
            self.on_unmount()

    def on_mount(self):
        """
        Called when the module is mounted.
        """
        pass

    def on_unmount(self):
        """
        Called when the module is unmounted.
        """
        pass


class Strategy[ConfigType: BaseModel](Module[ConfigType]):
    """
    A strategy describes how to do a specific type of task.
    """

    NAME: ClassVar[str]
    PRIORITY: ClassVar[int]
    MAX_BATCH_SIZE: ClassVar[int]
    RESOURCES: ClassVar[list[str]]

    def __init__(self, config: ConfigType):
        super().__init__(config)

        for attribute in ["NAME", "PRIORITY", "MAX_BATCH_SIZE", "RESOURCES"]:
            if not hasattr(self, attribute):
                raise ValueError(f"Strategy {self.__class__.__name__} must have a {attribute} attribute")

    @abc.abstractmethod
    async def process_all(self, tasks: list[Task]) -> None:
        ...


class Source[ConfigType: BaseModel](Module[ConfigType]):

    def __init__(self, config: ConfigType, title: str):
        super().__init__(config)
        self.title = title

    def data_folder_name(self):
        return self.title
