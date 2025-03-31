from importlib import import_module
from pathlib import Path
import abc
from typing import ClassVar

from src.tasks import Task


class Strategy:
    """
    A strategy describes how to do a specific type of task.
    """

    NAME: ClassVar[str]
    PRIORITY: ClassVar[int]
    MAX_BATCH_SIZE: ClassVar[int]
    RESOURCES: ClassVar[list[str]]

    def __post_init__(self):
        for attribute in ["NAME", "PRIORITY", "MAX_BATCH_SIZE", "RESOURCES"]:
            if not hasattr(self, attribute):
                raise ValueError(f"Strategy {self.__class__.__name__} must have a {attribute} attribute")

    @abc.abstractmethod
    def process_all(self, tasks: list[Task]) -> None:
        ...
