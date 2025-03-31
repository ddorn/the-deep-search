import abc
from typing import ClassVar

from src.tasks import Task


class Strategy:
    NAME: ClassVar[str]

    @abc.abstractmethod
    def process_all(self: list[Task]) -> None:
        ...
