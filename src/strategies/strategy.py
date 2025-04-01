from importlib import import_module
from pathlib import Path
import abc
from typing import ClassVar

from core_types import Task


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
    async def process_all(self, tasks: list[Task]) -> None:
        ...


def collect_built_in_strategies() -> dict[str, type[Strategy]]:
    strategies = {}

    for path in Path(__file__).parent.rglob("*.py"):
        module = import_module(f".{path.stem}", __package__)
        for attribute in dir(module):
            obj = getattr(module, attribute)
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj != Strategy:
                strategies[obj.NAME] = obj

    return strategies
