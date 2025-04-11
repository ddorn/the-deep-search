import abc
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from openai import BaseModel

from constants import DIRS
from core_types import Rule, Task

NOT_GIVEN = object()


class NoConfig(BaseModel):
    pass


class Module[ConfigType: BaseModel](abc.ABC):
    NAME: ClassVar[str]
    CONFIG_TYPE: type[ConfigType] = NoConfig

    MAX_BATCH_SIZE: ClassVar[int] = 1
    PRIORITY: ClassVar[int] = 0
    INPUT_ASSET_TYPE: ClassVar[str | None] = None

    def __init__(self, config: ConfigType):
        self.config = config

    async def process_all(self, tasks: list[Task]) -> None:
        """
        Processes a batch of assets.

        It is assumed that:
        - All tasks target this strategy
        - The input asset matches the rules from `add_rules`, or is otherwise suitable for processing by this strategy.
        - There are at most MAX_BATCH_SIZE tasks in the batch.

        Those assumptions should not be strictly enforced, as it enable
        other strategies/plugins to provide different but suitable inputs.
        """
        raise NotImplementedError()

    def add_rules(self, rules: list[Rule]) -> list[Rule]:
        """
        Modifies the rules list to tell the executor what this tasks need to process.

        This can also modify rules previously added by other strategies, but needs to be done carefully.
        """
        if self.INPUT_ASSET_TYPE is None:
            return rules
        else:
            return rules + [
                Rule(asset_type=self.INPUT_ASSET_TYPE, strategy=self.NAME),
            ]

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

    def path_for_asset(self, kind: str, name: str) -> Path:
        folder = self.data_folder / kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

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


class Source[ConfigType: BaseModel](Module[ConfigType]):

    def __init__(self, config: ConfigType, title: str):
        super().__init__(config)
        self.title = title

    def data_folder_name(self):
        return self.title
