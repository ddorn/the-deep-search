from core_types import Task
from strategies.strategy import Strategy


class TODOStrategy(Strategy):
    NAME = ...
    PRIORITY = ...
    MAX_BATCH_SIZE = ...
    RESOURCES = [...]

    async def process_all(self, tasks: list[Task]) -> None:
        raise NotImplementedError("Not implemented yet")