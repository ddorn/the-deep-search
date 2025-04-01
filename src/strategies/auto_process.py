from core_types import Task
from strategies.strategy import Strategy


class AutoProcessStrategy(Strategy):
    NAME = "auto_process"
    PRIORITY = 0
    MAX_BATCH_SIZE = 100
    RESOURCES = []

    async def process_all(self, tasks: list[Task]) -> None:
        for task in tasks:
            assert task.strategy == self.NAME

            # What do you execute next? This is autoprocess.
            # So,
            print(task)
