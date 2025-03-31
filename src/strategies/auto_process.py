from tasks import Task
from strategies.strategy import Strategy


class AutoProcessStrategy(Strategy):
    NAME = "auto_process"
    PRIORITY = 0
    MAX_BATCH_SIZE = 100
    RESOURCES = []

    def process_all(self, tasks: list[Task]) -> None:
        raise NotImplementedError("Not implemented yet")