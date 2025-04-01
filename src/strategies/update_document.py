from core_types import Task
from strategies.strategy import Strategy


class UpdateDocumentStrategy(Strategy):
    NAME = "update_document"
    PRIORITY = 10
    MAX_BATCH_SIZE = 100
    RESOURCES = []

    async def process_all(self, tasks: list[Task]) -> None:
        raise NotImplementedError("Not implemented yet")
