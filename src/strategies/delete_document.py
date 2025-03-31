from tasks import Task
from strategies.strategy import Strategy


class DeleteDocumentStrategy(Strategy):
    NAME = "delete_document"
    PRIORITY = 10
    MAX_BATCH_SIZE = 100
    RESOURCES = []

    def process_all(self, tasks: list[Task]) -> None:
        raise NotImplementedError("Not implemented yet")