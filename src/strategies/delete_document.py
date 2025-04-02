from core_types import Task
from storage import get_db
from strategies.strategy import Strategy


class DeleteDocumentStrategy(Strategy):
    NAME = "delete_document"
    PRIORITY = 10
    MAX_BATCH_SIZE = 100
    RESOURCES = []

    async def process_all(self, tasks: list[Task]) -> None:
        db = get_db()
        db.delete_documents([task.document_id for task in tasks])