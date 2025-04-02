import mimetypes
import os
from storage import get_db
from core_types import Task, PartialTask
from strategies.strategy import Strategy
from strategies import ChunkFromTextStrategy


class AutoProcessStrategy(Strategy):
    """
    Autoprocess tasks can be created when sources scan new documents.
    This strategy guesses what to do with the document based on the file type.
    """

    NAME = "auto_process"
    PRIORITY = 0
    MAX_BATCH_SIZE = 100
    RESOURCES = []

    async def process_all(self, tasks: list[Task]) -> None:
        for task in tasks:
            assert task.strategy == self.NAME
            self.process(task)

    def process(self, task: Task):
        db = get_db()
        mimetype, encoding = mimetypes.guess_type(task.args)

        if encoding is not None:
            raise ValueError(f"Can't determine actions for task {task} with encoding={encoding}")

        if mimetype is None:
            raise ValueError(f"Can't determine actions for task {task} with mimetype={mimetype}")

        if mimetype.startswith("text/"):
            db.create_task(
                PartialTask(
                    strategy=ChunkFromTextStrategy.NAME,
                    document_id=task.document_id,
                    args=task.args,
                    parent_id=task.id,
                )
            )
        else:
            raise ValueError(f"Can't determine actions for task {task} with mimetype={mimetype}")
