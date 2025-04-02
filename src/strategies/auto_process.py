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
            assert (
                task.strategy == self.NAME
            )  # Executor gave a task belonging to another strategy
            path = task.args
            _, ext = os.path.splitext(path)
            if ext.startswith("."):
                ext = ext[1:]

            # We could get smart here and look at magic numbers,
            # entropy... this is especially relevant for text files.
            self.process(task, ext, path)

    def process(self, task: Task, ext: str, path: str):
        db = get_db()
        if ext in ["md", "txt"]:
            db.create_task(
                PartialTask(
                    strategy=ChunkFromTextStrategy.NAME,
                    document_id=task.document_id,
                    args=path,
                    parent_id=task.id,
                )
            )
        else:
            raise ValueError(
                f"Could not autoprocess task {task.id} with ext={ext} and path={path}"
            )
