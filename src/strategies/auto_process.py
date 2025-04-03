import mimetypes
import os
from storage import get_db
from core_types import AssetType, PartialAsset, Rule, Task, PartialTask
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

    def add_rules(self, rules: list[Rule]) -> list[Rule]:
        return rules + [
            Rule(pattern=AssetType.GENERIC_FILE, strategy=self.NAME),
        ]

    async def process_all(self, tasks: list[Task]) -> None:
        for task in tasks:
            assert task.strategy == self.NAME
            self.process(task)

    def process(self, task: Task):
        db = get_db()
        asset = db.get_asset(task.input_asset_id)
        assert asset is not None
        path = asset.path
        assert path is not None
        mimetype, encoding = mimetypes.guess_type(path)

        if encoding is not None:
            raise ValueError(f"Can't determine actions for task {task} with encoding={encoding}")

        if mimetype is None:
            raise ValueError(f"Can't determine actions for task {task} with mimetype={mimetype}")

        if mimetype.startswith("text/"):
            db.create_asset(PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.TEXT_FILE,
                path=path,
            ))
        else:
            raise ValueError(f"Can't determine actions for task {task} with mimetype={mimetype}")
