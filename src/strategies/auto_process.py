import mimetypes

from core_types import Asset, AssetType, PartialAsset, Task
from storage import get_db
from strategies.strategy import Module


class AutoProcessStrategy(Module):
    """
    Autoprocess tasks can be created when sources scan new documents.
    This strategy guesses what to do with the document based on the file type.
    """

    NAME = "auto_process"
    PRIORITY = 0
    MAX_BATCH_SIZE = 100
    INPUT_ASSET_TYPE = AssetType.GENERIC_FILE

    async def process_all(self, tasks: list[Task]) -> None:
        assets = get_db().get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            self.process(task, asset)

    def process(self, task: Task, asset: Asset):
        db = get_db()
        path = asset.path
        assert path is not None
        mimetype, encoding = mimetypes.guess_type(path)

        if encoding is not None:
            raise ValueError(f"Can't determine actions for asset {asset} with encoding={encoding}")

        if mimetype is None:
            raise ValueError(f"Can't determine actions for asset {asset} with mimetype={mimetype}")

        if mimetype.startswith("text/"):
            db.create_asset(
                PartialAsset(
                    document_id=asset.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.TEXT_FILE,
                    path=path,
                )
            )
        else:
            raise ValueError(f"Can't determine actions for asset {asset} with mimetype={mimetype}")
