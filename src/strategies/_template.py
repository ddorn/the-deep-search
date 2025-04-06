from core_types import PartialAsset, Task
from storage import get_db
from strategies.strategy import Strategy


class TODOStrategy(Strategy):
    NAME = ...
    PRIORITY = ...
    MAX_BATCH_SIZE = ...
    RESOURCES = [...]

    async def process_all(self, tasks: list[Task]) -> None:
        raise NotImplementedError("Not implemented yet")

        db = get_db()
        assets = db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            db.create_asset(PartialAsset(
                created_by_task_id=task.id,
                document_id=task.document_id,
                type=...,
                content=...,  # One of the two
                path=...,
            ))