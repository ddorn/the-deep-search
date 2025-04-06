
from constants import SYNC_FORMAT
from core_types import AssetType, PartialAsset, Task
from storage import get_db
from strategies.strategy import Strategy


class AddSyncTokenStrategy(Strategy):
    NAME = "add_sync_tokens"
    PRIORITY = 1
    MAX_BATCH_SIZE = 100

    INPUT_ASSET_TYPE = AssetType.TEXT_FILE

    async def process_all(self, tasks: list[Task]) -> None:
        db = get_db()
        assets = db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            text = asset.path.read_text()
            with_sync_tokens = self.add_sync_tokens(text)
            out_path = self.path_for_asset("sync_tokens", asset.path.name)
            out_path.write_text(with_sync_tokens)

            db.create_asset(PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.SYNCED_TEXT_FILE,
                content=None,
                path=out_path,
            ))


    def add_sync_tokens(self, text: str) -> str:
        # We want to add tokens quite regularly, probably every ~100 characters
        chunk_size = 100
        text_parts = []
        for idx, start in enumerate(range(0, len(text), chunk_size)):
            text_parts.append(SYNC_FORMAT.format(id=idx))
            text_parts.append(text[start:start + chunk_size])

        return ''.join(text_parts)
