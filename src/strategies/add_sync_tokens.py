
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
        # We want to add tokens quite regularly, probably every 100 characters
        # but it would be nice to have them in nice places:
        # - end of a line
        # - end of a word
        # So we try to put it after ±10% of the chunk size, trying first to find
        # an end of a line there, or if not possible, an end of a word

        chunk_size = 100
        wiggle_room = chunk_size // 5
        text_pieces = []
        start = 0
        while start < len(text):
            candidate_end = start + chunk_size + wiggle_room
            candidate = text[start:candidate_end]

            # try to find a line break
            if (line_break := candidate.rfind("\n")) > chunk_size - wiggle_room:
                new_chunk = candidate[:line_break]
            # try to find a space
            elif (space := candidate.rfind(" ")) > chunk_size - wiggle_room:
                new_chunk = candidate[:space]
            # Just cut it at the chunk size
            else:
                new_chunk = candidate[:chunk_size]

            text_pieces.append(new_chunk)
            start += len(new_chunk)
            # Add a sync token. // 2 because we are adding two tokens per chunk
            text_pieces.append(SYNC_FORMAT.format(id=len(text_pieces) // 2))

        return "".join(text_pieces)
