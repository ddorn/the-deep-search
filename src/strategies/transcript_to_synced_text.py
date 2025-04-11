import json

from constants import SYNC_FORMAT
from core_types import AssetType, PartialAsset, Task
from storage import get_db
from strategies.strategy import Module


class TranscriptToSyncedTextStrategy(Module):
    NAME = "transcript_to_synced_text"
    PRIORITY = 0
    MAX_BATCH_SIZE = 1

    INPUT_ASSET_TYPE = AssetType.TRANSCRIPT

    async def process_all(self, tasks: list[Task]) -> None:
        db = get_db()
        assets = db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            # Read the transcript JSON file
            transcript_data = json.loads(asset.path.read_text())
            segments = transcript_data["segments"]

            # Process segments into plain text with sync tokens
            processed_text = []

            for segment in segments:
                text = segment["text"]
                start_time = segment["start"]

                # Add sync token with timestamp as ID
                sync_token = SYNC_FORMAT.format(id=f"{start_time:.2f}")
                processed_text.append(sync_token + text)

            # Join all segments adding a space between them, if they don't start or end with one.
            final_text = " ".join(segment.strip() for segment in processed_text if segment)

            # Write the processed text
            out_path = self.path_for_asset("sync_tokens", f"{task.document_id}.txt")
            out_path.write_text(final_text)

            # Create new asset
            db.create_asset(
                PartialAsset(
                    document_id=task.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.SYNCED_TEXT_FILE,
                    content=None,
                    path=out_path,
                )
            )
