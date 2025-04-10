import json
from pathlib import Path

import aiohttp
from pydantic import BaseModel

from core_types import AssetType, PartialAsset, PartialTask, Task
from logs import logger
from storage import get_db
from strategies.compress_audio import CompressAudioInPlaceStrategy
from strategies.strategy import Module


class DlAudioConfig(BaseModel):
    compress: bool = True


class DlAudioStrategy(Module[DlAudioConfig]):
    NAME = "dl_audio"
    PRIORITY = 0
    MAX_BATCH_SIZE = 1
    INPUT_ASSET_TYPE = AssetType.AUDIO_TO_DL

    async def process_all(self, tasks: list[Task]):
        db = get_db()

        assets = db.get_assets([task.input_asset_id for task in tasks])

        async with aiohttp.ClientSession() as session:
            for asset, task in zip(assets, tasks, strict=True):
                await self.process_one(asset, task, session)

    async def process_one(self, asset, task, session):
        db = get_db()

        if self.import_if_exists(task):
            return

        # TODO: the file might not be mp3. I think it's fine if it's compressed
        # as ffmpeg can handle anytime, even if the extension is wrong.
        audio_file_path = self.path_for_asset("original_audio", f"{task.document_id}.mp3")

        with open(audio_file_path, "wb") as f:
            async with session.get(asset.content) as response:
                logger.info(f"Downloading audio file from {asset.content} to {audio_file_path}")
                if response.status != 200:
                    raise ValueError(f"Failed to download audio file: {response.status}")
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)

        # Create the asset for the downloaded audio file
        output_asset = db.create_asset(
            PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.AUDIO_FILE,
                path=audio_file_path,
            )
        )

        if self.config.compress:
            db.create_task(
                PartialTask(
                    document_id=task.document_id,
                    strategy=CompressAudioInPlaceStrategy.NAME,
                    input_asset_id=output_asset,
                    one_shot=True,
                )
            )

    def import_if_exists(self, task: Task):
        try:
            pairs = json.loads(Path("src/pairs.json").read_text())
        except FileNotFoundError:
            return False

        if str(task.document_id) not in pairs:
            return False

        original_path = pairs[str(task.document_id)]
        target_path = self.path_for_asset("original_audio", str(task.document_id))

        # Move the file to the target path
        Path(original_path).rename(target_path)
        logger.info(f"Moved file from {original_path} to {target_path}")

        # Create the asset for the downloaded audio file
        db = get_db()
        db.create_asset(
            PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.AUDIO_FILE,
                path=target_path,
            )
        )

        return True
