import aiohttp
from core_types import AssetType, Task, PartialAsset
from strategies.strategy import Module
from storage import get_db
from logs import logger


class DlAudioStrategy(Module):
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


        audio_file_path = self.path_for_asset('original_audio', str(task.document_id))

        with open(audio_file_path, 'wb') as f:
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
        db.create_asset(PartialAsset(
            document_id=task.document_id,
            created_by_task_id=task.id,
            type=AssetType.AUDIO_FILE,
            path=audio_file_path,
        ))