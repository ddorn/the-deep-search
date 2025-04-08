import os
from pathlib import Path
import openai
from pydantic import BaseModel

from core_types import Asset, AssetType, PartialAsset, Task
from storage import get_db
from strategies.strategy import Module


class TranscribeStrategyConfig(BaseModel):
    model: str = "distil-whisper-large-v3-en"


class TranscribeStrategy(Module[TranscribeStrategyConfig]):
    NAME = "transcribe"
    PRIORITY = 0
    MAX_BATCH_SIZE = 5
    INPUT_ASSET_TYPE = AssetType.AUDIO_FILE
    CONFIG_TYPE = TranscribeStrategyConfig

    def __init__(self, config):
        super().__init__(config)
        self.openai = openai.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    async def process_all(self, tasks: list[Task]):
        db = get_db()

        assets = db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            await self.process_one(task, asset)


    async def process_one(self, task: Task, asset: Asset):

        audio_file = asset.path

        with open(audio_file, "rb") as f:
            transcript = await self.openai.audio.transcriptions.create(
                file=(str(audio_file), f.read()),
                model=self.config.model,
                response_format="verbose_json",
            )

        # Write the transcript to a file
        transcript_file = self.path_for_asset("transcript", str(task.document_id))
        transcript_file.write_text(transcript.model_dump_json())

        db = get_db()

        db.create_asset(
            PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.TRANSCRIPT,
                path=transcript_file,
            )
        )
