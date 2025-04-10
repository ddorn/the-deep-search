from pathlib import Path
import asyncio

from pydantic import BaseModel

from core_types import Rule, Task
from storage import get_db
from strategies.strategy import Module


class CompressAudioInPlaceConfig(BaseModel):
    level: int = 9


class CompressAudioInPlaceStrategy(Module[CompressAudioInPlaceConfig]):
    NAME = "compress_audio"
    PRIORITY = 5  # Needs to run before other strategies that use audio. Only transcribe currently does.
    MAX_BATCH_SIZE = 1
    CONFIG_TYPE = CompressAudioInPlaceConfig

    def add_rules(self, rules: list[Rule]) -> list[Rule]:
        return rules

    async def process_all(self, tasks: list[Task]):
        db = get_db()

        assets = db.get_assets([task.input_asset_id for task in tasks])
        paths = [asset.path for asset in assets]

        for path in paths:
            await self.compress_file_in_place(path)

    async def compress_file_in_place(self, path: Path):
        
        output_path = path.with_suffix(f"{path.suffix}_compressed.mp3")
        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-c:a", "libmp3lame",
            "-q:a", str(self.config.level),
            "-y",
            str(output_path)
        ]
        
        print(f"Compressing {path.name}...")
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Replace original with compressed version
                output_path.replace(path)
                print(f"Replaced original file with compressed version")
            else:
                print(f"Error compressing {path.name}:")
                print(f"Return code: {process.returncode}")
                print(f"Error output: {stderr.decode()}")
        except Exception as e:
            print(f"Error compressing {path.name}: {str(e)}")
            raise e
