import asyncio
import json
import os
import tempfile
from pathlib import Path

import openai
from pydantic import BaseModel

from core_types import Asset, AssetType, PartialAsset, Task
from logs import logger
from storage import Database
from strategies.strategy import Module


class Segment(BaseModel):
    text: str
    start: float
    end: float


class Transcript(BaseModel):
    segments: list[Segment]


class TranscribeStrategyConfig(BaseModel):
    model: str = "distil-whisper-large-v3-en"


class TranscribeStrategy(Module[TranscribeStrategyConfig]):
    NAME = "transcribe"
    PRIORITY = -1
    MAX_BATCH_SIZE = 1
    INPUT_ASSET_TYPE = AssetType.AUDIO_FILE
    CONFIG_TYPE = TranscribeStrategyConfig

    def __init__(self, config, db: Database):
        super().__init__(config, db)
        self.openai = openai.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    async def process_all(self, tasks: list[Task]):
        assets = self.db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            await self.process_one(task, asset)

    async def process_one(self, task: Task, asset: Asset):

        if self.import_if_exists(task):
            return

        audio_file = asset.path

        logger.info(f"Transcribing audio file {audio_file}")
        transcript = await self.chunked_transcribe(audio_file)

        # Write the transcript to a file
        transcript_file = self.path_for_asset("transcript", f"{task.document_id}.json")
        transcript_file.write_text(transcript.model_dump_json())

        self.db.create_asset(
            PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.TRANSCRIPT,
                path=transcript_file,
            )
        )

    async def chunked_transcribe(self, audio_file: Path):

        # If less than 25MB, we can send it directly
        if audio_file.stat().st_size < 25 * 1024 * 1024:
            transcript = await self.transcribe_short_audio(audio_file, 0)
            return transcript

        # Read the audio file in chunks of 20, and overlap by 1min
        chunk_spacing = 20 * 60
        chunk_overlap = 2 * 60

        transcripts_tasks = []
        audio_duration = await self.get_audio_duration(audio_file)
        with tempfile.TemporaryDirectory() as temp_dir:
            for start in range(0, int(audio_duration + 1), chunk_spacing):
                # Split
                end = start + chunk_spacing + chunk_overlap
                if end > audio_duration:
                    end = audio_duration
                chunk_file = Path(temp_dir) / f"{audio_file.stem}_{start}_{end}.mp3"
                await self.split_audio(audio_file, chunk_file, start, end)

                # Start Transcribing the chunk
                task = asyncio.create_task(self.transcribe_short_audio(chunk_file, start))
                transcripts_tasks.append(task)

            # Wait for all the transcripts to finish
            transcripts = await asyncio.gather(*transcripts_tasks)

        # Merge the transcripts
        segments = []
        current_time = 0
        for i, transcript in enumerate(transcripts):
            # Find a good time to switch to the next chunk
            # Take all the segments that start after the current time until the switchpoint
            segments_past_current_time = [s for s in transcript.segments if s.start >= current_time]

            if i == len(transcripts) - 1:
                segments.extend(segments_past_current_time)
            else:
                # We avoid the last segment as it's likely to be cut,
                # We don't need to do the same for the first, as it's never part of segments_past_current_time
                # as long as we ignore it in the previous chunk.
                next_segments = transcripts[i + 1].segments
                if len(next_segments) > 1:
                    next_segments = next_segments[1:]
                switchpoint = self.find_switchpoint(segments_past_current_time, next_segments)
                segments.extend([s for s in segments_past_current_time if s.start < switchpoint])

                current_time = switchpoint

        return Transcript(segments=segments)

    async def get_audio_duration(self, audio_file: Path) -> float:
        # ffprobe -v error -select_streams a:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 input.mp3
        result = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Read the duration from the output
        if await result.wait() != 0:
            raise Exception(f"Failed to get audio duration for {audio_file}")

        stdout, _ = await result.communicate()
        duration = stdout.decode("utf-8").strip()

        return float(duration)

    async def split_audio(self, audio_file: Path, output: Path, start: int, end: int):
        result = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            audio_file,
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c",
            "copy",
            output,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if await result.wait() != 0:
            stdout, _ = await result.communicate()
            logger.error(f"Failed to split audio for {audio_file}: {stdout.decode('utf-8')}")
            raise Exception(f"Failed to split audio for {audio_file}")

    async def transcribe_short_audio(self, audio_file: Path, start: int) -> Transcript:
        with open(audio_file, "rb") as f:
            transcript = await self.openai.audio.transcriptions.create(
                file=f,
                model=self.config.model,
                response_format="verbose_json",
            )

        return Transcript(
            segments=[
                Segment(text=s.text, start=s.start + start, end=s.end + start)
                for s in transcript.segments
            ]
        )

    def find_switchpoint(self, segments: list[Segment], next_segments: list[Segment]) -> int:
        # We search for the start in next_segments that is closest to the end of the last segment in segments

        distances = {
            (i_first, i_second): abs(first.end - second.start)
            for i_first, first in enumerate(segments)
            for i_second, second in enumerate(next_segments)
        }

        # Find the minimum distance
        first, second = min(distances, key=distances.get)
        distance = distances[(first, second)]

        if distance > 2:
            logger.debug(segments)
            logger.debug(next_segments)
            logger.warning(
                f"Switchpoint distance is large: {segments[first]} -> {segments[second]} ({distance}s)"
            )

        return next_segments[second].start

    def import_if_exists(self, task: Task):
        try:
            pairs = json.loads(Path("src/pairs.json").read_text())
        except FileNotFoundError:
            return False

        if str(task.document_id) not in pairs:
            return False

        original_path = Path(pairs[str(task.document_id)]).with_suffix(".json")
        target_path = self.path_for_asset("transcript", f"{task.document_id}.json")

        try:
            original = json.loads(original_path.read_text())
        except FileNotFoundError as e:
            print(e)
            print(original_path)
            return False

        paragraphs = original["results"]["channels"][0]["alternatives"][0]["paragraphs"][
            "paragraphs"
        ]
        segments = []
        for p in paragraphs:
            for sentence in p["sentences"]:
                segments.append(
                    dict(text=sentence["text"], start=sentence["start"], end=sentence["end"])
                )

        transcript = dict(segments=segments)
        target_path.write_text(json.dumps(transcript))
        logger.info(f"Imported transcript for document {task.document_id}")

        # Create the asset for the downloaded audio file
        self.db.create_asset(
            PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.TRANSCRIPT,
                path=target_path,
            )
        )

        return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)
    file = sys.argv[1]

    if not Path(file).exists():
        print(f"File {file} does not exist")
        sys.exit(1)

    db = None
    config = TranscribeStrategyConfig()
    strategy = TranscribeStrategy(config, db)

    transcript_data = asyncio.run(strategy.chunked_transcribe(Path(file)))

    transcript = "".join([s.text for s in transcript_data.segments])
    print(transcript)
