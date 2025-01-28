#!/usr/bin/env -S uv run --frozen

import os
import shlex
import subprocess
from typing import Annotated
import typer
import asyncio
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel, Field, AliasPath, AliasChoices
from rich import print as rprint
from openai import AsyncClient
import tempfile

app = typer.Typer()


class Chapter(BaseModel):
    start_time: float
    end_time: float
    # Title is loaded from chapter['tags']['title']
    title: Annotated[str, Field(alias=AliasChoices("title", AliasPath("tags", "title")))] = ""


class Podcast(BaseModel):
    chapters: list[Chapter]
    filename: Annotated[Path, Field(alias=AliasPath("format", "filename"))]
    duration: Annotated[float, Field(alias=AliasPath("format", "duration"))]
    title: Annotated[str, Field(alias=AliasPath("format", "tags", "title"))]


async def gather_metadata(podcast: Path) -> Podcast:
    cmd = f"ffprobe -show_chapters -show_format -print_format json -loglevel quiet {shlex.quote(str(podcast))}"

    process = await asyncio.create_subprocess_shell(cmd, stdout=subprocess.PIPE)

    stdout, stderr = await process.communicate()
    if stderr:
        rprint(stderr.decode())
    if process.returncode != 0:
        raise RuntimeError("ffprobe failed")

    return Podcast.model_validate_json(stdout)


def ensure_chapters_cover_podcast(podcast: Podcast) -> Podcast:
    """Add chapters for parts that are not covered by chapters."""

    current_time = 0.0
    chapters = []
    for chapter in podcast.chapters:
        if chapter.start_time > current_time + 1:
            chapters.append(
                Chapter(
                    start_time=current_time,
                    end_time=chapter.start_time,
                )
            )
        chapters.append(chapter)
        current_time = chapter.end_time

    if current_time < podcast.duration:
        chapters.append(
            Chapter(
                start_time=current_time,
                end_time=podcast.duration,
            )
        )

    return podcast.model_copy(update={"chapters": chapters})


async def run_in_parrallel(tasks, max_concurrent: int, progress=True):
    """Run async tasks in parallel with a maximum concurrency and a progress bar."""
    semaphore = asyncio.Semaphore(max_concurrent)

    if progress:
        bar = tqdm(total=len(tasks))

    async def run(task):
        async with semaphore:
            result = await task
            if progress:
                bar.update()
            return result

    try:
        return await asyncio.gather(*[run(task) for task in tasks])
    finally:
        if progress:
            bar.close()


@app.command()
def main(podcast_dir: Path, debug: bool = False, jobs: int = 20, n: int = 1):
    """
    This script will take a directory of podcast files and output a directory of their transcription as text files.
    """

    asyncio.run(true_main(podcast_dir, debug=debug, jobs=jobs, n=n))


async def true_main(podcast_dir: Path, debug: bool = False, jobs: int = 20, n: int = 1):
    podcasts = sorted((podcast_dir.glob("*.mp3")), reverse=True)

    if n < 0:
        n = len(podcasts)

    tasks = [process_one(podcast) for podcast in podcasts[:n]]

    if debug:
        jobs = 1

    await run_in_parrallel(tasks, max_concurrent=jobs)


async def process_one(podcast: Path):
    output = podcast.with_suffix(".txt")
    if output.exists():
        return

    rprint(f"Processing {shlex.quote(str(podcast))}")

    metadata = await gather_metadata(podcast)
    metadata = ensure_chapters_cover_podcast(metadata)

    chapters_folders = podcast.with_name(podcast.stem + "_chapters")
    chapter_paths = [chapters_folders / f"{i:02d}.mp3" for i in range(len(metadata.chapters))]
    transcripts_paths = [chapter_path.with_suffix(".txt") for chapter_path in chapter_paths]
    chapters_folders.mkdir(exist_ok=True)

    for chapter, audio, transcript in zip(
        metadata.chapters, chapter_paths, transcripts_paths, strict=True
    ):
        if transcript.exists():
            continue
        await split_chapter(podcast, chapter, audio)
        await transcribe(audio, transcript)
        # audio.unlink()

    full_text = ""
    for chapter, transcript_path in zip(metadata.chapters, transcripts_paths):
        if chapter.title:
            full_text += f"## {chapter.title}\n\n"
        full_text += transcript_path.read_text() + "\n\n"

    output.write_text(full_text)


async def split_chapter(podcast: Path, chapter: Chapter, chapter_path: Path):
    cmd = f"ffmpeg -loglevel quiet -y -i {shlex.quote(str(podcast))} -ss {chapter.start_time} -to {chapter.end_time} -codec:a libmp3lame -qscale:a 9 {shlex.quote(str(chapter_path))}"
    process = await asyncio.create_subprocess_shell(cmd)
    await process.communicate()
    if process.returncode != 0:
        raise RuntimeError("ffmpeg failed")


client = AsyncClient()


async def transcribe(audio: Path, text_output: Path):
    if text_output.exists():
        return

    file_size = audio.stat().st_size

    # Check that the file is less than 25Mb
    if file_size > 25 * 1024 * 1024:
        raise ValueError("File is too large")


    print(f"Transcribing {audio} ({file_size / 1024 / 1024:.2f}Mb)")

    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=audio.open("rb"),
    )

    text_output.write_text(response.text)



@app.command()
def compress(directory: Path, output: Path):
    """
    Compress all mp3 files in a directory.
    """

    output.mkdir(exist_ok=True)
    tasks = [compress_one(file, output / file.name) for file in directory.glob("*.mp3")]

    jobs = os.cpu_count()
    if jobs is None:
        jobs = 1
    else:
        jobs = jobs // 1.5

    asyncio.run(run_in_parrallel(tasks, max_concurrent=jobs))


async def compress_one(path: Path, output: Path):
    if output.exists():
        return

    cmd = f"ffmpeg -loglevel quiet -y -i {shlex.quote(str(path))} -codec:a libmp3lame -qscale:a 9 {shlex.quote(str(output))}"
    process = await asyncio.create_subprocess_shell(cmd)
    await process.communicate()
    if process.returncode != 0:
        raise RuntimeError("ffmpeg failed")



if __name__ == "__main__":
    app()
