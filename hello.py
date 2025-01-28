#!/usr/bin/env -S uv run --frozen

import os
import shlex
import subprocess
from typing import Annotated
import typer
import asyncio
import tqdm
from pathlib import Path
from pydantic import BaseModel, Field, AliasPath
from rich import print as rprint

app = typer.Typer()


class Chapter(BaseModel):
    start_time: float
    end_time: float
    # Title is loaded from chapter['tags']['title']
    title: Annotated[str, Field(alias=AliasPath("tags", "title"))]


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
        print("megafail")
        raise RuntimeError("ffprobe failed")

    # Load the chapters
    return Podcast.model_validate_json(stdout)


async def run_in_parrallel(tasks, max_concurrent: int):
    semaphore = asyncio.Semaphore(max_concurrent)
    progress = tqdm.tqdm(total=len(tasks))

    async def run(task):
        async with semaphore:
            result = await task
            progress.update()
            return result

    return await asyncio.gather(*[run(task) for task in tasks])


@app.command()
def main(podcast_dir: Path, debug: bool = False):
    """
    This script will take a directory of podcast files and output a directory of their transcription as text files.
    """

    asyncio.run(true_main(podcast_dir, debug))


async def true_main(podcast_dir: Path, debug: bool = False):
    output = podcast_dir / "raw_transcriptions"
    podcasts = sorted((podcast_dir.glob("*.mp3")), reverse=True)

    tasks = [process_one(podcast) for podcast in podcasts]

    if debug:
        max_concurrent = 1
    else:
        max_concurrent = os.cpu_count() or 1

    await run_in_parrallel(tasks, max_concurrent=max_concurrent)


async def process_one(podcast: Path):
    rprint(f"Processing {podcast}")
    metadata = await gather_metadata(podcast)
    rprint(metadata)

    # Add chapters for parts that are not covered by chapters
    current_time = 0.0
    chapters = []
    for chapter in metadata.chapters:
        if chapter.start_time > current_time + 1:
            chapters.append(
                Chapter(
                    start_time=current_time,
                    end_time=chapter.start_time,
                    title="Untitled part",
                )
            )
        chapters.append(chapter)
        current_time = chapter.end_time

    if current_time < metadata.duration:
        chapters.append(
            Chapter(
                start_time=current_time,
                end_time=metadata.duration,
                title="Untitled part",
            )
        )
    metadata.chapters = chapters

    rprint(metadata)
    # for chapter in chapters:
        # print(f"  {chapter.title} ({chapter.start} - {chapter.end})")


if __name__ == "__main__":
    app()
