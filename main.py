#!/usr/bin/env -S uv run --frozen

import json
import os
import shlex
import shutil
import subprocess
import sys
from typing import Annotated
import httpx
import typer
import asyncio
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel, Field, AliasPath, AliasChoices
from rich import print as rprint
from openai import AsyncClient
import tempfile
from joblib import Parallel, delayed
import deepgram
import jqpy

from runner import Parallel as MyParallel

app = typer.Typer(no_args_is_help=True, add_completion=False)


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

'''

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

    full_text = f"# {metadata.title}\n\n"
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
'''

@app.command(no_args_is_help=True)
def compress(directory: Path, output: Path):
    """
    Compress all mp3 files in a directory.
    """

    jobs = os.cpu_count()
    if jobs is None:
        jobs = 1
    else:
        jobs = jobs // 1.5

    output.mkdir(exist_ok=True)

    Parallel(n_jobs=jobs)(
        delayed(compress_one)(file, output / file.name) for file in directory.glob("*.mp3")
    )


@app.command(no_args_is_help=True)
def compress_one(path: Path, output: Path = None):

    if output is None:
        output = Path(tempfile.mktemp(suffix=".mp3"))
        compress_one(path, output)
        shutil.move(output, path)
        return

    if output.exists():
        return

    cmd = f"ffmpeg -loglevel quiet -y -i {shlex.quote(str(path))} -codec:a libmp3lame -qscale:a 9 {shlex.quote(str(output))}"
    print(cmd)
    process = subprocess.run(cmd, shell=True)
    if process.returncode != 0:
        print(process.stdout, process.stderr)
        print(process.stderr, process.stderr)
        raise RuntimeError("ffmpeg failed")


@app.command(no_args_is_help=True)
def dl_and_compress(rss_feed: str, directory: Path):
    """
    Download all mp3 files in a directory and compress them.
    """

    # cmd = f'pnpx podcast-dl --url "{shlex.quote(rss_feed)}" --dir "{shlex.quote(str(directory))}"'
    cmd = ["pnpx", "podcast-dl", "--url", rss_feed, "--out-dir", str(directory), "--threads", "3"]
    exec_after = [
        sys.executable,
        __file__,
        "compress-one",
        "EPISODEtemplatEVAR",
    ]
    cmd = cmd + ["--exec", shlex.join(exec_after)]
    cmd_as_str = shlex.join(cmd)
    cmd_as_str = cmd_as_str.replace("EPISODEtemplatEVAR", "{{episode_path}}")

    rprint(cmd_as_str)
    subprocess.run(cmd_as_str, shell=True, check=True)


@app.command(no_args_is_help=True)
def cost(directory: Path, cost_per_minute: float = 0.006):
    """Estimate the cost of transcribing all files in a directory."""

    tasks = [gather_metadata(file) for file in directory.glob("*.mp3")]

    async def gather_all_metadata(tasks):
        return await asyncio.gather(*tasks)

    metadata: list[Podcast] = asyncio.run(gather_all_metadata(tasks))

    total_duration = sum(podcast.duration for podcast in metadata) / 60  # in minutes

    cost = total_duration * cost_per_minute
    rprint(f"Total duration: {total_duration / 60:.2f} hours")
    rprint(f"Estimated cost: ${cost:.2f} at ${cost_per_minute}/minute")


deepgram_client = deepgram.DeepgramClient()

options = deepgram.PrerecordedOptions(
    model="nova-2",
    summarize="v2",
    topics=True,
    intents=True,
    detect_entities=True,
    smart_format=True,
    punctuate=True,
    paragraphs=True,
    diarize=True,
    # sentiment=True,
    language="en",
)


@app.command(no_args_is_help=True)
def deepgram_all(directory: Path, n: int = 1, jobs: int = 1):
    """
    Transcribe all files in a directory using Deepgram.
    """

    files = sorted(directory.glob("*.mp3"), reverse=True)
    files = [file for file in files if not file.with_suffix(".json").exists()]
    files = files[:n]
    rprint(files)

    runner = MyParallel(n_jobs=jobs)
    for file in files:
        runner.add(deepgram_one, title=file.stem)(file)

    results = runner.run()
    for file, result in zip(files, results):
        if result.is_error:
            rprint(f"[red]Error processing {file}[/red]")
            rprint(f"[red]{result.unwrap_error()}[/red]")


@app.command(no_args_is_help=True)
def deepgram_one(file: Path):
    output = file.with_suffix(".json")
    if output.exists():
        return

    rprint(f"Transcribing '{file}'")
    with open(file, "rb") as audio:
        source = {"buffer": audio}

        response: deepgram.PrerecordedResponse = deepgram_client.listen.rest.v("1").transcribe_file(
            source, options, timeout=httpx.Timeout(100, connect=10)
        )

    output.write_text(response.to_json())
    rprint(f"Transcribed '{file}'")


@app.command(no_args_is_help=True)
def transcript(directory: Path):
    """
    Convert the transcriptions in a directory to markdown.
    """

    for file in tqdm(list(directory.glob("*.json"))):
        data = json.loads(file.read_text())
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        with open(file.with_suffix(".md"), "w") as f:
            f.write(transcript)



if __name__ == "__main__":
    app()
