#!/usr/bin/env -S uv run --frozen

import itertools
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
import tempfile
from joblib import Parallel, delayed
import deepgram
import openai


import databases
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


def gather_metadata(podcast: Path) -> Podcast:
    cmd = f"ffprobe -show_chapters -show_format -print_format json -loglevel quiet {shlex.quote(str(podcast))}"

    output = subprocess.check_output(cmd, shell=True)
    return Podcast.model_validate_json(output)

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

    metadata = gather_metadata(podcast)
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

    runner = MyParallel(n_jobs=10, show_done=False, show_threads=False)
    for file in directory.glob("*.mp3"):
        runner.add(gather_metadata, title=file.stem)(file)
    metadata_results = runner.run()

    total_duration = sum(podcast.unwrap().duration for podcast in metadata_results) / 60  # in minutes

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
        with file.open() as f:
            data = json.load(f)
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        with open(file.with_suffix(".txt"), "w") as f:
            f.write(transcript)


def get_embedding(text: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        dimensions=databases.EMBDEDDING_DIMENSIONS,
        model="text-embedding-3-small",
        input=text,
    )

    out: list[list[float]] = [None] * len(text)
    for embedding in response.data:
        out[embedding.index] = embedding.embedding

    return out



@app.command(no_args_is_help=True)
def to_sql(directory: Path):
    """
    Add all transcriptions in a directory to a SQLite database.
    """

    db = databases.Databases(directory, milvus=False)

    for file in tqdm(list(directory.glob("*.json"))):
        with open(file) as f:
            transcript = json.load(f)

        # Add the podcast to the sql database, if it doesn't exist
        if db.get_podcast_by_filename(file.stem) is not None:
            # Podcast already exists
            continue
        else:
            metadata = gather_metadata(file.with_suffix(".mp3"))
            podcast_id = db.add_podcast_sql(databases.Podcast(id=None, title=metadata.title, filename=file.stem))

        paragraphs = transcript["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]
        paragraph_texts = [" ".join([sentence["text"] for sentence in paragraph["sentences"]]) for paragraph in paragraphs]

        db.add_paragraphs_sql([ databases.Paragraph.from_text(p, podcast_id, i) for i, p in enumerate(paragraph_texts) ])

    db.sql.commit()



@app.command(no_args_is_help=True)
def to_milvus(directory: Path):
    """
    store the embeddings of all transcriptions from the sql database to milvus.
    """

    db = databases.Databases(directory)

    rprint("Connection to Milvus and SQL established :tada:")

    in_milvus = db.get_all_milvus_ids()
    in_sql = db.get_all_paragraph_hashes()

    to_insert: set[str] = in_sql - in_milvus
    rprint(f"SQL: {len(in_sql)} Milvus: {len(in_milvus)} To insert: {len(to_insert)}")

    if len(to_insert) == 0:
        rprint("[green]Milvus is up to date!")
        return

    def process_one(batch: list[str]):
        paragraphs = db.get_paragraphs(batch)
        texts = [p.text for p in paragraphs]

        n_words = sum(len(t.split()) for t in texts)
        runner.progress.log(f"Embedding {len(paragraphs)} paragraphs ({n_words} words)")

        embeddings = get_embedding(texts)
        db.milvus_add(texts, embeddings)

    runner = MyParallel(progress_per_task=False, show_threads=False)
    for batch in itertools.batched(to_insert, 1000):
        runner.add(process_one)(batch)

    runner.run()


@app.command(no_args_is_help=True)
def search(directory: Path, query: str):
    """
    Search for a query in a directory of transcriptions.
    """

    db = databases.Databases(directory)

    embedding = get_embedding([query])[0]

    paragraphs = db.search(embedding)
    podcasts = db.get_podcasts(list({p.podcast_id for p in paragraphs}))

    # Group the paragraphs by podcast, using itertool's groupby
    paragraphs = sorted(paragraphs, key=lambda p: p.podcast_id)
    paragraphs_by_podcasts = {
        podcast_id: list(group)
        for podcast_id, group in itertools.groupby(paragraphs, key=lambda p: p.podcast_id)
    }

    for podcast_id, podcast_paragraphs in paragraphs_by_podcasts.items():
        podcast = next(p for p in podcasts if p.id == podcast_id)
        rprint(f"[green]{podcast.title}[/green]")
        for paragraph in podcast_paragraphs:
            rprint(paragraph.text)
        rprint("")


if __name__ == "__main__":
    app()
