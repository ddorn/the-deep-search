#!/usr/bin/env -S uv run --frozen
import asyncio
import contextlib
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated

import databases
import deepgram
import httpx
import numpy as np
import openai
import typer
from joblib import Parallel, delayed
from pydantic import AliasChoices, AliasPath, BaseModel, Field
from rich import print as rprint
from runner import Parallel as MyParallel
from tqdm import tqdm

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
    Download all mp3 files in a directory and compress them in place.
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

    total_duration = (
        sum(podcast.unwrap().duration for podcast in metadata_results) / 60
    )  # in minutes

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
            podcast_id = db.add_podcast_sql(
                databases.Podcast(id=None, title=metadata.title, filename=file.stem)
            )

        paragraphs = transcript["results"]["channels"][0]["alternatives"][0]["paragraphs"][
            "paragraphs"
        ]
        paragraph_texts = [
            " ".join([sentence["text"] for sentence in paragraph["sentences"]])
            for paragraph in paragraphs
        ]

        db.add_paragraphs_sql(
            [
                databases.Paragraph.from_text(podcast_id, text, i)
                for i, text in enumerate(paragraph_texts)
            ]
        )

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
def search_old(directory: Path, query: str):
    """
    Search for a query in a directory of transcriptions.
    """

    embedding = get_embedding([query])[0]
    db = databases.Databases(directory)
    with timeit():

        paragraphs = db.search(embedding)
        rprint(*paragraphs)
        return

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


@app.command(no_args_is_help=True)
def migrate_milvus(directory: Path):
    """
    Migrate the milvus database to a new file.
    """

    db = databases.Databases(directory)

    print(db.milvus.get_collection_stats(db.collection_name))
    progress = tqdm(total=db.milvus.get_collection_stats(db.collection_name)["row_count"])

    ids = np.empty((0,), dtype=str)
    embeddings = np.empty((0, databases.EMBDEDDING_DIMENSIONS), dtype=float)

    iterator = db.milvus.query_iterator(
        db.collection_name, output_fields=["id", "vector"], batch_size=100
    )
    while True:
        res = iterator.next()
        if len(res) == 0:
            break

        progress.update(len(res))
        ids = np.append(ids, [item["id"] for item in res])
        embeddings = np.append(embeddings, [item["vector"] for item in res], axis=0)

    iterator.close()
    progress.close()

    print(ids.shape, embeddings.shape)

    np.savez(directory / "embeddings.npz", ids=ids, embeddings=embeddings)


@app.command(no_args_is_help=True)
def search(file: Path, query: str, n: int = 10):

    emb = np.array(get_embedding([query])[0])
    data = np.load(file)
    ids = data["ids"]
    embeddings = data["embeddings"]

    # check highest similarity
    sim = np.dot(embeddings, emb)
    top10 = np.argsort(sim)[::-1][:n]

    db = databases.Databases(Path("/home/diego/Downloads/DeepQuestions"), milvus=False)
    paragraphs = db.get_paragraphs(ids[top10].tolist())

    nicely_show_search(db, paragraphs, sim[top10])


def nicely_show_search(
    db: databases.Databases,
    paragraphs: list[databases.Paragraph],
    score: list[float] | None = None,
    ctx_size=600,
):
    podcasts_ids = []
    # deduplicate podcasts, keeping order
    for paragraph in paragraphs:
        if paragraph.podcast_id not in podcasts_ids:
            podcasts_ids.append(paragraph.podcast_id)

    podcasts = db.get_podcasts(podcasts_ids)
    # Reorder podcasts to match the order of their ids
    podcasts = sorted(podcasts, key=lambda p: podcasts_ids.index(p.id))

    if score is not None:
        score_by_id = {p.id: s for p, s in zip(paragraphs, score)}
    else:
        score_by_id = {}

    for podcast in podcasts:
        highlighted_paragraphs_in_podcast = [p for p in paragraphs if p.podcast_id == podcast.id]
        all_paragraphs = db.get_paragraph_in_podcast(podcast.id)

        # Get context for each paragraph (at least ctx_size characters)
        paragraphs_indices_to_show = set()
        for paragraph in highlighted_paragraphs_in_podcast:
            paragraphs_indices_to_show.add(paragraph.paragraph_order)

            # Add previous paragraphs until we reach ctx_size characters
            ctx = 0
            for p in reversed(all_paragraphs[: paragraph.paragraph_order]):
                paragraphs_indices_to_show.add(p.paragraph_order)
                ctx += len(p.text)
                if ctx > ctx_size:
                    break

            # Add next paragraphs until we reach ctx_size characters
            ctx = 0
            for p in all_paragraphs[paragraph.paragraph_order + 1 :]:
                paragraphs_indices_to_show.add(p.paragraph_order)
                ctx += len(p.text)
                if ctx > ctx_size:
                    break

        paragraphs_to_show = db.get_paragraph_in_podcast(
            podcast.id, list(paragraphs_indices_to_show)
        )
        paragraphs_to_show = sorted(paragraphs_to_show, key=lambda p: p.paragraph_order)

        rprint(f"[purple]{podcast.title}")
        last_paragraph_idx = None
        for paragraph in paragraphs_to_show:
            if (
                last_paragraph_idx is not None
                and paragraph.paragraph_order != last_paragraph_idx + 1
            ):
                rprint("...")
            last_paragraph_idx = paragraph.paragraph_order

            if p_score := score_by_id.get(paragraph.id):
                rprint(f"[blue]({p_score:.2f})[/blue] [green]{paragraph.text}")
            else:
                rprint(paragraph.text)
        rprint("")


@contextlib.contextmanager
def timeit():
    start = time.time()
    yield
    print(f"Time: {time.time() - start:.2f}")


if __name__ == "__main__":
    app()
