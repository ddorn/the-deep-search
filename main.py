#!/usr/bin/env -S uv run --frozen

import itertools
import json
import os
import shlex
import shutil
import sqlite3
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
from pymilvus import MilvusClient
import openai
import hashlib


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

EMBDEDDING_DIMENSIONS = 1536

def get_embedding(text: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        dimensions=EMBDEDDING_DIMENSIONS,
        model="text-embedding-3-small",
        input=text,
    )

    out: list[list[float]] = [None] * len(text)
    for embedding in response.data:
        out[embedding.index] = embedding.embedding

    return out



def txt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def mk_milvus(directory: Path):

    db_file = directory / "milvus.db"
    client = MilvusClient(str(db_file))
    collection_name = "podcasts"
    # The primary key is the hash of the text
    client.create_collection(collection_name,
                            dimension=EMBDEDDING_DIMENSIONS,
                            id_type="string",
                            max_length=len(txt_hash(""))
    )

    return client, collection_name


def mk_sql(directory: Path):
    db_file = directory / "transcripts.db"
    conn = sqlite3.connect(str(db_file))
    c = conn.cursor()
    """
    Shema:

    Table: paragraphs
        - links to the podcast in which it is
        - id
        - hash (sha256 of the text)
        - text
        - order in the podcast

    Table: podcasts
        - id
        - title
        - filename
    """

    c.execute("CREATE TABLE IF NOT EXISTS podcasts (id INTEGER PRIMARY KEY, title TEXT, filename TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS paragraphs (id INTEGER PRIMARY KEY, podcast_id INTEGER, hash TEXT, text TEXT, paragraph_order INTEGER)")

    return conn


@app.command(no_args_is_help=True)
def to_sql(directory: Path):
    """
    Add all transcriptions in a directory to a SQLite database.
    """

    cursor = mk_sql(directory)

    for file in tqdm(list(directory.glob("*.json"))):
        with open(file) as f:
            transcript = json.load(f)

        # Add the podcast to the sql database, if it doesn't exist
        res = cursor.execute("SELECT id FROM podcasts WHERE filename = ?", (file.stem,))
        podcast_id = res.fetchone()
        if podcast_id is None:
            metadata = gather_metadata(file.with_suffix(".mp3"))
            res = cursor.execute("INSERT INTO podcasts (title, filename) VALUES (?, ?)", (metadata.title, file.stem))
            podcast_id = res.lastrowid
        else:
            continue

        paragraphs = transcript["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]
        paragraph_texts = [" ".join([sentence["text"] for sentence in paragraph["sentences"]]) for paragraph in paragraphs]

        cursor.executemany("INSERT INTO paragraphs (podcast_id, hash, text, paragraph_order) VALUES (?, ?, ?, ?)",
            [(podcast_id, txt_hash(p), p, i) for i, p in enumerate(paragraph_texts)]
        )

    cursor.commit()


class Paragraph(BaseModel):
    podcast_id: int
    hash: str
    text: str
    paragraph_order: int


class PodcastSQL(BaseModel):
    id: int
    title: str
    filename: str


def get_all_ids(milvus, collection_name) -> set[str]:
    in_milvus = set()
    iterator = milvus.query_iterator(collection_name, output_fields=["id"])
    while True:
        res = iterator.next()
        if len(res) == 0:
            break

        in_milvus.update(item["id"] for item in res)

    iterator.close()
    return in_milvus


@app.command(no_args_is_help=True)
def to_milvus(directory: Path):
    """
    store the embeddings of all transcriptions from the sql database to milvus.
    """

    milvus, collection_name = mk_milvus(directory)
    cursor = mk_sql(directory)

    rprint("Connection to Milvus and SQL established :tada:")

    in_milvus = get_all_ids(milvus, collection_name)
    in_sql = set(data[0] for data in cursor.execute("SELECT hash FROM paragraphs").fetchall())  # fetchall returns (hash,)

    to_insert = in_sql - in_milvus
    rprint(f"SQL: {len(in_sql)} Milvus: {len(in_milvus)} To insert: {len(to_insert)}")

    if len(to_insert) == 0:
        rprint("[green]Milvus is up to date!")
        return

    def process_one(batch: list[int]):
        paragraphs = cursor.execute("SELECT hash, text FROM paragraphs WHERE hash IN (%s)" % ",".join("?" * len(batch)), batch).fetchall()
        paragraphs = {hash_: text for hash_, text in paragraphs}

        n_words = sum(len(p.split()) for p in paragraphs.values())
        runner.progress.log(f"Embedding {len(paragraphs)} paragraphs ({n_words} words)")
        embeddings = get_embedding(list(paragraphs.values()))

        data_to_insert = [
            {
                "id": id,
                "vector": embedding,
            }
            for id, embedding in zip(paragraphs.keys(), embeddings)
        ]
        milvus.insert(collection_name, data_to_insert)

    runner = MyParallel(progress_per_task=False, show_threads=False)
    for batch in itertools.batched(to_insert, 1000):
        runner.add(process_one)(batch)

    runner.run()
    cursor.close()





@app.command(no_args_is_help=True)
def search(directory: Path, query: str):
    """
    Search for a query in a directory of transcriptions.
    """

    milvus, collection_name = mk_milvus(directory)
    cursor = mk_sql(directory)

    embedding = get_embedding([query])[0]

    response = milvus.search(collection_name, [embedding], top_k=10, anns_field="vector")

    rprint("Results:")

    # Retrieve all info of the paragraphs into Paragraph objects
    paragraph_hashs = [item['id'] for item in response[0]]
    paragraphs = cursor.execute("SELECT * FROM paragraphs WHERE hash IN (%s)" % ",".join("?" * len(paragraph_hashs)), paragraph_hashs).fetchall()
    paragraphs = [Paragraph(podcast_id=p[1], hash=p[2], text=p[3], paragraph_order=p[4]) for p in paragraphs]

    # Retrieve the podcast titles
    podcast_ids = {p.podcast_id for p in paragraphs}
    podcasts = cursor.execute("SELECT * FROM podcasts WHERE id IN (%s)" % ",".join("?" * len(podcast_ids)), tuple(podcast_ids)).fetchall()
    podcasts = {p[0]: PodcastSQL(id=p[0], title=p[1], filename=p[2]) for p in podcasts}

    # Group the paragraphs by podcast, using itertool's groupby
    paragraphs = sorted(paragraphs, key=lambda p: p.podcast_id)
    paragraphs = itertools.groupby(paragraphs, key=lambda p: p.podcast_id)
    paragraphs = {podcast_id: list(group) for podcast_id, group in paragraphs}

    for podcast_id, podcast_paragraphs in paragraphs.items():
        rprint(f"[green]{podcasts[podcast_id].title}[/green]")
        for paragraph in podcast_paragraphs:
            rprint(paragraph.text)
        rprint("")


if __name__ == "__main__":
    app()
