import hashlib
import sqlite3
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from pymilvus import MilvusClient

EMBDEDDING_DIMENSIONS = 1536


def txt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Paragraph(BaseModel):
    """Model of the paragraph table in SQL"""

    id: int = None
    podcast_id: int
    hash: str
    text: str
    paragraph_order: int

    @classmethod
    def from_text(cls, podcast_id: int, text: str, paragraph_order: int):
        return cls(
            podcast_id=podcast_id,
            hash=txt_hash(text),
            text=text,
            paragraph_order=paragraph_order,
        )


class Podcast(BaseModel):
    """Model of the podcast table in SQL"""

    id: int
    title: str
    filename: str


class Databases:
    def __init__(self, directory: Path, milvus: bool = True):
        self.directory = directory
        if milvus:
            self.milvus, self.collection_name = self.mk_milvus(directory)
        else:
            self.milvus = None
            self.collection_name = None
        self.sql = self.mk_sql(directory)

    def __del__(self):
        try:
            self.sql.close()
        except AttributeError:
            pass

    @staticmethod
    def mk_milvus(directory: Path):
        db_file = directory / "milvus.db"
        client = MilvusClient(str(db_file))
        collection_name = "podcasts"
        # The primary key is the hash of the text
        client.create_collection(
            collection_name,
            dimension=EMBDEDDING_DIMENSIONS,
            id_type="string",
            max_length=len(txt_hash("")),
        )

        return client, collection_name

    @staticmethod
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

        c.execute(
            "CREATE TABLE IF NOT EXISTS podcasts (id INTEGER PRIMARY KEY, title TEXT, filename TEXT)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS paragraphs (id INTEGER PRIMARY KEY, podcast_id INTEGER, hash TEXT, text TEXT, paragraph_order INTEGER)"
        )

        return conn

    def get_paragraphs(self, hashes: Sequence[str]) -> list[Paragraph]:
        paragraphs = self.sql.execute(
            "SELECT * FROM paragraphs WHERE hash IN (%s)" % ",".join("?" * len(hashes)),
            hashes,
        ).fetchall()
        paragraphs = [
            Paragraph(id=p[0], podcast_id=p[1], hash=p[2], text=p[3], paragraph_order=p[4])
            for p in paragraphs
        ]
        return paragraphs

    def get_paragraph_in_podcast(
        self, podcast_id: int, paragraph_order: Sequence[int] | None = None
    ) -> list[Paragraph]:
        query = "SELECT * FROM paragraphs WHERE podcast_id = ?"
        if paragraph_order is not None:
            query += " AND paragraph_order IN (%s)" % ",".join("?" * len(paragraph_order))
            paragraphs = self.sql.execute(query, [podcast_id] + list(paragraph_order)).fetchall()
        else:
            paragraphs = self.sql.execute(query, (podcast_id,)).fetchall()

        paragraphs = [
            Paragraph(id=p[0], podcast_id=p[1], hash=p[2], text=p[3], paragraph_order=p[4])
            for p in paragraphs
        ]
        return paragraphs

    def get_podcast_by_filename(self, filename: str) -> Podcast | None:
        """Filename is the name of the file without the extension"""
        return self.sql.execute("SELECT * FROM podcasts WHERE filename = ?", (filename,)).fetchone()

    def get_podcasts(self, ids: Sequence[int]) -> list[Podcast]:
        podcasts = self.sql.execute(
            "SELECT * FROM podcasts WHERE id IN (%s)" % ",".join("?" * len(ids)), ids
        ).fetchall()
        podcasts = [Podcast(id=p[0], title=p[1], filename=p[2]) for p in podcasts]
        return podcasts

    def search(self, embedding: list[float], n: int = 10) -> list[Paragraph]:
        response = self.milvus.search(
            self.collection_name, [embedding], top_k=10, anns_field="vector"
        )
        hashes = [r["id"] for r in response[0]]
        return self.get_paragraphs(hashes)

    def get_all_milvus_ids(self) -> set[str]:
        in_milvus: set[str] = set()
        iterator = self.milvus.query_iterator(self.collection_name, output_fields=["id"])
        while True:
            res = iterator.next()
            if len(res) == 0:
                break

            in_milvus.update(item["id"] for item in res)

        iterator.close()
        return in_milvus

    def get_all_paragraph_hashes(self) -> set[str]:
        return set(p[0] for p in self.sql.execute("SELECT hash FROM paragraphs").fetchall())

    def milvus_add(self, texts: list[str], embeddings: list[list[float]]):
        data = [
            {
                "id": txt_hash(text),
                "vector": embedding,
            }
            for text, embedding in zip(texts, embeddings, strict=True)
        ]
        self.milvus.insert(self.collection_name, data)

    def add_paragraphs_sql(self, paragraphs: list[Paragraph]):
        self.sql.executemany(
            "INSERT INTO paragraphs (podcast_id, hash, text, paragraph_order) VALUES (?, ?, ?, ?)",
            [(p.podcast_id, p.hash, p.text, p.paragraph_order) for p in paragraphs],
        )

    def add_podcast_sql(self, podcast: Podcast):
        res = self.sql.execute(
            "INSERT INTO podcasts (title, filename) VALUES (?, ?)",
            (podcast.title, podcast.filename),
        )
        return res.lastrowid
