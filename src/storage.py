from contextlib import contextmanager
from pathlib import Path
import sqlite3
import shutil

import numpy as np
from pydantic import BaseModel

from config import Config, get_config
from core_types import (
    Chunk,
    PartialChunk,
    Task,
    TaskStatus,
    PartialTask,
    Document,
    PartialDocument,
)
from constants import DIRS
from logs import logger


class Database:
    def __init__(self, path):
        self.path = path
        self.config = get_config()

        logger.debug(f"Using database at {self.path}")
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.cursor = self.db.cursor()
        self.migrate()

    # -- Tasks --

    def create_task(self, task: PartialTask, commit=True):
        cur = self.cursor.execute(
            "INSERT INTO tasks (strategy, document_id, status, parent_id, args) VALUES (?, ?, ?, ?, ?)",
            (task.strategy, task.document_id, task.status, task.parent_id, task.args),
        )

        if commit:
            self.db.commit()

        return cur.lastrowid

    def get_pending_tasks(self) -> list[Task]:
        tasks = self.cursor.execute(
            "SELECT * FROM tasks WHERE status = ?",
            (TaskStatus.PENDING,),
        ).fetchall()
        return [Task(**task) for task in tasks]

    def set_task_status(self, status: TaskStatus, task_ids: list[int]):
        self.cursor.execute(
            "UPDATE tasks SET status = ? WHERE id IN (%s)" % ",".join("?" * len(task_ids)),
            [status] + task_ids,
        )
        self.db.commit()

    def restart_crashed_tasks(self):
        self.cursor.execute(
            "UPDATE tasks SET status = ? WHERE status = ?",
            (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),
        )
        self.db.commit()

    # -- Documents --

    def create_document(self, doc: PartialDocument, commit=True):
        cur = self.cursor.execute(
            "INSERT INTO documents (urn, source_id) VALUES (?, ?)",
            (doc.urn, doc.source_id),
        )

        if commit:
            self.db.commit()

        return cur.lastrowid

    def get_document_by_id(self, id: str) -> Document | None:
        row = self.cursor.execute(
            "SELECT * FROM documents WHERE id = ?",
            (id,),
        ).fetchone()

        return Document(**row) if row else None

    def get_document_id_from_urn(self, urn: str) -> int | None:
        row = self.cursor.execute(
            "SELECT id FROM documents where urn = ?",
            (urn,),
        ).fetchone()
        return row["id"] if row else None

    def get_documents_from_source(self, source_id: str) -> list[Document]:
        rows = self.cursor.execute(
            "SELECT * FROM documents WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        return [Document(**row) for row in rows]

    def delete_document(self, id: int):
        # Get all the byproducts for the document
        byproducts = self.cursor.execute(
            "SELECT * FROM byproducts WHERE document_id = ?",
            (id,),
        ).fetchall()

        # Delete the files associated with the byproducts
        for byproduct in byproducts:
            path = Path(byproduct["path"])
            if path and path.exists():
                # Assert path is in the data directory
                if DIRS.user_data_path in path.resolve().parents:
                    raise ValueError(f"Byproduct file {path} is not in the data directory.")
                logger.debug(f"Deleting byproduct file: {path}")
                path.unlink()

        self.cursor.execute(
            "DELETE FROM documents WHERE id = ?",
            (id,),
        )
        self.db.commit()

        # TODO: Also delete the chunk embeddings for the document

    # -- Chunks --

    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]:
        chunks = self.cursor.execute(
            "SELECT * FROM chunks WHERE id IN (%s)" % ",".join("?" * len(chunk_ids)),
            chunk_ids,
        ).fetchall()

        # Debug: show everything in the table
        if True:
            all_chunks = self.cursor.execute("SELECT * FROM chunks").fetchall()
            for chunk in all_chunks:
                print({**chunk})

        if len(chunks) != len(chunk_ids):
            missing_ids = set(chunk_ids) - {chunk["id"] for chunk in chunks}
            raise ValueError(
                f"Missing chunks for ids: {missing_ids}"
            )

        # Convert the rows to Chunk objects
        chunk_dict = {chunk["id"]: Chunk(**chunk) for chunk in chunks}

        # We return the chunks in the order they were requested
        return [chunk_dict[chunk_id] for chunk_id in chunk_ids]

    def create_chunks(self, chunks: list[PartialChunk], commit=True) -> list[int]:
        new_ids = []
        for chunk in chunks:
            cur = self.cursor.execute(
                "INSERT INTO chunks (document_id, document_order, content) VALUES (?, ?, ?)",
                (chunk.document_id, chunk.document_order, chunk.content),
            )
            new_ids.append(cur.lastrowid)

        if commit:
            self.db.commit()
        return new_ids

    def get_chunks_by_document_id(self, document_id: int) -> list[Chunk]:
        rows = self.cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        return [Chunk(**row) for row in rows]

    # -- Embeddings --

    def update_embeddings(self, chunk_ids: list[int], embeddings: np.ndarray):
        print(f"Updating embeddings for {len(chunk_ids)} chunks")
        print(chunk_ids, embeddings)

    # -- Config --

    def save_config(self, config: Config):
        self.cursor.execute(
            "INSERT INTO config (config) VALUES (?)",
            (config.model_dump_json(),),
        )
        self.db.commit()

    def get_last_config(self) -> Config | None:
        row = self.cursor.execute(
            "SELECT * FROM config ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return Config.model_validate_json(row["config"]) if row else None

    # -- Migrations --

    @property
    def version(self):
        return self.cursor.execute("PRAGMA user_version").fetchone()[0]

    def set_version(self, value: str):
        return self.cursor.execute(f"PRAGMA user_version = {value}")

    def migrate(self):
        migrations = [
            self.migrate_0_to_1,
        ]

        if self.version == len(migrations):
            logger.info(f"No migrations to run - current version is {self.version}")
            return

        if self.version > len(migrations):
            raise ValueError(
                f"Database version {self.version} is higher than the latest migration {len(migrations) + 1}"
            )

        if self.path != ":memory:":
            backup_path = DIRS.user_data_path / f"database-backup-{self.version}.sqlite"
            logger.info(f"Backing up the database to {backup_path} before migration.")
            shutil.copy(self.path, backup_path)

        # Run the migrations
        to_run = migrations[self.version :]
        for migration in to_run:
            logger.info(f"Running migration {migration.__name__}")
            migration()
            self.set_version(self.version + 1)
            self.db.commit()

        logger.info(f"Database migrated to version {self.version}")

    def migrate_0_to_1(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            urn TEXT,
            source_id TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(urn)
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            strategy TEXT,
            status TEXT NOT NULL,
            args BLOB,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            parent_id INTEGER REFERENCES tasks(id)
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS byproducts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
            path TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(document_id, path)
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
            document_order INTEGER,
            content TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(document_id, document_order)
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc'))
        )"""
        )


CURRENT_DATABASE: str = "default"
DATABASES: dict[str, Database] = {
    CURRENT_DATABASE: Database(DIRS.user_data_path / "database.sqlite")
}


def get_db() -> Database:
    """Get the current database instance."""
    return DATABASES[CURRENT_DATABASE]


@contextmanager
def temporary_db(db_name: str = "test"):
    """Context manager to temporarily switch the database."""

    global CURRENT_DATABASE
    original_db = CURRENT_DATABASE
    assert db_name not in DATABASES, f"Database {db_name} already exists."

    try:
        DATABASES[db_name] = Database(":memory:")
        CURRENT_DATABASE = db_name
        yield DATABASES[db_name]
    finally:
        CURRENT_DATABASE = original_db
        DATABASES[db_name].db.close()
        del DATABASES[db_name]
