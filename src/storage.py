from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import shutil

import numpy as np

from config import Config
from core_types import (
    Chunk,
    PartialAsset,
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
    def __init__(self, path: str | Path, config: Config):
        self.path = path
        self.config = config

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
            "INSERT INTO documents (source_urn, source_id) VALUES (?, ?)",
            (doc.source_urn, doc.source_id),
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

    def get_ids_from_urns(self, source: str, urns: list[str]) -> dict[str, int]:
        rows = self.cursor.execute(
            "SELECT id, source_urn FROM documents WHERE source_id = ? AND source_urn IN (%s)"
            % ",".join("?" * len(urns)),
            [source] + urns,
        ).fetchall()

        out = {row["source_urn"]: row["id"] for row in rows}
        return out

    def get_documents_from_source(self, source_id: str) -> list[Document]:
        rows = self.cursor.execute(
            "SELECT * FROM documents WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        return [Document(**row) for row in rows]

    def delete_documents(self, document_ids: list[int]):
        # We want to be slightly careful here, to live a correct state if the task fails in the middle.

        # 1. Delete all related embeddings
        chunks = self.cursor.execute(
            "SELECT id FROM chunks WHERE document_id IN (%s)" % ",".join("?" * len(document_ids)),
            document_ids,
        ).fetchall()
        chunk_ids = [chunk["id"] for chunk in chunks]

        self.delete_embeddings(chunk_ids)

        # 2. Delete all the assets that are files
        assets = self.cursor.execute(
            "SELECT path FROM assets WHERE path IS NOT NULL AND document_id IN (%s)"
            % ",".join("?" * len(document_ids)),
            document_ids,
        ).fetchall()

        # Delete the files associated with the assets
        for asset in assets:
            path = Path(asset["path"])
            if path and path.exists():
                # Only if the file was produced by us: we DON'T want to delete user files.
                if DIRS.user_data_path in path.resolve().parents:
                    logger.debug(f"Deleting asset file: '{path}'")
                    path.unlink()

        # 3. Delete the documents, tasks, assets, and chunks
        self.cursor.execute(
            "DELETE FROM documents WHERE id IN (%s)" % ",".join("?" * len(document_ids)),
            document_ids,
        )

        self.db.commit()

    # -- Assets --

    def create_asset(self, asset: PartialAsset, commit=True) -> int:
        cur = self.cursor.execute(
            "INSERT INTO assets (document_id, created_by_task_id, next_step_id, type, content, path) VALUES (?, ?, ?, ?, ?, ?)",
            (
                asset.document_id,
                asset.created_by_task_id,
                asset.next_step_id,
                asset.type,
                asset.content,
                str(asset.path),
            ),
        )

        if commit:
            self.db.commit()

        return cur.lastrowid

    # -- Chunks --

    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]:
        chunks = self.cursor.execute(
            "SELECT * FROM chunks WHERE id IN (%s)" % ",".join("?" * len(chunk_ids)),
            chunk_ids,
        ).fetchall()

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

    # Embeddings are stored in 2 files:
    # - embeddings.json: Keeps a mapping of chunk_id to index in the embeddings array
    # - embeddings.npy: The embedding array, shape (n_chunks, embedding_dim)

    def load_embeddings(self) -> tuple[np.ndarray, dict[int, int]]:
        try:
            embeddings = np.load(DIRS.user_data_path / "embeddings.npy")
            chunk_to_idx = json.loads((DIRS.user_data_path / "embeddings.json").read_text())
        except FileNotFoundError:
            logger.info("Embeddings files not found, creating empty embeddings.")
            embeddings = np.zeros((0, self.config.global_config.embedding_dimension), dtype=np.float32)
            chunk_to_idx = {}

        # Saving ints as keys converts them to strings apparently -- we hide this.
        chunk_to_idx = {int(k): v for k, v in chunk_to_idx.items()}

        assert embeddings.shape[0] == len(chunk_to_idx), "Embeddings and chunk_to_idx length mismatch"
        return embeddings, chunk_to_idx

    def overwrite_embeddings_files(self, embeddings: np.ndarray, chunk_to_idx: dict[int, int]):
        assert len(embeddings) == len(chunk_to_idx), "Embeddings and chunk_to_idx length mismatch"
        np.save(DIRS.user_data_path / "embeddings.npy", embeddings)
        (DIRS.user_data_path / "embeddings.json").write_text(json.dumps(chunk_to_idx))

    def update_embeddings(self, chunk_ids: list[int], embeddings: np.ndarray):
        current_embeddings, chunk_to_idx = self.load_embeddings()

        # Make space for new chunks
        new_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunk_to_idx]
        new_space = np.empty((len(new_chunks), self.config.global_config.embedding_dimension), dtype=np.float32)
        current_embeddings = np.concatenate((current_embeddings, new_space), axis=0)
        chunk_to_idx.update({chunk_id: i for i, chunk_id in enumerate(chunk_ids, start=len(chunk_to_idx))})

        # Update existing ones
        embedding_indices = [chunk_to_idx[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_to_idx]
        current_embeddings[embedding_indices] = embeddings

        self.overwrite_embeddings_files(current_embeddings, chunk_to_idx)

    def delete_embeddings(self, chunk_ids: list[int]):
        embeddings, chunk_to_idx = self.load_embeddings()

        # Remove the embeddings for the chunks (with .pop())
        embedding_indices = [chunk_to_idx.pop(chunk_id) for chunk_id in chunk_ids if chunk_id in chunk_to_idx]
        embeddings = np.delete(embeddings, embedding_indices, axis=0)

        self.overwrite_embeddings_files(embeddings, chunk_to_idx)

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
            source_urn TEXT,
            source_id TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(source_urn, source_id)
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
            """CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
            created_by_task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
            next_step_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,

            type TEXT NOT NULL,
            content TEXT,
            path TEXT
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


CURRENT_DATABASE: str = None
DATABASES: dict[str, Database] = {}


def get_db() -> Database:
    """Get the current database instance."""
    return DATABASES[CURRENT_DATABASE]


def set_db(db_name: str, db: Database):
    """Set the current database instance."""
    global CURRENT_DATABASE

    if db_name in DATABASES:
        raise ValueError(f"Database {db_name} already exists.")

    DATABASES[db_name] = db
    CURRENT_DATABASE = db_name

@contextmanager
def temporary_db(db_name: str = "test", config: Config | None = None):
    """Context manager to temporarily switch the database."""

    global CURRENT_DATABASE
    original_db = CURRENT_DATABASE
    assert db_name not in DATABASES, f"Database {db_name} already exists."
    if config is None:
        config = Config()

    try:
        DATABASES[db_name] = Database(":memory:", config)
        CURRENT_DATABASE = db_name
        yield DATABASES[db_name]
    finally:
        CURRENT_DATABASE = original_db
        DATABASES[db_name].db.close()
        del DATABASES[db_name]
