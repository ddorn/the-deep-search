import sqlite3
import shutil

import numpy as np

from core_types import Chunk, Task, TaskStatus, PartialTask
from constants import DIRS
from logs import logger


class Database:
    def __init__(self, path):
        self.path = path
        logger.debug(f"Using database at {self.path}")
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.cursor = self.db.cursor()
        self.migrate()

    def create_task(self, task: PartialTask, commit: bool = True):
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

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        chunks = self.cursor.execute(
            "SELECT * FROM byproducts WHERE id IN (%s)"
            % ",".join("?" * len(chunk_ids)),
            chunk_ids,
        ).fetchall()
        return [Chunk(**chunk) for chunk in chunks]

    def update_embeddings(self, chunk_ids: list[str], embeddings: np.ndarray):
        print(f"Updating embeddings for {len(chunk_ids)} chunks")
        print(chunk_ids, embeddings)


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
            return

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
            """CREATE TABLE IF NOT EXISTS source (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                attributes BLOB ,
                UNIQUE(type, name),
                UNIQUE(name)
            )"""
        )
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            urn TEXT
            source_id REFERENCES sources(id) ON DELETE CASCADE,
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


DATABASE = Database(DIRS.user_data_path / "database.sqlite")
