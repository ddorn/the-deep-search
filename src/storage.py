import sqlite3
import shutil

from tasks import Task, TaskStatus, PartialTask
from constants import DIRS
from logs import logger


class Database:
    def __init__(self, path):
        self.path = path
        logger.debug(f"Using database at {self.path}")
        self.db = sqlite3.connect(self.path)
        self.cursor = self.db.cursor()
        self.migrate()

    def create_task(self, task: PartialTask, commit=True):
        cur = self.cursor.execute(
            "INSERT INTO tasks (strategy, document_id, status, args) VALUES (?, ?, ?, ?)",
            (task.strategy, task.document_id, task.status, task.args),
        )

        if commit:
            self.db.commit()

        return cur.lastrowid

    def create_dependent_tasks(self, *tasks: PartialTask):
        """Create tasks that will be executed in the order passed. The first will be pending, the rest blocked."""
        child = None
        for task in reversed(tasks):
            assert task.child_id is None
            task.child_id = child
            if child is None:
                task.status = TaskStatus.PENDING
            else:
                task.status = TaskStatus.BLOCKED
            child = self.create_task(task, commit=False)

        self.db.commit()

    def get_pending_tasks(self) -> list[Task]:
        tasks = self.cursor.execute(
            "SELECT id, document_id, strategy, status, args, created_at, child_id FROM tasks WHERE status = ?",
            (TaskStatus.PENDING,),
        ).fetchall()
        return [
            Task(
                id=task[0],
                document_id=task[1],
                strategy=task[2],
                status=task[3],
                args=task[4],
                created_at=task[5],
                child_id=task[6],
            )
            for task in tasks
        ]

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
            """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            urn TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(urn)
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id STRING REFERENCES documents(urn) ON DELETE CASCADE,
            strategy TEXT,
            status TEXT NOT NULL,
            args BLOB,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            child_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS byproducts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT REFERENCES documents(urn) ON DELETE CASCADE,
            path TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(document_id, path)
        )"""
        )


DATABASE = Database(DIRS.user_data_path / "database.sqlite")
