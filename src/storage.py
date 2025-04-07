import json
import shutil
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from config import Config
from constants import DIRS
from core_types import (
    Asset,
    AssetType,
    Chunk,
    Document,
    PartialAsset,
    PartialChunk,
    PartialDocument,
    PartialTask,
    Task,
    TaskStatus,
)
from logs import logger


class Database:
    def __init__(self, path: str | Path, config: Config):
        self.path = path
        self.config = config

        logger.info(f"Using database at {self.path}")
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.cursor = self.db.cursor()
        self.migrate()

    def _make_ordered_list_from_results[T](
        self,
        results: list[dict],
        ids: list[int],
        type_: type[T],
    ) -> list[T]:
        if len(results) != len(ids):
            missing_ids = set(ids) - {row["id"] for row in results}
            raise ValueError(f"Missing {type_.__name__} for ids: {missing_ids}")

        object_dict = {row["id"]: type_(**row) for row in results}
        return [object_dict[object_id] for object_id in ids]

    # -- Tasks --

    def create_task(self, task: PartialTask, commit=True):
        cur = self.cursor.execute(
            "INSERT INTO tasks (strategy, document_id, input_asset_id, status) VALUES (?, ?, ?, ?)",
            (
                task.strategy,
                task.document_id,
                task.input_asset_id,
                task.status,
            ),
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
            "INSERT INTO documents (source_urn, source_id, title) VALUES (?, ?, ?)",
            (doc.source_urn, doc.source_id, doc.title),
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
        logger.debug(f"Deleting documents with ids: {document_ids}")

        self._delete_assets_and_what_they_point_to(
            "document_id IN (%s)" % ",".join("?" * len(document_ids)),
            document_ids,
        )

        self.cursor.execute(
            "DELETE FROM documents WHERE id IN (%s)" % ",".join("?" * len(document_ids)),
            document_ids,
        )
        # FIXME: This should not be necessary, as the ON DELETE CASCADE should take care of it.
        # But it doesn't and I don't know why. It does work for assets and chunks though.
        self.cursor.execute(
            "DELETE FROM tasks WHERE document_id IN (%s)" % ",".join("?" * len(document_ids)),
            document_ids,
        )

        self.db.commit()

    # -- Assets --

    def create_asset(self, asset: PartialAsset, commit=True) -> int:
        cur = self.cursor.execute(
            "INSERT INTO assets (document_id, created_by_task_id, next_steps_created, type, content, path) VALUES (?, ?, ?, ?, ?, ?)",
            (
                asset.document_id,
                asset.created_by_task_id,
                asset.next_steps_created,
                asset.type,
                asset.content,
                str(asset.path) if asset.path else None,
            ),
        )

        if commit:
            self.db.commit()

        return cur.lastrowid

    def get_assets(self, asset_ids: list[int]) -> list[Asset]:
        rows = self.cursor.execute(
            "SELECT * FROM assets WHERE id IN (%s)" % ",".join("?" * len(asset_ids)),
            asset_ids,
        ).fetchall()

        return self._make_ordered_list_from_results(rows, asset_ids, Asset)

    def get_assets_for_document(self, document_id: int, type_: str | None = None) -> list[Asset]:
        if type_ is None:
            where = "document_id = ?"
            args = (document_id,)
        else:
            where = "document_id = ? AND type = ?"
            args = (document_id, type_)

        rows = self.cursor.execute(
            f"SELECT * FROM assets WHERE {where}",
            args,
        ).fetchall()
        return [Asset(**row) for row in rows]

    def get_unhandled_assets(self) -> list[Asset]:
        rows = self.cursor.execute(
            "SELECT * FROM assets WHERE next_steps_created = FALSE",
        ).fetchall()
        return [Asset(**row) for row in rows]

    def set_asset_next_steps_created(self, asset_id: int):
        self.cursor.execute(
            "UPDATE assets SET next_steps_created = TRUE WHERE id = ?",
            (asset_id,),
        )
        self.db.commit()

    def _delete_assets_and_what_they_point_to(self, where: str, args: list):
        """
        Delete assets from the database and the files/embeddings/chunks they point to.

        Note that this does not do anything about the tasks that point to those assets
        as inputs. This always needs to be taken care of by the caller.
        """
        # We want to be slightly careful here, to live an OK state if the task fails in the middle.

        # 1. Start with the embedding assets
        embedding_assets = self.cursor.execute(
            f"SELECT content FROM assets WHERE type = ? AND {where}",
            [AssetType.EMBEDDING_ID] + args,
        ).fetchall()

        if embedding_assets:
            embedding_ids = [int(asset["content"]) for asset in embedding_assets]
            self.delete_embeddings(embedding_ids)

        # 2. Delete all the assets that are files
        file_assets = self.cursor.execute(
            f"SELECT path FROM assets WHERE path IS NOT NULL AND {where}",
            args,
        ).fetchall()

        # Delete the files associated with the assets
        for asset in file_assets:
            path = Path(asset["path"])
            if path and path.exists():
                # Only if the file was produced by us: we DON'T want to delete user files.
                if DIRS.user_data_path in path.resolve().parents:
                    logger.debug(f"Deleting asset file: '{path}'")
                    path.unlink()

        # 3. Delete the assets that are chunks
        chunks = self.cursor.execute(
            f"SELECT content FROM assets WHERE type = ? AND {where}",
            [AssetType.CHUNK_ID] + args,
        ).fetchall()
        if chunks:
            chunk_ids = [int(asset["content"]) for asset in chunks]
            self.cursor.execute(
                "DELETE FROM chunks WHERE id IN (%s)" % ",".join("?" * len(chunk_ids)),
                chunk_ids,
            )

        # 4. Delete the assets in the db
        self.cursor.execute(
            f"DELETE FROM assets WHERE {where}",
            args,
        )
        self.db.commit()

    # -- Chunks --

    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]:
        chunks = self.cursor.execute(
            "SELECT * FROM chunks WHERE id IN (%s)" % ",".join("?" * len(chunk_ids)),
            chunk_ids,
        ).fetchall()

        return self._make_ordered_list_from_results(chunks, chunk_ids, Chunk)

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
            embeddings = np.zeros(
                (0, self.config.global_config.embedding_dimension), dtype=np.float32
            )
            chunk_to_idx = {}

        # Saving ints as keys converts them to strings apparently -- we hide this.
        chunk_to_idx = {int(k): v for k, v in chunk_to_idx.items()}

        assert embeddings.shape[0] == len(
            chunk_to_idx
        ), "Embeddings and chunk_to_idx length mismatch"
        return embeddings, chunk_to_idx

    def overwrite_embeddings_files(self, embeddings: np.ndarray, chunk_to_idx: dict[int, int]):
        assert len(embeddings) == len(chunk_to_idx), "Embeddings and chunk_to_idx length mismatch"
        np.save(DIRS.user_data_path / "embeddings.npy", embeddings)
        (DIRS.user_data_path / "embeddings.json").write_text(json.dumps(chunk_to_idx))

    def update_embeddings(self, chunk_ids: list[int], embeddings: np.ndarray):
        current_embeddings, chunk_to_idx = self.load_embeddings()

        # Make space for new chunks
        new_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunk_to_idx]
        new_space = np.empty(
            (len(new_chunks), self.config.global_config.embedding_dimension), dtype=np.float32
        )
        current_embeddings = np.concatenate((current_embeddings, new_space), axis=0)
        chunk_to_idx.update(
            {chunk_id: i for i, chunk_id in enumerate(chunk_ids, start=len(chunk_to_idx))}
        )

        # Update existing ones
        embedding_indices = [
            chunk_to_idx[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_to_idx
        ]
        current_embeddings[embedding_indices] = embeddings

        self.overwrite_embeddings_files(current_embeddings, chunk_to_idx)

    def delete_embeddings(self, chunk_ids: list[int]):
        embeddings, chunk_to_idx = self.load_embeddings()

        # Collect the remaining chunks
        to_delete = set(chunk_ids)
        remaining_chunks = [chunk_id for chunk_id in chunk_to_idx if chunk_id not in to_delete]
        corresponding_rows = [chunk_to_idx[chunk_id] for chunk_id in remaining_chunks]

        embeddings = embeddings[corresponding_rows]
        chunk_to_idx = {chunk_id: i for i, chunk_id in enumerate(remaining_chunks)}

        self.overwrite_embeddings_files(embeddings, chunk_to_idx)

    # -- Strategies --

    def rerun_strategy(self, strategy: str):
        # We need to:
        # - reset to PENDING all tasks from this strategy
        # - Delete all tasks that come after
        # - Delete all assets made by this strategy and further tasks
        tasks_to_rerun = self.cursor.execute(
            "SELECT id, strategy FROM tasks WHERE strategy = ?", (strategy,)
        ).fetchall()

        assets_created = []
        tasks_to_delete = []
        last_tasks = tasks_to_rerun
        while last_tasks:
            new_assets = self.cursor.execute(
                "SELECT * FROM assets WHERE created_by_task_id IN (%s)"
                % ",".join("?" * len(last_tasks)),
                [task["id"] for task in last_tasks],
            ).fetchall()
            assets_created.extend(new_assets)

            last_tasks = self.cursor.execute(
                "SELECT id, strategy FROM tasks WHERE input_asset_id IN (%s)"
                % ",".join("?" * len(new_assets)),
                [asset["id"] for asset in new_assets],
            ).fetchall()
            tasks_to_delete.extend(last_tasks)

        # Print a nice summary: nb of task per strategy, nb of assets per type, nb of tasks reset
        tasks_counter = Counter(task["strategy"] for task in tasks_to_delete)
        assets_counter = Counter(asset["type"] for asset in assets_created)
        logger.info(f"Rerunning strategy '{strategy}'")
        logger.info(f"  - {len(tasks_to_rerun)} tasks to reset")
        for strat, count in tasks_counter.items():
            logger.info(f"  - {count} tasks to delete from strategy '{strat}'")
        for asset_type, count in assets_counter.items():
            logger.info(f"  - {count} assets of type '{asset_type}' created")

        if input("Are you sure you want to continue? (y/n)") != "y":
            logger.info("Aborting...")
            return

        # We delete assets that were created by any of the tasks
        self._delete_assets_and_what_they_point_to(
            "id IN (%s)" % ",".join("?" * len(assets_created)),
            [asset["id"] for asset in assets_created],
        )
        self.cursor.execute(
            "DELETE FROM tasks WHERE id IN (%s)" % ",".join("?" * len(tasks_to_delete)),
            [task["id"] for task in tasks_to_delete],
        )
        self.db.commit()
        self.cursor.execute(
            "UPDATE tasks SET status = ? WHERE strategy = ?", [TaskStatus.PENDING, strategy]
        )
        self.db.commit()

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
            logger.debug(f"No migrations to run - current version is {self.version}")
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
            title TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(source_urn, source_id)
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),

            document_id INTEGER,
            strategy TEXT NOT NULL,
            status TEXT NOT NULL,
            input_asset_id INTEGER,

            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY(input_asset_id) REFERENCES assets(id) ON DELETE CASCADE
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),

            document_id INTEGER,
            created_by_task_id INTEGER,
            next_steps_created BOOLEAN,

            type TEXT NOT NULL,
            content TEXT,
            path TEXT,

            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY(created_by_task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            document_order INTEGER,
            content TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now', 'utc')),
            UNIQUE(document_id, document_order),

            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
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
