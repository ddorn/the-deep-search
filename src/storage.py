import os
import appdirs
import sqlite3
import shutil


def logger():
    pass


class Store:
    def __init__(self, root):
        self.root = root

    @classmethod
    def with_name(cls, name):
        config_dir = appdirs.user_config_dir("deepsearch")
        path = os.path.join(config_dir, name)
        os.makedirs(path, exist_ok=True)
        return cls(path)

    def open(self, filename, mode, *args, **kwargs):
        return open(self.path_for(filename), mode, *args, **kwargs)

    def path_for(self, filename):
        return os.path.join(self.root, filename)


class Database:
    def __init__(self, path):
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        self.migrate()

    @property
    def version(self):
        return self.cursor.execute("PRAGMA user_version").fetchone()[0]

    def set_version(self, value: str):
        return self.cursor.execute(f"PRAGMA user_version = {value}")

    def migrate(self):
        migrations = [
            self.migrate_0_to_1,
        ]

        if self.version == len(self.MIGRATIONS):
            logger.info(f"No migrations to run - current version is {self.version}")
            return

        if self.version > len(migrations):
            raise ValueError(
                f"Database version {self.version} is higher than the latest migration {len(migrations) + 1}"
            )
            return

        backup_path = CONFIG_DIR.path_for(f"database-backup-{self.version}.sqlite")
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
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER primary key autoincrement
            urn STRING not null
            source STRING not null
            created_at TIMESTAMP not null default now()
            UNIQUE(urn)
        )""")

        self.cursor.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER primary key autoincrement
            document_id INTEGER references documents(id) ON DELETE CASCADE
            kind STRING
            status STRING not null
            args BLOB
        )""")

        self.cursor.execute("""CREATE TABLE IF NOT EXISTS byproducts (
                id integer primary key autoincrement
                document_id integer references documents(id) ON DELETE CASCADE
                path string
                created_at timestamp not null default now()
                UNIQUE(document_id, path)
        )""")


CONFIG_DIR = Store.with_name("")
DATABASE = Database(CONFIG_DIR.path_for("database.sqlite"))
