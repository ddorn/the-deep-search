import asyncio

from constants import DIRS
from sources.directory_source import DirectorySource
from config import SourceConfig
from pathlib import Path
import shutil

from typer import Typer

from executor import Executor

app = Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main():
    asyncio.run(Executor().main())


@app.command()
def seed(p: Path):
    ds = DirectorySource(
        "Test",
        SourceConfig(type="local-files", args={"path": p}),
    )

    ds.add_tasks_from_changes()


@app.command()
def delete_all_data():
    shutil.rmtree(DIRS.user_data_path)
    shutil.rmtree(DIRS.user_cache_path)

if __name__ == "__main__":
    app()
