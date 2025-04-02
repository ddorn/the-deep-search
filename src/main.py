import asyncio

from constants import DIRS
from sources.directory_source import DirectorySource
from config import Config, SourceConfig, load_config
from pathlib import Path
import shutil

from typer import Typer

from logs import logger
from executor import Executor

app = Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main():
    asyncio.run(Executor().main())

@app.command()
def new_main(config: Path = None):
    if config is None:
        parsed_config = Config(sources={})
    else:
        parsed_config = load_config(config)

    logger.debug(f"Config: {parsed_config}")
    asyncio.run(Executor(parsed_config).new_main())


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
    logger.warning(f"Deleted all data in {DIRS.user_data_path} and {DIRS.user_cache_path}")

if __name__ == "__main__":
    app()
