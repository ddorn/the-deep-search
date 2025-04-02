import asyncio

from constants import DIRS
from sources.directory_source import DirectorySource
from config import Config, SourceConfig, load_config
from pathlib import Path
import shutil

from typer import Typer

from logs import logger
from executor import Executor
from storage import Database, get_db, set_db

app = Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(config: Path = None):
    if config is None:
        parsed_config = Config(sources={})
    else:
        parsed_config = load_config(config)

    logger.debug(f"Config: {parsed_config}")

    set_db("default", Database(DIRS.user_data_path / "db.sqlite", config=parsed_config))

    asyncio.run(Executor().new_main())


@app.command()
def delete_all_data():
    shutil.rmtree(DIRS.user_data_path)
    shutil.rmtree(DIRS.user_cache_path)
    logger.warning(f"Deleted all data in {DIRS.user_data_path} and {DIRS.user_cache_path}")

if __name__ == "__main__":
    app()
