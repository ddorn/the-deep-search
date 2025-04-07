# %%
import asyncio
import shutil
from pathlib import Path

from typer import Typer

from config import Config, load_config
from constants import DIRS
from executor import Executor
from logs import logger
from storage import Database, set_db

app = Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(config: Path = None, fresh: bool = False, no_sync: bool = False):
    if fresh:
        delete_all_data()

    if config is None:
        parsed_config = Config(sources={})
    else:
        parsed_config = load_config(config)

    logger.debug(f"Config: {parsed_config}")

    set_db("default", Database(DIRS.user_data_path / "db.sqlite", config=parsed_config))

    asyncio.run(Executor().main())


@app.command()
def rerun_strategy(strategy: str, config: Path = None):
    """Rerun the specified strategy."""
    if config is None:
        parsed_config = Config(sources={})
    else:
        parsed_config = load_config(config)

    db = Database(DIRS.user_data_path / "db.sqlite", config=parsed_config)

    strategies = set(
        task["strategy"]
        for task in db.cursor.execute("SELECT DISTINCT strategy FROM tasks").fetchall()
    )
    if strategy not in strategies:
        logger.info(f"Available strategies: {strategies}")
        logger.error(f"Strategy '{strategy}' not found in the database.")
        return

    db.rerun_strategy(strategy)


@app.command()
def test(doc_id: int):
    db = Database(DIRS.user_data_path / "db.sqlite", config=Config(sources={}))

    db.delete_documents([doc_id])


@app.command()
def delete_all_data():
    shutil.rmtree(DIRS.user_data_path)
    shutil.rmtree(DIRS.user_cache_path)
    logger.warning(f"Deleted all data in {DIRS.user_data_path} and {DIRS.user_cache_path}")


@app.command()
def drive():
    from sources.gdrive_source import list_all_gdocs_fast

    # Replace with your folder ID
    folder_id = "1aOP3HEOtSCrNDKyTV4nXaNTf5Zcql3ff"
    service_account_file = "/home/diego/prog/ML4G2.0/meta/service_account_token.json"
    # Replace with your folder ID
    list_all_gdocs_fast(folder_id, service_account_file)


# %%

if __name__ == "__main__":
    app()
