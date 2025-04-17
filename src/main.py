# %%
import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from typer import Typer

from constants import DIRS
from executor import Executor
from logs import logger, setup_logging
from storage import setup_db

load_dotenv(override=True)

app = Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(
    config: Path = None,
    fresh: bool = False,
    no_sync: bool = False,
    log_level: int = logging.DEBUG,
    on_server: bool = False,
):
    setup_logging(log_level, is_server=on_server)

    if fresh:
        delete_all_data()

    if no_sync:
        os.environ["DS_NO_SYNC"] = "1"

    db = setup_db(extra_path_for_config=config)
    asyncio.run(Executor(db).main())


@app.command()
def rerun_strategy(
    strategy: str,
    doc_ids: Annotated[
        str,
        typer.Option(
            help="Document IDs to rerun the strategy on. Default is all. Space separated."
        ),
    ] = "",
    config: Path = None,
):
    """Rerun the specified strategy."""

    db = setup_db(extra_path_for_config=config)

    strategies = set(
        task["strategy"]
        for task in db.cursor.execute("SELECT DISTINCT strategy FROM tasks").fetchall()
    )
    if strategy not in strategies:
        logger.info(f"Available strategies: {strategies}")
        logger.error(f"Strategy '{strategy}' not found in the database.")
        return

    doc_ids = [int(id) for id in doc_ids.split()]
    db.rerun_strategy(strategy, doc_ids)
    db.commit()


@app.command()
def reprocess_doc(doc_id: int):
    db = setup_db()
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
