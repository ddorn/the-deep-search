import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from constants import DIRS

LOGGER_NAME = "thedeepsearch"


def setup_logging(log_level=logging.DEBUG, is_server: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    logger.setLevel(log_level)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    rich_handler = RichHandler(
        show_time=True,
        enable_link_path=True,
        rich_tracebacks=True,
    )
    logger.addHandler(rich_handler)

    if not is_server:
        # TODO: Re-discuss if this should even exist.
        # TODO: This should ideally use storage_path.
        file_handler = RotatingFileHandler(
            DIRS.user_log_path / "main.log", maxBytes=20 * 1024 * 1024, backupCount=20
        )
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging(log_level=logging.DEBUG, is_server=True)

if __name__ == "__main__":
    # Example usage of the logger
    logger.debug("This is a debug message, it's very detailed and saved in the file.")
    logger.info(
        "This is an info message, less verbose, shown on both file and console with rich formatting."
    )
    logger.warning(
        "This warning message appears both in the log file and on console, formatted by Rich."
    )
    logger.error("Error message logged to both file and console, richly formatted.")
    logger.critical("Critical issue logged to both file and console, with rich formatting.")

    try:
        1 / 0
    except ZeroDivisionError:
        # Show exception traceback in the logs
        logger.exception("Exception occurred: Division by zero!")
