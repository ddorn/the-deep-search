import logging
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from constants import DIRS

LOGGER_NAME = 'thedeepsearch'

def setup_logging() -> logging.Logger:
    # Create a logger object
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # Captures all levels of log messages

    # Set up a file handler with detailed debug level messages
    file_handler = RotatingFileHandler(DIRS.user_log_path / 'main.log', maxBytes=20*1024*1024, backupCount=20)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Set up Rich console handler
    rich_handler = RichHandler(
        level=logging.DEBUG,  # Setting this to INFO to reduce verbosity in the console
        show_time=True,
        enable_link_path=True,
        rich_tracebacks=True,
        omit_repeated_times=False,
    )

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)

    return logger


logger = setup_logging()


if __name__ == '__main__':

    # Example usage of the logger
    logger.debug("This is a debug message, it's very detailed and saved in the file.")
    logger.info("This is an info message, less verbose, shown on both file and console with rich formatting.")
    logger.warning("This warning message appears both in the log file and on console, formatted by Rich.")
    logger.error("Error message logged to both file and console, richly formatted.")
    logger.critical("Critical issue logged to both file and console, with rich formatting.")

    try:
        1/0
    except ZeroDivisionError:
        # Show exception traceback in the logs
        logger.exception("Exception occurred: Division by zero!")
