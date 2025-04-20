# %%
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field
from yaml import safe_load

from constants import DIRS
from logs import logger

class SourceConfig(BaseModel):
    type: str
    args: dict


class Config(BaseModel):
    sources: Annotated[dict[str, SourceConfig], Field(default_factory=dict)]
    strategies: Annotated[dict[str, dict], Field(default_factory=dict)]

    embedding_dimension: int = 1536
    paused_strategies: list[str] = []

    storage_path: Annotated[Path, AfterValidator(lambda p: p.expanduser())] = DIRS.user_data_path

    splash_text: str = "No results found"
    search_placeholder: str = "Search for..."

    # Task retry configuration
    task_retries: int = 3
    task_retry_delay: int = 60  # seconds
    crash_on_task_failure: bool = False

def load_config(path: Path) -> Config:
    with open(path, "r") as f:
        return Config.model_validate(safe_load(f))


def setup_config(extra_path_for_config: Path | None = None, overrides: list[str] = []) -> Config:
    """
    Loads the config from the given path, or from the default paths if not provided.

    Overrides are in the format "topkey.subkey=value".
    """

    paths_to_try = [
        Path("/var/lib/deepsearch/config.yaml"),
        DIRS.user_config_path / "config.yaml",
    ]

    config_path = None
    if not extra_path_for_config:
        for path in paths_to_try:
            if path.exists():
                config_path = path
                break
    elif extra_path_for_config.expanduser().exists():
        config_path = extra_path_for_config.expanduser()
    else:
        logger.critical(f"No config file found at {extra_path_for_config}")
        raise FileNotFoundError(f"No config file found at {extra_path_for_config}")

    if config_path:
        config = load_config(config_path)
        logger.info(f"Using config file: {config_path.resolve()}")
    else:
        logger.warning(f"No config file found in any of {paths_to_try}, using default config.")
        config = Config()

    # Apply overrides:
    # - dump the config to a dict
    # - apply the overrides
    # - load the config from the dict
    config_dict = config.model_dump()
    for override in overrides:
        key, _, value = override.partition("=")
        keys = key.split(".")
        current = config_dict
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                raise ValueError(f"Key {'.'.join(keys[:i+1])} is not a valid key in the config")
            current = current[k]

        if keys[-1] not in current:
            raise ValueError(f"Key {key} is not a valid key in the config")
        current[keys[-1]] = value

    config = Config.model_validate(config_dict)

    logger.info(f"Config: {config}")

    return config