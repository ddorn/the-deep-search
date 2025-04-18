# %%
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from yaml import safe_load


class SourceConfig(BaseModel):
    type: str
    args: dict


class Config(BaseModel):
    sources: Annotated[dict[str, SourceConfig], Field(default_factory=dict)]
    strategies: Annotated[dict[str, dict], Field(default_factory=dict)]

    embedding_dimension: int = 1536
    paused_strategies: list[str] = []

    storage_path: Path | None = None


def load_config(path: Path) -> Config:
    with open(path, "r") as f:
        return Config.model_validate(safe_load(f))
