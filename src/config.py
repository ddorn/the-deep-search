# %%
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from yaml import safe_load


class GlobalConfig(BaseModel):
    embedding_dimension: int = 1536


class SourceConfig(BaseModel):
    type: str
    args: dict


class Config(BaseModel):
    sources: Annotated[dict[str, SourceConfig], Field(default_factory=dict)]
    strategies: Annotated[dict[str, dict], Field(default_factory=dict)]
    global_config: Annotated[GlobalConfig, Field(default_factory=GlobalConfig)]


def load_config(path: Path) -> Config:
    with open(path, "r") as f:
        return Config.model_validate(safe_load(f))
