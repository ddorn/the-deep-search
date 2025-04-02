# %%
from pathlib import Path
from typing import Annotated
from yaml import safe_load
from pydantic import BaseModel, Field


example_config = """
sources:
  Deep Questions Podcast:
    type: rss-podcast
    args:
        url: https://feeds.simplecast.com/_IjaDYA

  My Notes:
    type: local-files
    args:
        root: ~/notes
"""


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
