# %%
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


class SourceConfig(BaseModel):
    type: str
    args: dict

class Config(BaseModel):
    sources: Annotated[dict[str, SourceConfig], Field(default_factory=dict)]
    strategies: Annotated[dict[str, dict], Field(default_factory=dict)]


def get_config() -> Config:
    return Config.model_validate(safe_load(example_config))

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        return Config.model_validate(safe_load(f))
