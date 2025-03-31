# %%
from yaml import safe_load
from pydantic import BaseModel


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


class Source(BaseModel):
    type: str
    args: dict

class Config(BaseModel):
    sources: dict[str, Source]


def get_config() -> Config:
    return Config.model_validate(safe_load(example_config))
