import datetime
import time
from typing import Annotated

import feedparser
from pydantic import BaseModel, Field

from core_types import AssetType, PartialAsset
from logs import logger
from sources.fingerprinted_source import DocInfo, FingerprintedSource, FingerprintedConfig


class RssPodcastConfig(FingerprintedConfig):
    feed: Annotated[str, Field(description="RSS feed URL")]
    after: Annotated[datetime.datetime | None, Field(description="Optional: only include podcast released after this date")] = None


class RssPodcastSource(FingerprintedSource[RssPodcastConfig]):
    NAME = "rss-podcast"
    CONFIG_TYPE = RssPodcastConfig

    DISPLAY_NAME = "Podcasts from an RSS feed"
    DESCRIPTION = "Indexes podcasts from a given RSS feed and transcribes them."

    def list_documents(self):
        feed = feedparser.parse(self.config.feed)

        for entry in feed.entries:
            entry_time = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
            if self.config.after and entry_time < self.config.after:
                continue

            yield DocInfo(
                urn=entry.id,
                title=entry.title,
                fingerprint="",  # We never update them
                extra=dict(
                    audio_url=self.get_audio_url(entry),
                ),
            )

    def mk_asset(self, document_id, doc):
        return PartialAsset(
            document_id=document_id,
            created_by_task_id=None,
            type=AssetType.AUDIO_TO_DL,
            content=doc.extra["audio_url"],
        )

    def get_audio_url(self, feed_entry):
        # First, we try to find among the links on of type "audio/..."
        for link in feed_entry.links:
            if link.type.startswith("audio/"):
                return link.href

        logger.error(f"Could not find audio link in {feed_entry}")
        raise ValueError("No audio link found")
