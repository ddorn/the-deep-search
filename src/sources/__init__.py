from .gdrive_source import GDriveSource
from .rss_podcast_source import RssPodcastSource
from .directory_source import DirectorySource

BUILT_IN_SOURCES = {
    DirectorySource.NAME: DirectorySource,
    GDriveSource.NAME: GDriveSource,
    RssPodcastSource.NAME: RssPodcastSource,
}