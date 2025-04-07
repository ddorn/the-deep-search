from sources.gdrive_source import GDriveSource
from .directory_source import DirectorySource

BUILT_IN_SOURCES = {
    DirectorySource.NAME: DirectorySource,
    GDriveSource.NAME: GDriveSource,
}