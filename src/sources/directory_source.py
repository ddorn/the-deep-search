import hashlib
from pathlib import Path

from pydantic import BaseModel
from gitignore_filter import git_ignore_filter

from sources.fingerprinted_source import FingerprintedSource, DocInfo
from core_types import AssetType, PartialAsset, PartialDocument
from logs import logger


class DirectorySourceConfig(BaseModel):
    path: Path
    ignore: str = ""


class DirectorySource(FingerprintedSource[DirectorySourceConfig]):
    NAME = "local-files"
    CONFIG_TYPE = DirectorySourceConfig

    def list_documents(self):
        for path in git_ignore_filter(
            self.config.path,
            self.config.ignore.splitlines(),
        ):
            yield DocInfo(
                urn=self.urn_for_path(path),
                title=str(path),
                fingerprint=self.compute_hash(self.config.path / path),
            )

    def mk_asset(self, document_id, doc):
        return PartialAsset(
            document_id=document_id,
            created_by_task_id=None,
            type=AssetType.GENERIC_FILE,
            path=self.full_path_from_urn(doc.urn),
        )

    def compute_hash(self, file_path: Path) -> str:
        file_hash = hashlib.blake2b()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                file_hash.update(chunk)

        return file_hash.hexdigest()

    def urn_for_path(self, path: Path) -> str:
        sep = "_%_%_"
        assert sep not in str(path), f"Path contains reserved separator: {path}"
        return str(path).replace("/", sep)

    def path_from_urn(self, urn: str) -> Path:
        sep = "_%_%_"
        return Path(urn.replace(sep, "/"))

    def full_path_from_urn(self, urn: str) -> Path:
        return self.config.path / self.path_from_urn(urn)
