import hashlib
from pathlib import Path

from pydantic import BaseModel
from gitignore_filter import git_ignore_filter

from strategies.strategy import Source
from storage import get_db
from core_types import AssetType, PartialAsset, PartialDocument
from logs import logger


class DirectorySourceConfig(BaseModel):
    path: Path
    ignore: str = ""


class DirectorySource(Source[DirectorySourceConfig]):
    NAME = "local-files"
    CONFIG_TYPE = DirectorySourceConfig

    def on_mount(self):
        # Scan the source folder for new/changed/deleted documents
        # and create tasks to sync the database
        # with the source folder.

        db = get_db()
        past_documents_urn = self.get_past_documents_urn()

        to_delete = []
        to_create = {}

        for path in git_ignore_filter(
            self.config.path,
            self.config.ignore.splitlines()
        ):
            urn = self.urn_for_path(path)
            past_documents_urn.discard(urn)

            hashed_document = self.compute_hash(self.config.path / path)
            last_hash = self.read_hash(urn)

            if not last_hash:
                logger.info(f"New document: {path}")
                to_create[urn] = hashed_document
            elif hashed_document != last_hash:
                logger.info(f"Changed document: {path}")
                to_delete.append(urn)
                to_create[urn] = hashed_document

        # Remove documents that are not in the source folder
        for urn in past_documents_urn:
            logger.info(f"Deleted document: {urn}")
            to_delete.append(urn)

        # Direclty delete document from the db (deleted & changed ones)
        ids_to_delete = db.get_ids_from_urns(self.NAME, to_delete)
        db.delete_documents(list(ids_to_delete.values()))
        for urn in to_delete:
            self.delete_hash(urn)

        # (Re)create documents
        for urn, hashed_document in to_create.items():
            self.on_new_document(urn, hashed_document)

    def on_new_document(self, urn: str, hashed_document: str):
        db = get_db()

        document_id = db.create_document(
            PartialDocument(source_urn=urn, source_id=self.NAME), commit=False
        )
        db.create_asset(
            PartialAsset(
                document_id=document_id,
                type=AssetType.GENERIC_FILE,
                path=self.full_path_from_urn(urn),
            )
        )

        self.hash_path(urn).write_text(hashed_document)

    def hash_path(self, urn: str) -> Path:
        return self.path_for_asset("hash", urn)

    def read_hash(self, urn: str) -> str:
        try:
            return self.hash_path(urn).read_text()
        except FileNotFoundError:
            return ""

    def delete_hash(self, urn: str):
        try:
            self.hash_path(urn).unlink()
        except FileNotFoundError:
            pass

    def get_past_documents_urn(self) -> set[str]:
        db = get_db()
        documents = db.get_documents_from_source(self.NAME)
        return {document.source_urn for document in documents}

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
