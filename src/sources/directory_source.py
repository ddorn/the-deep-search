import hashlib
import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from gitignore_filter import git_ignore_filter

from strategies.strategy import Source
from storage import get_db
from strategies import (
    AutoProcessStrategy,
    DeleteDocumentStrategy,
    UpdateDocumentStrategy,
)
from core_types import PartialTask, PartialDocument
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

        past_documents_urn = set(git_ignore_filter(self.data_folder))

        for path in git_ignore_filter(
            self.config.path,
            self.config.ignore.splitlines()
        ):
            urn = self.get_urn_for_path(path)
            past_documents_urn.discard(urn)
            hash_file = self.data_folder / urn

            if hash_file.exists():
                hashed_document = self._hash_file(path)
                if hash_file.read_text() != hashed_document:
                    self.on_document_changed(path, hashed_document)

            else:
                self.on_new_document(path)

        # Remove documents that are not in the source folder
        for urn in past_documents_urn:
            self.on_document_deleted(urn)

    def on_document_deleted(self, urn: str):
        db = get_db()
        path = self.get_path_from_urn(urn)
        logger.debug(f"File '{path}' is deleted")
        document_id = db.get_document_id_from_urn(urn)
        if document_id is None:
            return
        db.create_task(
            PartialTask(
                strategy=DeleteDocumentStrategy.NAME,
                document_id=document_id,
                args=str(path),
            )
        )

    def on_new_document(self, path: Path):
        db = get_db()

        logger.debug(f"File '{path}' is new")
        urn = self.get_urn_for_path(path)

        document_id = db.create_document(
            PartialDocument(urn=urn, source_id=self.NAME)
        )
        db.create_task(
            PartialTask(
                strategy=AutoProcessStrategy.NAME,
                document_id=document_id,
                args=str(self.get_path_from_urn(urn)),
            )
        )

        hash_file = self.data_folder / urn
        hash_file.write_text(self._hash_file(path))

    def on_document_changed(self, path: Path, hashed_document: str):
        db = get_db()
        logger.debug(f"File '{path}' has changed")
        urn = self.get_urn_for_path(path)

        db.create_task(
            PartialTask(
                strategy=UpdateDocumentStrategy.NAME,
                document_id=db.get_document_id_from_urn(urn),
                args=str(self.get_path_from_urn(urn)),
            )
        )

        hash_file = self.data_folder / urn
        hash_file.write_text(hashed_document)


    def _hash_file(self, file_path):
        file_hash = hashlib.blake2b()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                file_hash.update(chunk)

        return file_hash.hexdigest()

    def get_urn_for_path(self, path: Path) -> str:
        sep = "_%_%_"
        assert sep not in str(path), f"Path contains reserved separator: {path}"
        return str(path).replace("/", sep)

    def get_path_from_urn(self, urn: str) -> Path:
        sep = "_%_%_"
        return Path(urn.replace(sep, "/"))
