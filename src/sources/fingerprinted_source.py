import abc
import os
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

from core_types import PartialAsset, PartialDocument
from logs import logger
from storage import get_db
from strategies.strategy import Source


class DocInfo(BaseModel):
    urn: str
    title: str
    fingerprint: str
    extra: dict = Field(default_factory=dict)


class FingerprintedSource[ConfigType: BaseModel](Source[ConfigType]):

    def on_mount(self):

        if os.getenv("DS_NO_SYNC"):
            return

        db = get_db()
        to_create = []
        to_delete = set()

        past_docs = self.get_past_documents_urn()
        for doc in self.list_documents():
            past_docs.discard(doc.urn)
            fingerprint = self.stored_fingerprint(doc.urn)

            if fingerprint is None:
                logger.info(f"Document {doc.urn} new, creating")
                to_create.append(doc)
            elif fingerprint != doc.fingerprint:
                logger.info(f"Document {doc.urn} changed, re-creating")
                to_create.append(doc)
                to_delete.add(doc.urn)

        # The ones we haven't see were deleted
        for urn in past_docs:
            logger.info(f"Document {urn} deleted, removing from db")
            to_delete.add(urn)

        # Directly delete document from the db (deleted & changed ones)
        ids_to_delete = db.get_ids_from_urns(self.title, list(to_delete))
        if ids_to_delete:
            db.delete_documents(list(ids_to_delete.values()))
            for urn in to_delete:
                self.delete_fingerprint(urn)

        # (Re)create documents
        for doc in to_create:
            self.on_new_document(doc)

    def on_new_document(self, doc: DocInfo):
        db = get_db()

        document_id = db.create_document(
            PartialDocument(source_urn=doc.urn, source_id=self.title, title=doc.title), commit=False
        )
        db.create_asset(self.mk_asset(document_id, doc))

        self.save_fingerprint(doc.urn, doc.fingerprint)

    @abc.abstractmethod
    def mk_asset(self, document_id: int, doc: DocInfo) -> PartialAsset:
        """Create an asset from the document information."""
        pass

    @abc.abstractmethod
    def list_documents(self) -> Iterator[DocInfo]:
        """List documents from the source."""
        pass

    def fingerprint_path(self, urn: str) -> Path:
        return self.path_for_asset("fingerprint", urn)

    def stored_fingerprint(self, urn: str) -> str | None:
        try:
            return self.fingerprint_path(urn).read_text()
        except FileNotFoundError:
            return None

    def save_fingerprint(self, urn: str, fingerprint: str):
        self.fingerprint_path(urn).write_text(fingerprint)

    def delete_fingerprint(self, urn: str):
        try:
            self.fingerprint_path(urn).unlink()
        except FileNotFoundError:
            pass

    def get_past_documents_urn(self) -> set[str]:
        db = get_db()
        documents = db.get_documents_from_source(self.NAME)
        return {document.source_urn for document in documents}
