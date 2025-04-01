import hashlib
import os
from pathlib import Path

from pydantic import BaseModel

from source import Source
from storage import DATABASE
from strategies import AutoProcessStrategy, DeleteDocumentStrategy, UpdateDocumentStrategy
from core_types import PartialTask


class ExtraConfig(BaseModel):
    path: Path


class DirectorySource(Source[ExtraConfig]):
    NAME = "local-files"
    EXTRA_CONFIG = ExtraConfig

    def add_tasks_from_changes(self):
        previous_scan = self._scan_directory(
            self.data_folder, lambda p: open(p).read().trim()
        )
        current_scan = self._scan_directory(
            self.config.path, lambda p: self._hash_file(p)
        )
        print(current_scan)

        for urn in previous_scan:
            if urn not in current_scan:
                # Document deleted
                DATABASE.create_task(
                    PartialTask(
                        strategy=DeleteDocumentStrategy.NAME,
                        document_id=urn,
                    )
                )

        for urn, current_hash in current_scan.items():
            if urn not in previous_scan:
                # New document
                DATABASE.create_task(
                    PartialTask(
                        strategy=AutoProcessStrategy.NAME,
                        document_id=urn,
                    )
                )
            elif previous_scan[urn] != current_hash:
                # Document has changed
                DATABASE.create_dependent_tasks(
                    PartialTask.create(
                        strategy=UpdateDocumentStrategy.NAME,
                        document_id=urn,
                    ),
                )

    def _scan_directory(self, root: str, fn):
        map = {}
        for top, dirs, files in os.walk(root):
            for file in files:
                file_path = os.path.join(top, file)

                map[self.get_urn_for_path(file_path)] = fn(file_path)

        return map

    def _hash_file(self, file_path):
        file_hash = hashlib.blake2b()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                file_hash.update(chunk)

        return file_hash.hexdigest()

    def get_urn_for_path(self, path):
        return f"urn:file:{path}"

    def get_path_from_urn(self, urn: str):
        return urn.replace('urn:file:', '')
