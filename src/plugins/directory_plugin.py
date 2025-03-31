import hashlib
import os
import socket
from pathlib import Path
from storage import DATABASE
from source import Source


def get_previous_scan():
    # This where we'd query the database and return
    # the list of documents produced by this plugin
    # SELECT * from documents WHERE plugin_id = ?

    pass


class DirectorySource(Source):
    def __init__(self, path: str):
        self.path = path

    def retrieve_source(self, document_id: str) -> Path:
        urn = DATABASE.get_doc(document_id)["urn"]
        return self.get_path_from_urn(urn)

    # e -> get_changes() -> pour chaque changement

    def get_tasks_from_changes(self):
        previous_scan = self.get_previous_scan()
        current_scan
        changes = []

        for path in previous_scan:
            if path not in current_files:
                changes.append((path, "deleted"))

    def _scan_directory(self, root):
        map = {}
        for _, dirs, files in os.walk(root):
            for file in files:
                file_path = os.path.join(root, file)
                file_hash = self._hash_file(file_path)

                map[self.get_urn_for_path(file_path)] = file_hash
        return map

    def _hash_file(file_path):
        file_hash = hashlib.blake2b()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                file_hash.update(chunk)

        return file_hash.hexdigest()

    def get_urn_for_path(self, path):
        return f"urn:file:{socket.gethostname()}:{path}"

    def get_path_from_urn(self, urn):
        # TODO: If the path contains a colon, this fails.
        return urn.split(":")[-1]
