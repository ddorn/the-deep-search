import asyncio
import datetime
import re
from asyncio import Task
from pathlib import Path
from typing import Iterator

import aiohttp
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pydantic import BaseModel

from core_types import Asset, AssetType, PartialAsset, Rule
from logs import logger
from sources.fingerprinted_source import DocInfo, FingerprintedSource


class GDriveSourceConfig(BaseModel):
    folder_id: str
    service_account_file: Path


class GDriveSource(FingerprintedSource[GDriveSourceConfig]):
    NAME = "google-drive"
    CONFIG_TYPE = GDriveSourceConfig
    MAX_BATCH_SIZE = 20

    def __init__(self, config, title):
        super().__init__(config, title)

        self.credentials = service_account.Credentials.from_service_account_file(
            config.service_account_file,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )

    def list_documents(self) -> Iterator[DocInfo]:
        logger.debug(f"Listing google docs for {self.title}. This may take a few seconds...")
        yield from list_all_gdocs_fast(self.config.folder_id, self.credentials)
        logger.debug(f"Finished listing google docs for {self.title}.")

    def mk_asset(self, document_id, doc):
        return PartialAsset(
            document_id=document_id,
            created_by_task_id=None,
            type=AssetType.GOOGLE_DOC,
            content=doc.urn,
        )

    def add_rules(self, rules):
        return rules + [
            Rule(
                source=self.title,
                asset_type=AssetType.GOOGLE_DOC,
                strategy=self.title,
            ),
        ]

    async def process_all(self, tasks: list[Task]):
        """Download Google Docs."""

        assets = self.db.get_assets([task.input_asset_id for task in tasks])

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[self.process(task, asset, session) for task, asset in zip(tasks, assets)]
            )

    async def process(self, task: Task, asset: Asset, session: aiohttp.ClientSession):
        """Download Google Docs."""

        # Download the Google Doc as markdown
        export_link = f"https://www.googleapis.com/drive/v3/files/{asset.content}/export?mimeType=text/markdown"
        # Use access token to authenticate
        headers = {
            "Authorization": f"Bearer {self.credentials.token}",
        }

        async with session.get(export_link, headers=headers) as response:
            response.raise_for_status()
            gdoc = await response.text()

        # Remove images, which are inline! (eg: [image1]: <data:image/png;base64...)
        gdoc = re.sub(r"\[image\d+]:\s*<data:image/png;base64,[^>]+>", "", gdoc)

        path = self.path_for_asset("gdoc", asset.content)
        path.write_text(gdoc)

        self.db.create_asset(
            PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.TEXT_FILE,
                path=path,
            )
        )


# Thanks claude!
def list_all_gdocs_fast(folder_id: str, credentials) -> list[DocInfo]:

    # Build the Drive service
    service = build("drive", "v3", credentials=credentials)

    # print("Fetching folder structure and documents...")
    # Store folder info and docs
    folders = {}
    folder_hierarchy = {}
    all_docs = []

    # Get root folder info
    root_folder = service.files().get(fileId=folder_id, fields="name").execute()
    folders[folder_id] = root_folder["name"]

    # Use breadth-first traversal
    current_level = [folder_id]
    all_folder_ids = set(current_level)

    while current_level:
        # Build a query to get BOTH folders AND Google Docs at this level in one request
        if len(current_level) == 1:
            query = f"('{current_level[0]}' in parents) and (mimeType='application/vnd.google-apps.folder' or mimeType='application/vnd.google-apps.document') and trashed=false"
        else:
            parent_conditions = [f"'{parent}' in parents" for parent in current_level]
            query = f"({' or '.join(parent_conditions)}) and (mimeType='application/vnd.google-apps.folder' or mimeType='application/vnd.google-apps.document') and trashed=false"

        next_level = []
        page_token = None

        # Paginate through results if needed
        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, parents, modifiedTime)",
                    pageToken=page_token,
                    pageSize=1000,
                )
                .execute()
            )

            # Process each item (folder or document)
            for item in response.get("files", []):
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    # It's a folder
                    folder_id = item["id"]
                    if folder_id not in all_folder_ids:
                        folders[folder_id] = item["name"]
                        # Store the parent (first parent if multiple)
                        if "parents" in item and item["parents"]:
                            folder_hierarchy[folder_id] = item["parents"][0]
                        next_level.append(folder_id)
                        all_folder_ids.add(folder_id)
                else:
                    # It's a Google Doc
                    if "parents" in item and item["parents"]:
                        item["folder_id"] = item["parents"][0]
                    all_docs.append(item)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        # Move to the next level
        current_level = next_level

    logger.info(f"Found {len(folders)} folders and {len(all_docs)} Google Docs")

    # Build full paths for each document
    returned_docs = []
    for doc in all_docs:
        modified_time = datetime.datetime.fromisoformat(doc["modifiedTime"].replace("Z", "+00:00"))

        # Build the full path
        path_parts = [doc["name"]]

        # Only process hierarchy if folder_id is available
        if "folder_id" in doc:
            current_folder = doc["folder_id"]

            # Traverse up the folder hierarchy
            while current_folder in folder_hierarchy:
                path_parts.insert(0, folders[current_folder])
                current_folder = folder_hierarchy[current_folder]

            # Add the root folder
            if current_folder == folder_id:
                path_parts.insert(0, folders[folder_id])

        # Join the path parts
        # full_path = '/'.join(path_parts)
        # print(f"{full_path} - Last modified: {modified_time.isoformat()}")

        returned_docs.append(
            DocInfo(
                urn=doc["id"],
                title=doc["name"],
                fingerprint=modified_time.isoformat(),
            )
        )

    return returned_docs
