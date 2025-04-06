from pathlib import Path

from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import SYNC_PATTERN
from core_types import AssetType, PartialAsset, PartialChunk, Task
from strategies.strategy import Strategy
from storage import get_db


class ResponseModel(BaseModel):
    chunks: list[str]


class ChunkFromTextConfig(BaseModel):
    # model: str = "gpt-4o-mini"
    chars_per_chunk: int = 400


class ChunkFromTextStrategy(Strategy[ChunkFromTextConfig]):
    NAME = "chunk_from_text"
    PRIORITY = 0
    MAX_BATCH_SIZE = 100
    INPUT_ASSET_TYPE = AssetType.SYNCED_TEXT_FILE

    CONFIG_TYPE = ChunkFromTextConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        chunk_size = config.chars_per_chunk
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10,
            keep_separator="start",
            is_separator_regex=True,
            strip_whitespace=False,
            separators=["\n\n", "\n", " ", SYNC_PATTERN.pattern, ""],
        )


    async def process_all(self, tasks: list[Task]) -> None:
        db = get_db()

        assets = db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            assert asset.path is not None
            path = Path(asset.path)
            text = path.read_text()
            chunks = await self.chunk_text(text)

            ids = db.create_chunks([
                PartialChunk(
                    document_id=asset.document_id,
                    document_order=i,
                    content=chunk,
                )
                for i, chunk in enumerate(chunks)
            ])

            for chunk_id in ids:
                db.create_asset(PartialAsset(
                    document_id=asset.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.CHUNK_ID,
                    content=str(chunk_id),
                ))

    async def chunk_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
