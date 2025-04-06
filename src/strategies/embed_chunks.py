import openai
from constants import SYNC_PATTERN
from core_types import AssetType, Task, PartialAsset
from strategies.strategy import Strategy
from storage import get_db
import numpy as np


class EmbedChunksStrategy(Strategy):
    NAME = "embed_chunks"
    PRIORITY = 1
    MAX_BATCH_SIZE = 100
    INPUT_ASSET_TYPE = AssetType.CHUNK_ID

    def __init__(self, config):
        super().__init__(config)
        self.openai = openai.AsyncClient()

    async def process_all(self, tasks: list[Task]):
        db = get_db()

        assets = db.get_assets([task.input_asset_id for task in tasks])
        chunks = db.get_chunks([int(asset.content) for asset in assets])

        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts(texts)

        for chunk, task in zip(chunks, tasks, strict=True):
            db.create_asset(PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.EMBEDDING_ID,
                content=str(chunk.id),
            ))

        db.update_embeddings([chunk.id for chunk in chunks], embeddings)

        return embeddings

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        db = get_db()

        # We first remove the syncing tokens
        texts = [
            SYNC_PATTERN.sub("", text)
            for text in texts
        ]

        response = await self.openai.embeddings.create(
            dimensions=db.config.global_config.embedding_dimension,
            model="text-embedding-3-small",
            input=texts,
        )

        embeddings = np.zeros((len(texts), db.config.global_config.embedding_dimension), dtype=np.float32)
        for embedding in response.data:
            embeddings[embedding.index] = embedding.embedding

        return embeddings