import numpy as np
import openai

from constants import SYNC_PATTERN
from core_types import AssetType, PartialAsset, Task
from storage import Database
from strategies.strategy import Module


class EmbedChunksStrategy(Module):
    NAME = "embed_chunks"
    PRIORITY = -1
    MAX_BATCH_SIZE = 100
    INPUT_ASSET_TYPE = AssetType.CHUNK_ID

    def __init__(self, config, db: Database):
        super().__init__(config, db)
        self.openai = openai.AsyncClient()
        self.openai_sync = openai.Client()

    async def process_all(self, tasks: list[Task]):
        assets = self.db.get_assets([task.input_asset_id for task in tasks])
        chunks = self.db.get_chunks([int(asset.content) for asset in assets])

        texts = [chunk.content for chunk in chunks]
        # We first remove the syncing tokens
        texts = [SYNC_PATTERN.sub("", text) for text in texts]

        embeddings = await self.embed_texts(texts)

        for chunk, task in zip(chunks, tasks, strict=True):
            self.db.create_asset(
                PartialAsset(
                    document_id=task.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.EMBEDDING_ID,
                    content=str(chunk.id),
                )
            )

        self.db.update_embeddings([chunk.id for chunk in chunks], embeddings)

        return embeddings

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = await self.openai.embeddings.create(
            dimensions=self.db.config.embedding_dimension,
            model="text-embedding-3-small",
            input=texts,
        )

        return self._response_to_embeddings(response)

    def embed_texts_sync(self, texts: list[str]) -> np.ndarray:
        response = self.openai_sync.embeddings.create(
            dimensions=self.db.config.embedding_dimension,
            model="text-embedding-3-small",
            input=texts,
        )

        return self._response_to_embeddings(response)

    def _response_to_embeddings(self, response) -> np.ndarray:
        embeddings = np.zeros((len(response.data), self.db.config.embedding_dimension), dtype=np.float32)
        for embedding in response.data:
            embeddings[embedding.index] = embedding.embedding

        return embeddings
