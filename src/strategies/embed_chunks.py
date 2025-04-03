import asyncio
import openai
from core_types import Task
from strategies.strategy import Strategy
from storage import get_db
import numpy as np


class EmbedChunksStrategy(Strategy):
    NAME = "embed_chunks"
    PRIORITY = 1
    MAX_BATCH_SIZE = 100
    RESOURCES = ["openai"]

    def __init__(self, config):
        super().__init__(config)
        self.openai = openai.AsyncClient()

    async def process_all(self, tasks: list[Task]):
        db = get_db()

        chunk_ids = [int(task.args) for task in tasks]
        chunks = db.get_chunks(chunk_ids)
        texts = [chunk.content for chunk in chunks]

        embeddings = await self.embed_texts(texts)

        db.update_embeddings(chunk_ids, embeddings)
        return embeddings

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        db = get_db()
        response = await self.openai.embeddings.create(
            dimensions=db.config.global_config.embedding_dimension,
            model="text-embedding-3-small",
            input=texts,
        )

        embeddings = np.zeros((len(texts), db.config.global_config.embedding_dimension), dtype=np.float32)
        for embedding in response.data:
            embeddings[embedding.index] = embedding.embedding

        return embeddings