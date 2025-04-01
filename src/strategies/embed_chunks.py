import openai
from core_types import Task
from strategies.strategy import Strategy
from storage import DATABASE
import numpy as np


class EmbedChunksStrategy(Strategy):
    NAME = "embed_chunks"
    PRIORITY = 1
    MAX_BATCH_SIZE = 100
    RESOURCES = ["openai"]

    EMBEDDING_DIMENSIONS = 1536

    def __init__(self):
        super().__init__()
        self.openai = openai.AsyncClient()

    async def process_all(self, tasks: list[Task]) -> None:
        chunk_ids = [task.args for task in tasks]
        chunks = DATABASE.get_chunks(chunk_ids)
        texts = [chunk.content for chunk in chunks]

        response = await self.openai.embeddings.create(
            dimensions=self.EMBEDDING_DIMENSIONS,
            model="text-embedding-3-small",
            input=texts,
        )

        embeddings = np.zeros((len(texts), self.EMBEDDING_DIMENSIONS), dtype=np.float32)
        for embedding in response.data:
            embeddings[embedding.index] = embedding.embedding

        DATABASE.update_embeddings(chunk_ids, embeddings)