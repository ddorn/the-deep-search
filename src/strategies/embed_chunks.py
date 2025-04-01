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

    EMBEDDING_DIMENSIONS = 1536

    def __init__(self):
        super().__init__()
        self.openai = openai.AsyncClient()

    async def process_all(self, tasks: list[Task]):
        db = get_db()

        chunk_ids = [int(task.args) for task in tasks]
        chunks = db.get_chunks(chunk_ids)
        texts = [chunk.content for chunk in chunks]

        print(texts)
        response = await self.openai.embeddings.create(
            dimensions=self.EMBEDDING_DIMENSIONS,
            model="text-embedding-3-small",
            input=texts,
        )

        embeddings = np.zeros((len(texts), self.EMBEDDING_DIMENSIONS), dtype=np.float32)
        for embedding in response.data:
            embeddings[embedding.index] = embedding.embedding

        db.update_embeddings(chunk_ids, embeddings)

        return embeddings
