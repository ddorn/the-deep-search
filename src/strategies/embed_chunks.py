import openai
from core_types import AssetType, Rule, Task, PartialAsset
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

    def add_rules(self, rules):
        return rules + [
            Rule(pattern=AssetType.CHUNK_ID, action=self.NAME),
        ]

    async def process_all(self, tasks: list[Task]):
        db = get_db()

        asset_ids = [task.input_asset_id for task in tasks]
        assets = db.get_assets(asset_ids)
        chunk_ids = [int(asset.content) for asset in assets]
        chunks = db.get_chunks(chunk_ids)

        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts(texts)

        for chunk_id, task in zip(chunk_ids, tasks, strict=True):
            assert task.input_asset_id == str(chunk_id)
            db.create_asset(PartialAsset(
                document_id=task.document_id,
                created_by_task_id=task.id,
                type=AssetType.EMBEDDING_ID,
                content=str(chunk_id),
            ))

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