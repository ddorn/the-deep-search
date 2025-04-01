import numpy as np

import pytest
import dotenv

from strategies.embed_chunks import EmbedChunksStrategy
from core_types import PartialChunk, Task
from storage import temporary_db

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_embed_chunks_strategy():
    strategy = EmbedChunksStrategy()
    assert strategy.NAME == "embed_chunks"

    with temporary_db("test") as db:
        # Need to create chunks in the temporary database
        ids = db.create_chunks([
            PartialChunk(
                document_id=1,
                document_order=i,
                content=f"This is a test chunk {i}",
            )
            for i in range(3)
        ])

        print("Chunk IDs:", ids)
        # Create a task for each chunk
        tasks = [
            Task(
                id=-1,  # Placeholder, the ID would be set by the database
                strategy=strategy.NAME,
                document_id=1,
                args=str(chunk_id),
            )
            for chunk_id in ids
        ]

        # Process the tasks
        embedings = await strategy.process_all(tasks)

        assert embedings.shape == (len(tasks), strategy.EMBEDDING_DIMENSIONS)
        assert embedings.dtype == np.float32
