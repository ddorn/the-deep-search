import dotenv
import numpy as np

from core_types import PartialChunk, Task
from storage import temporary_db
from strategies.embed_chunks import EmbedChunksStrategy

dotenv.load_dotenv()


async def test_embed_chunks_strategy():
    strategy = EmbedChunksStrategy()
    assert strategy.NAME == "embed_chunks"

    with temporary_db() as db:
        # Need to create chunks in the temporary database
        ids = db.create_chunks(
            [
                PartialChunk(
                    document_id=1,
                    document_order=i,
                    content=f"This is a test chunk {i}",
                )
                for i in range(3)
            ]
        )

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
        embeddings = await strategy.process_all(tasks)

        assert embeddings.shape == (len(tasks), db.config.embedding_dimension)
        assert embeddings.dtype == np.float32
