from pathlib import Path

import openai
from pydantic import BaseModel
from core_types import AssetType, PartialAsset, PartialChunk, PartialTask, Rule, Task, TaskStatus
from strategies.strategy import Strategy
from strategies.embed_chunks import EmbedChunksStrategy
from storage import get_db


PROMPT = """
This part the automated pipeline, splits text into chunks of 2-6 sentences each.

Output the chunks in a JSON array, with each chunk being a string, e.g.:

[
    "This is the first chunk.",
    "This is the second chunk.",
    ...
]
"""

class ResponseModel(BaseModel):
    chunks: list[str]


class ChunkFromTextConfig(BaseModel):
    model: str = "gpt-4o-mini"
    chars_per_chunk: int | None = None


class ChunkFromTextStrategy(Strategy[ChunkFromTextConfig]):
    NAME = "chunk_from_text"
    PRIORITY = 0
    MAX_BATCH_SIZE = 1
    RESOURCES = ["openai"]

    CONFIG_TYPE = ChunkFromTextConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.openai = openai.AsyncClient()

    def add_rules(self, rules):
        return rules + [
            Rule(pattern=AssetType.TEXT_FILE, strategy=self.NAME),
        ]

    async def process_all(self, tasks: list[Task]) -> None:
        db = get_db()

        assets = db.get_assets([task.input_asset_id for task in tasks])

        for asset in assets:
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
                    type=AssetType.CHUNK_ID,
                    content=str(chunk_id),
                ))

    async def chunk_text(self, text: str) -> list[str]:
        if self.config.chars_per_chunk:
            return self.chunk_text_by_chars(text, self.config.chars_per_chunk)
        else:
            return await self.llm_chunk_text(text)

    async def llm_chunk_text(self, text: str) -> list[str]:
        response = await self.openai.beta.chat.completions.parse(
            model=self.config.model,
            messages=[
                dict(role="system", content=PROMPT),
                dict(role="user", content=text),
            ],
            response_format=ResponseModel,
        )

        chunks = response.choices[0].message.parsed
        assert chunks is not None

        return chunks.chunks

    def chunk_text_by_chars(self, text: str, chunk_size: int) -> list[str]:
        """
        Splits the text into chunks of a given size.
        The last chunk can be ±50% of the chunk size.
        """

        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

        if len(chunks) > 2 and len(chunks[-1]) < chunk_size // 2:
            # Merge the last chunk with the previous one
            chunks[-2] += chunks[-1]
            chunks.pop()

        return chunks