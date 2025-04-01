from pathlib import Path

import openai
from pydantic import BaseModel
from core_types import PartialChunk, Task
from strategies.strategy import Strategy
from storage import get_db


PROMPT = """
This part the automated pipeline, splits text into chunks of 2-6 sentences each.

Ouptut the chunks in a JSON array, with each chunk being a string, e.g.:
[
    "This is the first chunk.",
    "This is the second chunk.",
    ...
]
"""

class ResponseModel(BaseModel):
    chunks: list[str]



class ChunkFromTextStrategy(Strategy):
    NAME = "chunk_from_text"
    PRIORITY = 0
    MAX_BATCH_SIZE = 1
    RESOURCES = ["openai"]
    MODEL = "gpt-4o-mini"

    def __init__(self) -> None:
        super().__init__()
        self.openai = openai.AsyncClient()

    async def process_all(self, tasks: list[Task]) -> None:
        db = get_db()

        for task in tasks:
            path = Path(task.args)
            text = path.read_text()
            chunks = await self.chunk_text(text)

            db.create_chunks([
                PartialChunk(
                    document_id=task.document_id,
                    document_order=i,
                    content=chunk,
                )
                for i, chunk in enumerate(chunks)
            ])

    async def chunk_text(self, text: str) -> list[str]:
        response = await self.openai.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                dict(role="system", content=PROMPT),
                dict(role="user", content=text),
            ],
            response_format=ResponseModel,
        )

        chunks = response.choices[0].message.parsed
        assert chunks is not None

        return chunks.chunks