# %%
import re
import sqlite3
from pathlib import Path
from pprint import pprint

from pydantic import BaseModel

from constants import SYNC_PATTERN
from core_types import AssetType, PartialAsset, PartialChunk, Task
from logs import logger
from storage import get_db
from strategies.strategy import Module


class ChunkFromTextConfig(BaseModel):
    # model: str = "gpt-4o-mini"
    chars_per_chunk: int = 1000


class ChunkFromTextStrategy(Module[ChunkFromTextConfig]):
    NAME = "chunk_from_text"
    PRIORITY = 0
    MAX_BATCH_SIZE = 1
    INPUT_ASSET_TYPE = AssetType.SYNCED_TEXT_FILE

    CONFIG_TYPE = ChunkFromTextConfig

    def __init__(self, config) -> None:
        super().__init__(config)

    async def process_all(self, tasks: list[Task]) -> None:
        db = get_db()

        assets = db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            assert asset.path is not None
            path = Path(asset.path)
            text = path.read_text()
            chunks = self.chunk_text(text)

            try:
                ids = db.create_chunks(
                    [
                        PartialChunk(
                            document_id=task.document_id,
                            document_order=i,
                            content=chunk,
                        )
                        for i, chunk in enumerate(chunks)
                    ],
                )
            except sqlite3.IntegrityError:
                pprint(asset)
                pprint(task)
                raise

            for chunk_id in ids:
                db.create_asset(
                    PartialAsset(
                        document_id=task.document_id,
                        created_by_task_id=task.id,
                        type=AssetType.CHUNK_ID,
                        content=str(chunk_id),
                    )
                )

    def chunk_text(self, text: str) -> list[str]:
        # We want chunks of self.config.chars_per_chunk ± 20%
        # They should overlap the previous and next chunk by 20%
        # We try to split them by, in order of priority:
        # - end of paragraph \n\n
        # - end of line \n
        # - end of sentence . or ? or !
        # - end of word
        # - or abruptly cut the text - but not in the sync token

        parts: list[tuple[str, str]] = []
        start = 0
        chunk_size = self.config.chars_per_chunk
        overlap = int(chunk_size * 0.2)
        chunk_size_buffer = int(chunk_size * 0.2)

        while start < len(text):
            chunk_center_end = start + chunk_size - chunk_size_buffer
            chunk_center = text[start:chunk_center_end]

            chunk_end_candidate = text[chunk_center_end : chunk_center_end + chunk_size_buffer * 2]
            if chunk_end_candidate:
                chunk_end_end = find_best_split(chunk_end_candidate)
                chunk_end = chunk_end_candidate[:chunk_end_end]
            else:
                chunk_end = ""

            chunk_start = text[start - int(1.5 * overlap) : start]
            if chunk_start:
                chunk_start_start = find_best_split(chunk_start[: -overlap // 2])
                chunk_start = chunk_start[chunk_start_start:]
            else:
                chunk_start = ""

            parts.append((chunk_start, chunk_center + chunk_end))
            start += len(chunk_center) + len(chunk_end)

        # If the last chunk is too short, merge it with the previous one
        if len(parts) > 1 and len(parts[-1][1]) < chunk_size * 0.5:
            parts[-2] = (parts[-2][0], parts[-2][1] + parts[-1][1])
            parts.pop()

        return [overlap + chunk for overlap, chunk in parts]


def find_best_split(text: str) -> int:
    # Find the best split point in the text
    # We want to find the first occurrence of one of the following:
    # - end of line \n (with more \n having higher priority)
    # - end of sentence . or ? or !
    # - end of word
    # - or abruptly cut the text

    # Find the largest amount of consecutive newlines
    consecutive_newlines = 0
    best_split = -1
    best_split_score = 0
    for i, c in enumerate(text):
        if c == "\n":
            consecutive_newlines += 1
            if consecutive_newlines > best_split_score:
                best_split = i + 1
                best_split_score = consecutive_newlines
        else:
            consecutive_newlines = 0
    if best_split != -1:
        return best_split

    # Find a sentence end
    match = re.search(r"[.!?]\s+", text)
    if match:
        return match.start() + 1

    # Find a space
    if (space := text.find(" ")) != -1:
        return space + 1

    # Split after a sync token
    for match in re.finditer(SYNC_PATTERN, text):
        return match.end()

    # There are no sync tokens in the center, but some might be cut at the edges
    # So we do after a '>' or before a '<'
    if (gt := text.rfind(">")) != -1:
        return gt + 1
    if (lt := text.find("<")) != -1:
        return lt

    # No sync tokens, and no other split point, just cut the text
    logger.warning("Could not find a split point in the text")
    logger.warning(f"Text: {text}")
    return 0


# %%

if __name__ == "__main__":
    chars_per_chunk = 40
    config = ChunkFromTextConfig(chars_per_chunk=chars_per_chunk)
    strategy = ChunkFromTextStrategy(config)

    text = """
    This is a test text.
    It has multiple lines.
    And some sentences.

    It also has some sync tokens <sync-id="0"> and <sync-id="1">.
    AndSomeMoreSyncTokens<sync-id="2">ItAlsoHasSomeSyncTokens<sync-id="3">.
    Lorem-ipsum-dolor-sit-amet-consectetur-adipiscing-elit-sed-do-eiusmod-tempor.
    """

    chunks = strategy.chunk_text(text)
    # Show a rule to count chars
    max_len = int(chars_per_chunk * 1.4)
    last_digits = "".join(str(i)[-1] for i in range(max_len))
    second_digits = "".join(f"{i:02d}"[-2] for i in range(max_len))
    print(f"   Rule: {second_digits}")
    print(f"   Rule: {last_digits}")
    markers = [" "] * (max_len + 1)
    markers[chars_per_chunk] = "*"
    markers[int(chars_per_chunk * 0.8)] = "<"
    markers[int(chars_per_chunk * 1.2)] = ">"
    print(f"   Rule: {''.join(markers)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk!r}")
    print(f"Number of chunks: {len(chunks)}")
