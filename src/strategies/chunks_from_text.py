from core_types import Task
from strategies.strategy import Strategy


class ChunkFromTextStrategy(Strategy):
    NAME = "chunk_from_text"
    PRIORITY = 0
    MAX_BATCH_SIZE = 100
    RESOURCES = []

    async def process_all(self, tasks: list[Task]) -> None:
        raise NotImplemented("")

