from .add_sync_tokens import AddSyncTokenStrategy
from .embed_chunks import EmbedChunksStrategy
from .chunks_from_text import ChunkFromTextStrategy
from .auto_process import AutoProcessStrategy
from .strategy import Strategy, Source

BUILT_IN_STRATEGIES: dict[str, type[Strategy]] = {
    EmbedChunksStrategy.NAME: EmbedChunksStrategy,
    ChunkFromTextStrategy.NAME: ChunkFromTextStrategy,
    AutoProcessStrategy.NAME: AutoProcessStrategy,
    AddSyncTokenStrategy.NAME: AddSyncTokenStrategy,
}