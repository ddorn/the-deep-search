from .delete_document import DeleteDocumentStrategy
from .update_document import UpdateDocumentStrategy
from .embed_chunks import EmbedChunksStrategy
from .chunks_from_text import ChunkFromTextStrategy
from .auto_process import AutoProcessStrategy
from .strategy import Strategy, Source

BUILT_IN_STRATEGIES: dict[str, type[Strategy]] = {
    DeleteDocumentStrategy.NAME: DeleteDocumentStrategy,
    UpdateDocumentStrategy.NAME: UpdateDocumentStrategy,
    EmbedChunksStrategy.NAME: EmbedChunksStrategy,
    ChunkFromTextStrategy.NAME: ChunkFromTextStrategy,
    AutoProcessStrategy.NAME: AutoProcessStrategy,
}