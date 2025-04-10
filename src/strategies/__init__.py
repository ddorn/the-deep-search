from .add_sync_tokens import AddSyncTokenStrategy
from .auto_process import AutoProcessStrategy
from .chunks_from_text import ChunkFromTextStrategy
from .compress_audio import CompressAudioInPlaceStrategy
from .dl_audio import DlAudioStrategy
from .embed_chunks import EmbedChunksStrategy
from .strategy import Module
from .strategy import Source as Source
from .transcribe import TranscribeStrategy

BUILT_IN_STRATEGIES: dict[str, type[Module]] = {
    EmbedChunksStrategy.NAME: EmbedChunksStrategy,
    ChunkFromTextStrategy.NAME: ChunkFromTextStrategy,
    AutoProcessStrategy.NAME: AutoProcessStrategy,
    AddSyncTokenStrategy.NAME: AddSyncTokenStrategy,
    DlAudioStrategy.NAME: DlAudioStrategy,
    TranscribeStrategy.NAME: TranscribeStrategy,
    CompressAudioInPlaceStrategy.NAME: CompressAudioInPlaceStrategy,
}
