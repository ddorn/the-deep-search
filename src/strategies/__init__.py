from .add_sync_tokens import AddSyncTokenStrategy
from .auto_process import AutoProcessStrategy
from .chunks_from_text import ChunkFromTextStrategy
from .compress_audio import CompressAudioInPlaceStrategy
from .dl_audio import DlAudioStrategy
from .embed_chunks import EmbedChunksStrategy
from .strategy import Module
from .strategy import Source as Source
from .transcribe import TranscribeStrategy
from .transcript_to_synced_text import TranscriptToSyncedTextStrategy

BUILT_IN_STRATEGIES: dict[str, type[Module]] = {
    EmbedChunksStrategy.NAME: EmbedChunksStrategy,
    ChunkFromTextStrategy.NAME: ChunkFromTextStrategy,
    AutoProcessStrategy.NAME: AutoProcessStrategy,
    AddSyncTokenStrategy.NAME: AddSyncTokenStrategy,
    TranscriptToSyncedTextStrategy.NAME: TranscriptToSyncedTextStrategy,
    DlAudioStrategy.NAME: DlAudioStrategy,
    TranscribeStrategy.NAME: TranscribeStrategy,
    CompressAudioInPlaceStrategy.NAME: CompressAudioInPlaceStrategy,
}
