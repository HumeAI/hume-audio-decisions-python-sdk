# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

"""
Helpers that split audio longer than the Decisions API accepts into chunks it does accept.
"""

from ._chunks import AudioChunk, ChunkSplit, iter_audio_chunks
from ._downmix import DownmixStrategy
from ._results import ChunkedDecisionsResult, ChunkResult, ChunkStatus

__all__ = [
    "AudioChunk",
    "ChunkResult",
    "ChunkSplit",
    "ChunkStatus",
    "ChunkedDecisionsResult",
    "DownmixStrategy",
    "iter_audio_chunks",
]
