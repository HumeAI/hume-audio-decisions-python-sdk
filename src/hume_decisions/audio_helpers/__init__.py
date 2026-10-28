# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

"""
Helpers that split audio longer than the Decisions API accepts into chunks it does accept.
"""

from .chunks import AudioChunk, iter_audio_chunks

__all__ = ["AudioChunk", "iter_audio_chunks"]
