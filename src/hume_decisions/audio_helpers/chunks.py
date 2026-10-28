# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import dataclasses
import operator
import typing

from ._wav import WavAudio, is_wav, parse_wav, wav_chunk

# The Decisions API accepts audio up to 5 minutes long.
MAX_CHUNK_MS = 300_000


@dataclasses.dataclass(frozen=True)
class AudioChunk:
    """
    One chunk of a longer recording, a complete audio file to send to the Decisions API on its own.

    Attributes
    ----------
    index : int
        Position of the chunk in the recording, starting at 0.

    start_ms : int
        Where the chunk starts in the recording, in milliseconds, rounded to the nearest millisecond.

    end_ms : int
        Where the chunk ends in the recording, in milliseconds, rounded to the nearest millisecond.

    data : bytes
        The chunk as a file in the recording's format.
    """

    index: int
    start_ms: int
    end_ms: int
    data: bytes


def iter_audio_chunks(file: bytes, *, chunk_ms: int) -> typing.Iterator[AudioChunk]:
    """
    Splits a recording into chunks short enough for the Decisions API.

    Every chunk is `chunk_ms` long except the last, which holds what remains. No chunk is empty.
    Cuts fall on whole samples, so a chunk can start or end up to one sample earlier than a whole
    multiple of `chunk_ms`.

    Each WAV chunk is a new WAV file with the recording's sample format and only its audio; other
    chunks in the file, such as metadata, are left out.

    Parameters
    ----------
    file : bytes
        The recording, a WAV file. Integer PCM (8, 16, 24 or 32-bit) and floating point (32 or
        64-bit) are supported, with either the basic or the extensible WAV header.

    chunk_ms : int
        Duration of each chunk in milliseconds, from 1 to 300000.

    Returns
    -------
    typing.Iterator[AudioChunk]
        The chunks in order.

    Raises
    ------
    ValueError
        chunk_ms is out of range, or the recording is empty or not a WAV file this helper supports,
        such as a compressed or RF64 file.

    Examples
    --------
    from hume_decisions.audio_helpers import iter_audio_chunks

    with open("call.wav", "rb") as f:
        recording = f.read()
    for chunk in iter_audio_chunks(recording, chunk_ms=60_000):
        print(chunk.index, chunk.start_ms, chunk.end_ms, len(chunk.data))
    """
    chunk_ms = _check_chunk_ms(chunk_ms)
    data = memoryview(file)
    if not is_wav(data):
        raise ValueError("Unsupported audio format. Only WAV files can be split into chunks.")
    audio = parse_wav(data)
    if audio.sample_count == 0:
        raise ValueError("The audio is empty.")
    return _iter_wav_chunks(data, audio, chunk_ms)


def _iter_wav_chunks(data: memoryview, audio: WavAudio, chunk_ms: int) -> typing.Iterator[AudioChunk]:
    index = 0
    start = 0
    while start < audio.sample_count:
        # Each cut is computed from the start of the recording, so rounding does not accumulate.
        end = min((index + 1) * chunk_ms * audio.sample_rate // 1000, audio.sample_count)
        yield AudioChunk(
            index=index,
            start_ms=_to_ms(start, audio.sample_rate),
            end_ms=_to_ms(end, audio.sample_rate),
            data=wav_chunk(data, audio, start, end),
        )
        index += 1
        start = end


def _to_ms(sample: int, sample_rate: int) -> int:
    return (sample * 2000 + sample_rate) // (2 * sample_rate)


def _check_chunk_ms(chunk_ms: int) -> int:
    # operator.index also accepts NumPy integers, which are not subclasses of int. Its result is a
    # plain int, which arithmetic needs because a small NumPy integer can overflow.
    try:
        chunk_ms = operator.index(chunk_ms)
    except TypeError:
        raise TypeError(f"chunk_ms must be an integer, got {chunk_ms!r}.") from None
    if not 1 <= chunk_ms <= MAX_CHUNK_MS:
        raise ValueError(f"chunk_ms must be between 1 and {MAX_CHUNK_MS}, got {chunk_ms}.")
    return chunk_ms
