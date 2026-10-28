# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import dataclasses
import io
import operator
import typing

from ..core.file import File
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


def iter_audio_chunks(
    file: File,
    *,
    chunk_ms: int,
    start_ms: typing.Optional[int] = None,
    end_ms: typing.Optional[int] = None,
) -> typing.Iterator[AudioChunk]:
    """
    Splits a recording, or part of one, into chunks short enough for the Decisions API.

    Every chunk is `chunk_ms` long except the last, which holds what remains. No chunk is empty.
    Cuts fall on whole samples, so a chunk can start or end up to one sample earlier than
    `start_ms` plus a whole multiple of `chunk_ms`.

    Each WAV chunk is a new WAV file with the recording's sample format and only its audio; other
    chunks in the file, such as metadata, are left out.

    Parameters
    ----------
    file : core.File
        The recording, a WAV file, in any form `create` accepts. Integer PCM (8, 16, 24 or 32-bit)
        and floating point (32 or 64-bit) are supported, with either the basic or the extensible
        WAV header. A file object is read in full from its start and left open. A `str` is the
        file's contents, not a path.

    chunk_ms : int
        Duration of each chunk in milliseconds, from 1 to 300000.

    start_ms : typing.Optional[int]
        Where in the recording to start, in milliseconds. Defaults to the start of the recording.

    end_ms : typing.Optional[int]
        Where in the recording to stop, in milliseconds, no later than the recording's duration.
        Defaults to the end of the recording.

    Returns
    -------
    typing.Iterator[AudioChunk]
        The chunks in order. Their times are measured from the start of the recording, not from
        `start_ms`.

    Raises
    ------
    ValueError
        chunk_ms is out of range, start_ms and end_ms do not select audio within the recording,
        or the recording is empty or not a WAV file this helper supports, such as a compressed or
        RF64 file.

    Examples
    --------
    from hume_decisions.audio_helpers import iter_audio_chunks

    with open("call.wav", "rb") as f:
        chunks = iter_audio_chunks(f, chunk_ms=60_000)
    for chunk in chunks:
        print(chunk.index, chunk.start_ms, chunk.end_ms, len(chunk.data))
    """
    chunk_ms = _check_integer("chunk_ms", chunk_ms)
    if not 1 <= chunk_ms <= MAX_CHUNK_MS:
        raise ValueError(f"chunk_ms must be between 1 and {MAX_CHUNK_MS}, got {chunk_ms}.")
    if start_ms is not None:
        start_ms = _check_integer("start_ms", start_ms)
        if start_ms < 0:
            raise ValueError(f"start_ms must not be negative, got {start_ms}.")
    if end_ms is not None:
        end_ms = _check_integer("end_ms", end_ms)
        if end_ms <= (start_ms or 0):
            raise ValueError(f"end_ms must be greater than start_ms, got {end_ms}.")
    data, given_as_str = _read(file)
    if not is_wav(data):
        message = "Unsupported audio format. Only WAV files can be split into chunks."
        if given_as_str:
            message += (
                " The file was given as a str, which is read as the file's contents rather than a path."
                " Open the file in binary mode and pass the file object or its bytes."
            )
        raise ValueError(message)
    audio = parse_wav(data)
    if audio.sample_count == 0:
        raise ValueError("The audio is empty.")
    # The longest range allowed ends at the duration that the final chunk reports.
    duration_ms = _to_ms(audio.sample_count, audio.sample_rate)
    if start_ms is None:
        start_ms = 0
    elif _to_sample(start_ms, audio.sample_rate) >= audio.sample_count:
        raise ValueError(f"start_ms must be less than the recording's duration of {duration_ms} ms, got {start_ms}.")
    if end_ms is None:
        end_ms = duration_ms
    elif end_ms > duration_ms:
        raise ValueError(f"end_ms must not exceed the recording's duration of {duration_ms} ms, got {end_ms}.")
    return _iter_wav_chunks(data, audio, chunk_ms, start_ms, end_ms)


def _read(file: File) -> typing.Tuple[memoryview, bool]:
    content = file[1] if isinstance(file, tuple) else file
    if not isinstance(content, (str, bytes)):
        # Read the way httpx reads a file it uploads, so the chunks cover what `create` would send.
        try:
            content.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        content = content.read()
    if isinstance(content, str):
        return memoryview(content.encode("utf-8")), True
    return memoryview(content), False


def _iter_wav_chunks(
    data: memoryview, audio: WavAudio, chunk_ms: int, start_ms: int, end_ms: int
) -> typing.Iterator[AudioChunk]:
    last = min(_to_sample(end_ms, audio.sample_rate), audio.sample_count)
    index = 0
    start = _to_sample(start_ms, audio.sample_rate)
    while start < last:
        # Each cut is computed from start_ms rather than from the previous cut, so rounding does not
        # accumulate.
        end = min(_to_sample(start_ms + (index + 1) * chunk_ms, audio.sample_rate), last)
        yield AudioChunk(
            index=index,
            start_ms=_to_ms(start, audio.sample_rate),
            end_ms=_to_ms(end, audio.sample_rate),
            data=wav_chunk(data, audio, start, end),
        )
        index += 1
        start = end


def _to_sample(ms: int, sample_rate: int) -> int:
    return ms * sample_rate // 1000


def _to_ms(sample: int, sample_rate: int) -> int:
    return (sample * 2000 + sample_rate) // (2 * sample_rate)


def _check_integer(name: str, value: int) -> int:
    # operator.index also accepts NumPy integers, which are not subclasses of int. Its result is a
    # plain int, which arithmetic needs because a small NumPy integer can overflow.
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}.") from None
