# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import dataclasses
import functools
import io
import operator
import typing

from ..core.file import File
from ._mp3 import Mp3Audio, is_mp3, mp3_chunk, parse_mp3
from ._wav import WavAudio, is_wav, parse_wav, wav_chunk

# The Decisions API accepts audio up to 5 minutes long.
MAX_CHUNK_MS = 300_000

# The API accepts a request body of up to 64,000,000 bytes, its JSON part and multipart headers
# included. The rest is left for those, so a chunk this size is accepted with any common request.
MAX_CHUNK_BYTES = 63_000_000


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
        The first chunk of an MP3 file can start a few tens of milliseconds before 0.

    end_ms : int
        Where the chunk ends in the recording, in milliseconds, rounded to the nearest millisecond.

    data : bytes
        The chunk as a file in the recording's format.
    """

    index: int
    start_ms: int
    end_ms: int
    data: bytes


@dataclasses.dataclass(frozen=True)
class PlannedChunk:
    """A chunk's place in the recording, with its data left to be copied when it is needed."""

    index: int
    start_ms: int
    end_ms: int
    build: typing.Callable[[], bytes]


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
    The audio is copied as it is, never decoded or re-encoded, so cuts fall where the format
    allows:

    - A WAV file is cut between samples, so a chunk can start or end up to one sample earlier than
      `start_ms` plus a whole multiple of `chunk_ms`. Each chunk is a new WAV file with the
      recording's sample format and only its audio; other chunks in the file, such as metadata,
      are left out.
    - An MP3 file is cut between frames, which last 24 to 72 ms depending on the sample rate. Each
      chunk holds as many whole frames as fit in `chunk_ms`, so it can be up to one frame shorter,
      and `start_ms` and `end_ms` move to the nearest frame boundary. A frame can store part of
      its audio data in the frames before it, and its sound continues from the frame before, so
      each chunk also starts with the frames its first new frame needs to decode exactly. That is
      usually a few frames, and up to about 2 seconds at the lowest bitrates. Those frames count
      toward `chunk_ms` and repeat the end of the previous chunk, or start before `start_ms`. Each
      chunk holds only the recording's frames; tags, and the header frame that states the whole
      file's length, are left out. When that header frame has a LAME tag, as files from LAME and
      ffmpeg do, it states the silence the encoder added at each end, which players skip. Times are
      then measured from where a player starts the whole file, so the first chunk starts before 0,
      by about 25 ms for a typical 44.1 kHz file, and the last ends where the player stops.

    `start_ms` and `end_ms` on each chunk give where it actually starts and ends.

    The whole recording is read into memory and checked before this function returns, so that a
    file that cannot be split is rejected before any chunk is produced. Each chunk's `data` is a
    copy, made as the chunk is produced, so keeping every chunk, as `list()` does, holds the
    recording in memory twice.

    Parameters
    ----------
    file : core.File
        The recording, an MP3 or WAV file, in any form `create` accepts. MP3 files of MPEG-1, 2 and
        2.5 layer III are supported, except free format. WAV files of integer PCM (8, 16, 24 or
        32-bit) and floating point (32 or 64-bit) are supported, with either the basic or the
        extensible WAV header. A file object is read in full from its start and left open. A `str` is the
        file's contents, not a path.

    chunk_ms : int
        Duration of each chunk in milliseconds, from 1 to 300000. A WAV chunk must also be at most
        63,000,000 bytes, so that the API accepts it, which limits 5-minute chunks to WAV files of
        up to about 210 kB per second, such as 48 kHz 16-bit stereo.

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
        chunk_ms is out of range, too long for a WAV chunk to fit the API's size limit, or too
        short for an MP3 chunk to hold a new frame and the frames it needs, start_ms and end_ms do
        not select audio within the recording, or the recording is empty or not a file this helper
        supports, such as a compressed WAV or RF64 file. Any damage to an MP3 file is found before
        this call returns, so no chunk is produced from a file that cannot be read to the end.

    Examples
    --------
    from hume_decisions.audio_helpers import iter_audio_chunks

    with open("call.wav", "rb") as f:
        chunks = iter_audio_chunks(f, chunk_ms=60_000)
    for chunk in chunks:
        print(chunk.index, chunk.start_ms, chunk.end_ms, len(chunk.data))
    """
    chunks = plan_audio_chunks(file, chunk_ms=chunk_ms, start_ms=start_ms, end_ms=end_ms)
    return (AudioChunk(chunk.index, chunk.start_ms, chunk.end_ms, chunk.build()) for chunk in chunks)


def plan_audio_chunks(
    file: File,
    *,
    chunk_ms: int,
    start_ms: typing.Optional[int] = None,
    end_ms: typing.Optional[int] = None,
) -> typing.Iterator[PlannedChunk]:
    """Checks and plans the chunks `iter_audio_chunks` gives, without copying any chunk's data."""
    chunk_ms = _check_integer("chunk_ms", chunk_ms)
    if not 1 <= chunk_ms <= MAX_CHUNK_MS:
        raise ValueError(f"chunk_ms must be between 1 and {MAX_CHUNK_MS}, got {chunk_ms}.")
    start = 0 if start_ms is None else _check_integer("start_ms", start_ms)
    if start < 0:
        raise ValueError(f"start_ms must not be negative, got {start}.")
    if end_ms is not None:
        end_ms = _check_integer("end_ms", end_ms)
        if end_ms <= start:
            raise ValueError(f"end_ms must be greater than start_ms, got {end_ms}.")
    data, given_as_str = _read(file)
    if is_wav(data):
        return _wav_chunks(data, parse_wav(data), chunk_ms, start, end_ms)
    if is_mp3(data):
        return _mp3_chunks(data, parse_mp3(data), chunk_ms, start, end_ms)
    message = "Unsupported audio format. Only MP3 and WAV files can be split into chunks."
    if given_as_str:
        message += (
            " The file was given as a str, which is read as the file's contents rather than a path."
            " Open the file in binary mode and pass the file object or its bytes."
        )
    raise ValueError(message)


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


def _check_range(count: int, duration_ms: int, start_ms: int, end_ms: typing.Optional[int]) -> None:
    if count == 0:
        raise ValueError("The audio is empty.")
    # The longest range allowed ends at the duration that the final chunk reports.
    # A recording shorter than half a millisecond has a duration of 0 ms but still starts at 0.
    if start_ms > 0 and start_ms >= duration_ms:
        raise ValueError(f"start_ms must be less than the recording's duration of {duration_ms} ms, got {start_ms}.")
    if end_ms is not None and end_ms > duration_ms:
        raise ValueError(f"end_ms must not exceed the recording's duration of {duration_ms} ms, got {end_ms}.")


def _wav_chunks(
    data: memoryview, wav: WavAudio, chunk_ms: int, start_ms: int, end_ms: typing.Optional[int]
) -> typing.Iterator[PlannedChunk]:
    duration_ms = _to_ms(wav.sample_count, wav.sample_rate)
    _check_range(wav.sample_count, duration_ms, start_ms, end_ms)
    # The duration is rounded to the millisecond, so it can fall short of the final sample.
    if end_ms is None or end_ms >= duration_ms:
        last = wav.sample_count
    else:
        last = min(_to_sample(end_ms, wav.sample_rate), wav.sample_count)
    _check_wav_size(wav, chunk_ms, last - _to_sample(start_ms, wav.sample_rate))
    return _iter_wav_chunks(data, wav, chunk_ms, start_ms, last)


def _check_wav_size(wav: WavAudio, chunk_ms: int, range_samples: int) -> None:
    # An MP3 chunk needs no such check: at the highest bitrate, 5 minutes of MP3 is about 12 MB.
    # The header holds the RIFF, fmt and data chunk headers, and the data can need a padding byte.
    header_bytes = 28 + len(wav.fmt) + 1
    samples = min(-(-chunk_ms * wav.sample_rate // 1000), range_samples)
    if header_bytes + samples * wav.frame_bytes <= MAX_CHUNK_BYTES:
        return
    fitting_ms = (MAX_CHUNK_BYTES - header_bytes) // wav.frame_bytes * 1000 // wav.sample_rate
    raise ValueError(
        f"chunk_ms must be at most {fitting_ms} for this WAV file, so that each chunk stays within the"
        f" {MAX_CHUNK_BYTES} bytes the API accepts along with the request."
    )


def _mp3_chunks(
    data: memoryview, mp3: Mp3Audio, chunk_ms: int, start_ms: int, end_ms: typing.Optional[int]
) -> typing.Iterator[PlannedChunk]:
    frame_samples = mp3.samples_per_frame
    duration_ms = _to_ms(mp3.end_sample - mp3.leading_samples, mp3.sample_rate)
    _check_range(mp3.frame_count, duration_ms, start_ms, end_ms)
    # Rounding down keeps every chunk within chunk_ms, so a chunk_ms of 300000 stays within the
    # API's limit.
    frames_per_chunk = chunk_ms * mp3.sample_rate // (1000 * frame_samples)
    frame_ms = _frames_ms(mp3, 1)
    if frames_per_chunk == 0:
        raise ValueError(f"chunk_ms must be at least {frame_ms} ms, the length of one frame of this MP3 file.")
    # Times given by the caller are often ones a chunk reported, which are rounded to the
    # millisecond, so they move to the nearest frame boundary rather than the one before. The start
    # and end of the recording can fall partway through a frame, so they keep that whole frame.
    first = 0 if start_ms == 0 else _frame_at(mp3, start_ms)
    last = mp3.frame_count if end_ms is None or end_ms >= duration_ms else min(_frame_at(mp3, end_ms), mp3.frame_count)
    if last <= first:
        raise ValueError(
            f"start_ms and end_ms must be at least {frame_ms} ms apart, the length of one frame of this MP3 file."
        )
    # A chunk that could hold only the frames needed before its first new frame would never advance.
    needed_before = max(mp3.frames_needed_before(frame) for frame in range(first, last))
    if frames_per_chunk <= needed_before:
        raise ValueError(
            f"chunk_ms must be at least {_frames_ms(mp3, needed_before + 1)} ms for this MP3 file. Its frames"
            f" need up to {needed_before} frames before them to decode, and a chunk must hold those along with"
            " at least one new frame."
        )
    return _iter_mp3_chunks(data, mp3, frames_per_chunk, first, last)


def _frame_at(mp3: Mp3Audio, ms: int) -> int:
    return _round_div(ms * mp3.sample_rate + 1000 * mp3.leading_samples, 1000 * mp3.samples_per_frame)


def _frames_ms(mp3: Mp3Audio, frames: int) -> int:
    return -(-frames * mp3.samples_per_frame * 1000 // mp3.sample_rate)


def _iter_mp3_chunks(
    data: memoryview, mp3: Mp3Audio, frames_per_chunk: int, first: int, last: int
) -> typing.Iterator[PlannedChunk]:
    index = 0
    new = first
    while new < last:
        # Without the frames before it, the first new frame decodes wrongly. They repeat the end of
        # the previous chunk.
        start = new - mp3.frames_needed_before(new)
        end = min(start + frames_per_chunk, last)
        # Times are measured from where a player starts the recording, after the samples the
        # encoder and decoder add, so the first chunk can start before 0 ms.
        yield PlannedChunk(
            index=index,
            start_ms=_to_ms(start * mp3.samples_per_frame - mp3.leading_samples, mp3.sample_rate),
            end_ms=_to_ms(min(end * mp3.samples_per_frame, mp3.end_sample) - mp3.leading_samples, mp3.sample_rate),
            build=functools.partial(mp3_chunk, data, mp3, start, end),
        )
        index += 1
        new = end


def _iter_wav_chunks(
    data: memoryview, audio: WavAudio, chunk_ms: int, start_ms: int, last: int
) -> typing.Iterator[PlannedChunk]:
    index = 0
    start = _to_sample(start_ms, audio.sample_rate)
    while start < last:
        # Each cut is computed from start_ms rather than from the previous cut, so rounding does not
        # accumulate.
        end = min(_to_sample(start_ms + (index + 1) * chunk_ms, audio.sample_rate), last)
        yield PlannedChunk(
            index=index,
            start_ms=_to_ms(start, audio.sample_rate),
            end_ms=_to_ms(end, audio.sample_rate),
            build=functools.partial(wav_chunk, data, audio, start, end),
        )
        index += 1
        start = end


def _to_sample(ms: int, sample_rate: int) -> int:
    return ms * sample_rate // 1000


def _to_ms(sample: int, sample_rate: int) -> int:
    return _round_div(sample * 1000, sample_rate)


def _round_div(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _check_integer(name: str, value: int) -> int:
    # operator.index also accepts NumPy integers, which are not subclasses of int. Its result is a
    # plain int, which arithmetic needs because a small NumPy integer can overflow.
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}.") from None
