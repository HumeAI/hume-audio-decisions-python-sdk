# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import dataclasses
import functools
import typing

from ..core.file import File
from ._downmix import DownmixStrategy, check_downmix_strategy, downmix_wav
from ._files import read_file
from ._integers import ceil_div, check_integer, round_div
from ._mp3 import Mp3Audio, is_mp3, mp3_chunk, parse_mp3
from ._wav import WavAudio, is_wav, parse_wav, wav_chunk

# The Decisions API accepts audio up to 5 minutes long.
MAX_CHUNK_MS = 300_000

# The API accepts a request body of up to 25,000,000 bytes, its JSON part and multipart headers
# included. The rest is left for those, so a chunk this size is accepted with any common request.
MAX_CHUNK_BYTES = 24_000_000

ChunkSplit = typing.Literal["even", "fixed"]


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
        The chunk as a file in the recording's format, after any mixdown `downmix_strategy` asks for.
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
    max_chunk_ms: int,
    start_ms: typing.Optional[int] = None,
    end_ms: typing.Optional[int] = None,
    split: ChunkSplit = "even",
    downmix_strategy: typing.Optional[DownmixStrategy] = "mix",
) -> typing.Iterator[AudioChunk]:
    """
    Splits a recording, or part of one, into chunks short enough for the Decisions API.

    `max_chunk_ms` is the longest a chunk can be, and `split` decides how the recording is divided:

    - "even", the default, cuts the fewest chunks that fit, of equal length, so that the last chunk
      is not a short remainder: 10 minutes and 1 second at a `max_chunk_ms` of 300000 is cut into 3
      chunks of about 200 seconds.
    - "fixed" makes every chunk `max_chunk_ms` long except the last, which holds what remains: the
      same recording is cut into 2 chunks of 5 minutes and 1 of 1 second. The last chunk is billed
      for its context and questions like any other, however short it is. Use it when chunks must be
      the same length across recordings.

    No chunk is empty. Apart from the WAV mixdown `downmix_strategy` asks for, the audio is copied
    as it is, never decoded or re-encoded, so cuts fall where the format allows:

    - A WAV file is cut between samples, after the mixdown. With "fixed", a cut can fall up to one sample before
      `start_ms` plus a whole multiple of `max_chunk_ms`, so the cuts stay on that grid. Each chunk
      is a new WAV file in the sample format of the mixed-down audio, holding only its audio; other
      chunks in the file, such as metadata, are left out.
    - An MP3 file is cut between frames, which last 24 to 72 ms depending on the sample rate, so
      chunks can differ in length by a frame, and a "fixed" chunk can be up to one frame shorter
      than `max_chunk_ms`. `start_ms` and `end_ms` move to the nearest frame boundary. A frame can
      store part of its audio data in the frames before it, and its sound continues from the frame
      before, so each chunk also starts with the frames its first new frame needs to decode exactly.
      That is usually a few frames, and up to about 2 seconds at the lowest bitrates. Those frames
      count toward `max_chunk_ms` and repeat the end of the previous chunk, or start before
      `start_ms`. Each chunk holds only the recording's frames; tags, and the header frame that
      states the whole file's length, are left out. When that header frame
      has a LAME tag, as files from LAME and ffmpeg do, it states the silence the encoder added at
      each end, which players skip. Times are then measured from where a player starts the whole
      file, so the first chunk starts before 0, by about 25 ms for a typical 44.1 kHz file, and the
      last ends where the player stops.

    `start_ms` and `end_ms` on each chunk give where it actually starts and ends.

    The model hears audio at 16 kHz in one channel: the API resamples and mixes down every file
    before the model hears it. A WAV file is mixed down the same way before it is cut, by default,
    so that the chunks are smaller to upload while the model hears the same audio.
    `downmix_strategy` describes how. MP3 files are cut as they are, because they are already
    compressed.

    The whole recording is read into memory, mixed down and checked before this function returns,
    so that a file that cannot be split is rejected before any chunk is produced. Each chunk's
    `data` is a copy, made as the chunk is produced, so keeping every chunk, as `list()` does, holds
    the recording in memory twice.

    Parameters
    ----------
    file : core.File
        The recording, an MP3 or WAV file, in any form `create` accepts. MP3 files of MPEG-1, 2 and
        2.5 layer III are supported, except free format. WAV files of integer PCM (8, 16, 24 or
        32-bit) and floating point (32 or 64-bit) are supported, with either the basic or the
        extensible WAV header. A file object is read in full from its start and left open. A `str`
        is the file's contents, not a path.

    max_chunk_ms : int
        The longest a chunk can be, in milliseconds, from 1 to 300000. A WAV chunk must also be at
        most 24,000,000 bytes, so that the API accepts it. 5 minutes of WAV fits only at just under
        80,000 bytes per second, and always fits for mono or stereo under "mix". With "even", a WAV
        file that does not fit is cut into shorter chunks that do. With "fixed", it raises, and 16-bit stereo
        that is not mixed down needs a `max_chunk_ms` of at most 136054 at 44.1 kHz and 124999 at
        48 kHz.

    start_ms : typing.Optional[int]
        Where in the recording to start, in milliseconds. Defaults to the start of the recording.

    end_ms : typing.Optional[int]
        Where in the recording to stop, in milliseconds, no later than the recording's duration.
        Defaults to the end of the recording.

    split : typing.Literal["even", "fixed"]
        "even" for the fewest chunks of equal length, or "fixed" for chunks of `max_chunk_ms` with a
        shorter last chunk. Defaults to "even".

    downmix_strategy : typing.Optional[typing.Literal["mix", "first_channel"]]
        How a WAV file is mixed down before it is cut. Audio above 16 kHz is resampled to 16 kHz with
        the filter the API uses, and audio at 16 kHz or below keeps its rate. Resampled audio is
        written as 32-bit floating point, the format the API converts every file to, so loud audio
        cannot clip.

        - "mix", the default, mixes two channels into one the way the API does, adding them at -3 dB
          each, so the model hears the same audio as it would from the original file. A file that
          this would not make smaller, such as 16-bit mono at 16 kHz, is cut as it is, and so is
          audio with more than two channels, because the API weighs those channels by the speaker
          positions its header declares.
        - "first_channel" keeps only the first channel of a file with more than one, so the model
          hears only that channel. The channel is copied exactly in the recording's sample format
          when that is no larger than resampling it, as it is at 16 kHz or below for every sample
          format but 64-bit floating point. Use it when that channel holds the speaker to judge.
          Mono audio is treated as by "mix".
        - None cuts the audio as it is.

    Returns
    -------
    typing.Iterator[AudioChunk]
        The chunks in order. Their times are measured from the start of the recording, not from
        `start_ms`.

    Raises
    ------
    ValueError
        split is not "even" or "fixed", downmix_strategy is not "mix", "first_channel" or None,
        max_chunk_ms is out of range, too long with "fixed" for a
        WAV chunk to fit the API's size limit, or too short for an MP3 chunk to hold a new frame and
        the frames it needs, start_ms and end_ms do not select audio within the recording, or the
        recording is empty or not a file this helper supports, such as a compressed WAV or RF64
        file. Any damage to an MP3 file is found before this call returns, so no chunk is produced
        from a file that cannot be read to the end.

    Examples
    --------
    from hume_decisions.audio_helpers import iter_audio_chunks

    with open("call.wav", "rb") as f:
        chunks = iter_audio_chunks(f, max_chunk_ms=60_000)
    for chunk in chunks:
        print(chunk.index, chunk.start_ms, chunk.end_ms, len(chunk.data))
    """
    chunks = plan_audio_chunks(
        file,
        max_chunk_ms=max_chunk_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        split=split,
        downmix_strategy=downmix_strategy,
    )
    return (AudioChunk(chunk.index, chunk.start_ms, chunk.end_ms, chunk.build()) for chunk in chunks)


def plan_audio_chunks(
    file: File,
    *,
    max_chunk_ms: int,
    start_ms: typing.Optional[int],
    end_ms: typing.Optional[int],
    split: ChunkSplit,
    downmix_strategy: typing.Optional[DownmixStrategy],
) -> typing.Iterator[PlannedChunk]:
    """Checks and plans the chunks `iter_audio_chunks` gives, without copying any chunk's data."""
    if split not in typing.get_args(ChunkSplit):
        raise ValueError(f'split must be "even" or "fixed", got {split!r}.')
    check_downmix_strategy(downmix_strategy)
    max_chunk_ms = check_integer("max_chunk_ms", max_chunk_ms)
    if not 1 <= max_chunk_ms <= MAX_CHUNK_MS:
        raise ValueError(f"max_chunk_ms must be between 1 and {MAX_CHUNK_MS}, got {max_chunk_ms}.")
    start = 0 if start_ms is None else check_integer("start_ms", start_ms)
    if start < 0:
        raise ValueError(f"start_ms must not be negative, got {start}.")
    if end_ms is not None:
        end_ms = check_integer("end_ms", end_ms)
        if end_ms <= start:
            raise ValueError(f"end_ms must be greater than start_ms, got {end_ms}.")
    data, given_as_str = read_file(file)
    if is_wav(data):
        data, wav = downmix_wav(data, parse_wav(data), downmix_strategy)
        return _wav_chunks(data, wav, max_chunk_ms, start, end_ms, split)
    if is_mp3(data):
        return _mp3_chunks(data, parse_mp3(data), max_chunk_ms, start, end_ms, split)
    message = "Unsupported audio format. Only MP3 and WAV files can be split into chunks."
    if given_as_str:
        message += (
            " The file was given as a str, which is read as the file's contents rather than a path."
            " Open the file in binary mode and pass the file object or its bytes."
        )
    raise ValueError(message)


def _check_range(duration_ms: int, start_ms: int, end_ms: typing.Optional[int]) -> None:
    # The longest range allowed ends at the duration that the final chunk reports.
    # A recording shorter than half a millisecond has a duration of 0 ms but still starts at 0.
    if start_ms > 0 and start_ms >= duration_ms:
        raise ValueError(f"start_ms must be less than the recording's duration of {duration_ms} ms, got {start_ms}.")
    if end_ms is not None and end_ms > duration_ms:
        raise ValueError(f"end_ms must not exceed the recording's duration of {duration_ms} ms, got {end_ms}.")


def _wav_chunks(
    data: memoryview, wav: WavAudio, max_chunk_ms: int, start_ms: int, end_ms: typing.Optional[int], split: ChunkSplit
) -> typing.Iterator[PlannedChunk]:
    if wav.sample_count == 0:
        raise ValueError("The audio is empty.")
    duration_ms = _to_ms(wav.sample_count, wav.sample_rate)
    _check_range(duration_ms, start_ms, end_ms)
    # The duration is rounded to the millisecond, so it can fall short of the final sample.
    if end_ms is None or end_ms >= duration_ms:
        last = wav.sample_count
    else:
        last = min(_to_sample(end_ms, wav.sample_rate), wav.sample_count)
    first = _to_sample(start_ms, wav.sample_rate)
    if split == "even":
        cuts = _even_wav_cuts(wav, max_chunk_ms, first, last)
    else:
        _check_wav_size(wav, max_chunk_ms, last - first)
        cuts = _fixed_wav_cuts(wav, max_chunk_ms, start_ms, last)
    return _plan(
        cuts,
        functools.partial(_to_ms, sample_rate=wav.sample_rate),
        functools.partial(wav_chunk, data, wav),
    )


def _wav_samples_that_fit(wav: WavAudio) -> int:
    # An MP3 chunk needs no such limit: at the highest bitrate, 5 minutes of MP3 is about 12 MB.
    # The header holds the RIFF, fmt and data chunk headers, and the data can need a padding byte.
    header_bytes = 28 + len(wav.fmt) + 1
    return (MAX_CHUNK_BYTES - header_bytes) // wav.frame_bytes


def _check_wav_size(wav: WavAudio, max_chunk_ms: int, range_samples: int) -> None:
    fitting = _wav_samples_that_fit(wav)
    if min(ceil_div(max_chunk_ms * wav.sample_rate, 1000), range_samples) <= fitting:
        return
    fitting_ms = fitting * 1000 // wav.sample_rate
    raise ValueError(
        f"max_chunk_ms must be at most {fitting_ms} for this WAV file, so that each chunk stays within the"
        f" {MAX_CHUNK_BYTES} bytes the API accepts along with the request."
    )


def _mp3_chunks(
    data: memoryview, mp3: Mp3Audio, max_chunk_ms: int, start_ms: int, end_ms: typing.Optional[int], split: ChunkSplit
) -> typing.Iterator[PlannedChunk]:
    if mp3.frame_count == 0:
        raise ValueError("The audio is empty.")
    duration_ms = _mp3_time_ms(mp3, mp3.frame_count)
    _check_range(duration_ms, start_ms, end_ms)
    # Rounding down keeps every chunk within max_chunk_ms, so a max_chunk_ms of 300000 stays within
    # the API's limit.
    frames_per_chunk = max_chunk_ms * mp3.sample_rate // (1000 * mp3.samples_per_frame)
    frame_ms = _frames_ms(mp3, 1)
    if frames_per_chunk == 0:
        raise ValueError(f"max_chunk_ms must be at least {frame_ms} ms, the length of one frame of this MP3 file.")
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
    # Filling fails only when this check does, which also rejects a max_chunk_ms that happens to work
    # for this range's cuts but not for every frame in it.
    needed_before = max(mp3.frames_needed_before(frame) for frame in range(first, last))
    filled = _fill_mp3_chunks(mp3, frames_per_chunk, first, last)
    if filled is None or frames_per_chunk <= needed_before:
        raise ValueError(
            f"max_chunk_ms must be at least {_frames_ms(mp3, needed_before + 1)} ms for this MP3 file. Its frames"
            f" need up to {needed_before} frames before them to decode, and a chunk must hold those along with"
            " at least one new frame."
        )
    return _plan(
        _even_mp3_cuts(mp3, filled, frames_per_chunk, first, last) if split == "even" else filled,
        functools.partial(_mp3_time_ms, mp3),
        functools.partial(mp3_chunk, data, mp3),
    )


def _frame_at(mp3: Mp3Audio, ms: int) -> int:
    return round_div(ms * mp3.sample_rate + 1000 * mp3.leading_samples, 1000 * mp3.samples_per_frame)


def _frames_ms(mp3: Mp3Audio, frames: int) -> int:
    return ceil_div(frames * mp3.samples_per_frame * 1000, mp3.sample_rate)


def _mp3_time_ms(mp3: Mp3Audio, frame: int) -> int:
    # Times are measured from where a player starts the recording, after the samples the encoder and
    # decoder add, so the first chunk can start before 0 ms. The player also stops before the
    # samples added at the end.
    return _to_ms(min(frame * mp3.samples_per_frame, mp3.end_sample) - mp3.leading_samples, mp3.sample_rate)


def _even_mp3_cuts(
    mp3: Mp3Audio, filled: typing.List[typing.Tuple[int, int]], frames_per_chunk: int, first: int, last: int
) -> typing.List[typing.Tuple[int, int]]:
    # Filling every chunk gives the fewest chunks but can leave a short last chunk. The shortest
    # chunk length that needs no more chunks than that shares the frames out evenly instead.
    cuts = filled
    low, high = 1, frames_per_chunk
    while low < high:
        middle = (low + high) // 2
        shorter = _fill_mp3_chunks(mp3, middle, first, last)
        if shorter is not None and len(shorter) <= len(cuts):
            cuts, high = shorter, middle
        else:
            low = middle + 1
    return cuts


def _fill_mp3_chunks(
    mp3: Mp3Audio, frames_per_chunk: int, first: int, last: int
) -> typing.Optional[typing.List[typing.Tuple[int, int]]]:
    """Cuts chunks of `frames_per_chunk` frames, or None when a chunk cannot hold a new frame."""
    cuts: typing.List[typing.Tuple[int, int]] = []
    new = first
    while new < last:
        # Without the frames before it, the first new frame decodes wrongly. They repeat the end of
        # the previous chunk.
        start = new - mp3.frames_needed_before(new)
        if start + frames_per_chunk <= new:
            return None
        end = min(start + frames_per_chunk, last)
        cuts.append((start, end))
        new = end
    return cuts


def _even_wav_cuts(wav: WavAudio, max_chunk_ms: int, first: int, last: int) -> typing.Iterator[typing.Tuple[int, int]]:
    # When max_chunk_ms is not a whole number of samples, a chunk can be up to one sample longer.
    most = min(ceil_div(max_chunk_ms * wav.sample_rate, 1000), _wav_samples_that_fit(wav))
    count = ceil_div(last - first, most)
    for index in range(count):
        yield first + (last - first) * index // count, first + (last - first) * (index + 1) // count


def _fixed_wav_cuts(
    wav: WavAudio, max_chunk_ms: int, start_ms: int, last: int
) -> typing.Iterator[typing.Tuple[int, int]]:
    start = _to_sample(start_ms, wav.sample_rate)
    index = 0
    while start < last:
        # Each cut is computed from start_ms rather than from the previous cut, so rounding does not
        # accumulate.
        end = min(_to_sample(start_ms + (index + 1) * max_chunk_ms, wav.sample_rate), last)
        yield start, end
        index += 1
        start = end


def _plan(
    cuts: typing.Iterable[typing.Tuple[int, int]],
    to_ms: typing.Callable[[int], int],
    build: typing.Callable[[int, int], bytes],
) -> typing.Iterator[PlannedChunk]:
    # Cuts are in samples for WAV and frames for MP3.
    for index, (start, end) in enumerate(cuts):
        yield PlannedChunk(index, to_ms(start), to_ms(end), functools.partial(build, start, end))


def _to_sample(ms: int, sample_rate: int) -> int:
    return ms * sample_rate // 1000


def _to_ms(sample: int, sample_rate: int) -> int:
    return round_div(sample * 1000, sample_rate)
