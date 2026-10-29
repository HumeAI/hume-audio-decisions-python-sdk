import io
import os
import random
import struct
import threading
import typing
import wave

import pytest
from _wav_files import read_wav, wav_file

from hume_decisions.audio_helpers import iter_audio_chunks


def _raw_wav(
    *,
    tag: int = 0x0001,
    sample_rate: int = 16000,
    channels: int = 1,
    bits: int = 16,
    data: bytes,
    extensible: bool = False,
    data_size: typing.Optional[int] = None,
    after_data: bytes = b"",
    block_align: typing.Optional[int] = None,
    riff_size: typing.Optional[int] = None,
    riff_id: bytes = b"RIFF",
) -> bytes:
    if block_align is None:
        block_align = channels * ((bits + 7) // 8)
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, channels, sample_rate, 0, block_align, bits)
    if extensible:
        subformat = struct.pack("<H", tag) + bytes.fromhex("000000001000800000aa00389b71")
        fmt += struct.pack("<HHI", 22, bits, 0) + subformat
    # An odd-sized chunk before the data exercises the padding byte that follows it.
    info = b"INFOISFT\x05\x00\x00\x00Lavf\x00"
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"LIST" + struct.pack("<I", len(info)) + info + b"\x00"
    body += b"data" + struct.pack("<I", len(data) if data_size is None else data_size) + data + after_data
    return riff_id + struct.pack("<I", len(body) if riff_size is None else riff_size) + body


def _pcm(size: int) -> bytes:
    return random.Random(size).randbytes(size)


def _read(chunk: bytes) -> typing.Tuple[typing.Tuple[int, int, int], bytes]:
    with wave.open(io.BytesIO(chunk), "rb") as wav:
        return (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), wav.readframes(wav.getnframes())


def test_wav_is_split_into_the_fewest_equal_chunks_no_longer_than_requested() -> None:
    pcm = _pcm(2500 * 16 * 2)
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    # 40,000 samples in 3 chunks, rather than 2 full chunks and one of 500 ms.
    assert [(c.index, c.start_ms, c.end_ms) for c in chunks] == [(0, 0, 833), (1, 833, 1667), (2, 1667, 2500)]
    read = [_read(c.data) for c in chunks]
    assert all(params == (1, 2, 16000) for params, _ in read)
    assert [len(audio) for _, audio in read] == [26666, 26666, 26668]
    assert b"".join(audio for _, audio in read) == pcm


def test_recording_that_divides_evenly_gives_no_empty_final_chunk() -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(2000 * 16 * 2))

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(0, 1000), (1000, 2000)]


def test_cuts_between_samples_round_down_without_drifting() -> None:
    # At 22,050 Hz a 10 ms chunk is 220.5 samples long.
    pcm = _pcm(22050 * 2 * 2)
    recording = wav_file(sample_rate=22050, channels=2, sample_width=2, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=10, downmix_strategy=None))

    assert len(chunks) == 100
    assert [len(_read(c.data)[1]) // 4 for c in chunks[:4]] == [220, 221, 220, 221]
    assert [(c.start_ms, c.end_ms) for c in chunks] == [(i * 10, i * 10 + 10) for i in range(100)]
    assert b"".join(_read(c.data)[1] for c in chunks) == pcm


def test_final_samples_are_kept_when_the_duration_rounds_down() -> None:
    # 44,122 samples at 44,100 Hz last 1000.499 ms, which rounds down to 1000 ms.
    pcm = _pcm(44122 * 2)
    recording = wav_file(sample_rate=44100, channels=1, sample_width=2, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000, downmix_strategy=None))

    assert b"".join(_read(c.data)[1] for c in chunks) == pcm


def test_final_samples_are_kept_when_end_ms_is_the_duration_the_final_chunk_reports() -> None:
    pcm = _pcm(44122 * 2)
    recording = wav_file(sample_rate=44100, channels=1, sample_width=2, data=pcm)
    duration_ms = list(iter_audio_chunks(recording, max_chunk_ms=1000, downmix_strategy=None))[-1].end_ms

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000, end_ms=duration_ms, downmix_strategy=None))

    assert b"".join(_read(c.data)[1] for c in chunks) == pcm


@pytest.mark.parametrize("start_ms", [None, 0])
def test_recording_shorter_than_half_a_millisecond_gives_one_chunk(start_ms: typing.Optional[int]) -> None:
    pcm = _pcm(2)
    recording = wav_file(sample_rate=3000, channels=1, sample_width=2, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000, start_ms=start_ms))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(0, 0)]
    assert _read(chunks[0].data)[1] == pcm


def test_chunks_hold_only_the_fmt_and_data_chunks() -> None:
    pcm = _pcm(16 * 2 * 30)
    after = b"LIST" + struct.pack("<I", 4) + b"INFO"
    recording = _raw_wav(data=pcm, after_data=after)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=20))

    assert len(chunks) == 2
    assert all(c.data.count(b"LIST") == 0 for c in chunks)
    assert b"".join(_read(c.data)[1] for c in chunks) == pcm


def test_odd_sized_audio_is_padded_to_keep_the_file_valid() -> None:
    # 8-bit mono samples are one byte each, so 3 ms at 1 kHz is an odd number of bytes.
    pcm = _pcm(5)
    recording = wav_file(sample_rate=1000, channels=1, sample_width=1, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=3))

    assert [len(c.data) % 2 for c in chunks] == [0, 0]
    assert [_read(c.data)[1] for c in chunks] == [pcm[:2], pcm[2:]]


@pytest.mark.parametrize(
    ("tag", "bits", "extensible"),
    [(0x0001, 24, True), (0x0003, 32, False), (0x0003, 64, True)],
)
def test_chunk_of_a_non_basic_format_keeps_that_format(tag: int, bits: int, extensible: bool) -> None:
    pcm = _pcm(16 * 2 * (bits // 8) * 10)
    recording = _raw_wav(tag=tag, channels=2, bits=bits, data=pcm, extensible=extensible)

    (chunk,) = iter_audio_chunks(recording, max_chunk_ms=10, downmix_strategy=None)
    (rechunked,) = iter_audio_chunks(chunk.data, max_chunk_ms=10, downmix_strategy=None)

    assert rechunked.data == chunk.data
    assert chunk.data[20:22] == struct.pack("<H", 0xFFFE if extensible else tag)
    assert chunk.data.endswith(pcm)


@pytest.mark.parametrize("block_align", [0, 1])
def test_block_size_left_at_0_or_1_is_corrected_in_the_chunks(block_align: int) -> None:
    pcm = _pcm(16 * 2 * 10)
    recording = _raw_wav(data=pcm, block_align=block_align)

    (chunk,) = iter_audio_chunks(recording, max_chunk_ms=10)

    assert struct.unpack_from("<IH", chunk.data, 28) == (32000, 2)
    assert _read(chunk.data)[1] == pcm


def test_data_of_unknown_size_is_read_to_the_end_of_the_recording() -> None:
    pcm = _pcm(16 * 2 * 10)
    recording = _raw_wav(data=pcm, data_size=0xFFFFFFFF, riff_size=0xFFFFFFFF)

    (chunk,) = iter_audio_chunks(recording, max_chunk_ms=10)

    assert _read(chunk.data)[1] == pcm


def test_recording_that_was_never_closed_is_read_to_the_end() -> None:
    pcm = _pcm(16 * 2 * 10)
    recording = _raw_wav(data=pcm, data_size=0, riff_size=4)

    (chunk,) = iter_audio_chunks(recording, max_chunk_ms=10)

    assert _read(chunk.data)[1] == pcm


def test_recording_cut_off_partway_through_a_sample_is_read_to_the_last_whole_sample() -> None:
    pcm = _pcm(16 * 2 * 10)
    recording = _raw_wav(data=pcm + b"\x01", data_size=len(pcm) + 100)

    (chunk,) = iter_audio_chunks(recording, max_chunk_ms=10)

    assert _read(chunk.data)[1] == pcm


@pytest.mark.parametrize(
    ("recording", "message"),
    [
        (_raw_wav(data=_pcm(32), riff_id=b"RF64"), "RF64"),
        (_raw_wav(tag=0x0002, bits=4, data=_pcm(32)), "compressed"),
        (_raw_wav(tag=0x0003, bits=24, data=_pcm(32)), "24-bit floating point"),
        (_raw_wav(sample_rate=999, data=_pcm(32)), "below the minimum"),
        (_raw_wav(data=_pcm(32), block_align=4), "inconsistent"),
        (b"RIFF\x04\x00\x00\x00WAVE", "no data chunk"),
    ],
)
def test_unsupported_wav_is_rejected_with_a_conversion_command(recording: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message) as error:
        iter_audio_chunks(recording, max_chunk_ms=1000)

    assert "ffmpeg -i input.wav" in str(error.value)


def test_audio_that_is_neither_mp3_nor_wav_is_rejected() -> None:
    with pytest.raises(ValueError, match="Only MP3 and WAV files"):
        iter_audio_chunks(b"OggS" + _pcm(100), max_chunk_ms=1000)


def test_empty_recording_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        iter_audio_chunks(wav_file(sample_rate=16000, channels=1, sample_width=2, data=b""), max_chunk_ms=1000)


@pytest.mark.parametrize("max_chunk_ms", [0, -1, 300_001])
def test_chunk_duration_out_of_range_is_rejected(max_chunk_ms: int) -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(32))

    with pytest.raises(ValueError, match="max_chunk_ms must be between 1 and 300000"):
        iter_audio_chunks(recording, max_chunk_ms=max_chunk_ms)


def test_fixed_chunk_too_large_for_the_api_is_rejected_with_the_longest_chunk_that_fits() -> None:
    # 32-bit float stereo at 48 kHz is 384,000 bytes a second, so 70 seconds is 26,880,000 bytes.
    recording = _raw_wav(tag=0x0003, sample_rate=48000, channels=2, bits=32, data=bytes(70 * 384_000))

    with pytest.raises(ValueError, match="max_chunk_ms must be at most 62499 for this WAV file"):
        iter_audio_chunks(recording, max_chunk_ms=300_000, split="fixed", downmix_strategy=None)
    chunk = next(iter_audio_chunks(recording, max_chunk_ms=62499, split="fixed", downmix_strategy=None))

    assert len(chunk.data) <= 24_000_000


def test_even_chunks_too_large_for_the_api_are_cut_into_the_fewest_that_fit() -> None:
    # 32-bit float stereo at 48 kHz is 384,000 bytes a second, so 130 seconds is 49,920,000 bytes,
    # which takes 3 chunks of at most 24,000,000 bytes.
    recording = _raw_wav(tag=0x0003, sample_rate=48000, channels=2, bits=32, data=bytes(130 * 384_000))

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=300_000, downmix_strategy=None))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(0, 43333), (43333, 86667), (86667, 130_000)]
    assert all(len(c.data) <= 24_000_000 for c in chunks)


def test_chunk_duration_longer_than_the_api_accepts_is_allowed_when_the_range_is_shorter() -> None:
    recording = _raw_wav(tag=0x0003, sample_rate=48000, channels=2, bits=32, data=bytes(70 * 384_000))

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=300_000, start_ms=10_000))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(10_000, 70_000)]


def test_chunk_duration_that_is_not_an_integer_is_rejected() -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(32))

    with pytest.raises(TypeError, match="max_chunk_ms must be an integer, got 1.5"):
        iter_audio_chunks(recording, max_chunk_ms=typing.cast(typing.Any, 1.5))


@pytest.mark.parametrize(
    ("start_ms", "end_ms", "times"),
    [
        (None, None, [(0, 1000), (1000, 2000), (2000, 2500)]),
        (500, 2500, [(500, 1500), (1500, 2500)]),
    ],
)
def test_fixed_wav_chunks_are_cut_on_a_grid_from_the_start_and_the_last_holds_what_remains(
    start_ms: typing.Optional[int], end_ms: typing.Optional[int], times: typing.List[typing.Tuple[int, int]]
) -> None:
    pcm = _pcm(2500 * 16 * 2)
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000, start_ms=start_ms, end_ms=end_ms, split="fixed"))

    assert [(c.start_ms, c.end_ms) for c in chunks] == times
    assert b"".join(_read(c.data)[1] for c in chunks) == pcm[times[0][0] * 32 : times[-1][1] * 32]


def test_split_other_than_even_or_fixed_is_rejected() -> None:
    with pytest.raises(ValueError, match='split must be "even" or "fixed", got \'equal\''):
        iter_audio_chunks(
            wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(32000)),
            max_chunk_ms=1000,
            split=typing.cast(typing.Any, "equal"),
        )


def test_downmix_strategy_other_than_mix_first_channel_or_none_is_rejected() -> None:
    with pytest.raises(ValueError, match='downmix_strategy must be "mix", "first_channel" or None, got \'average\''):
        iter_audio_chunks(
            wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(32000)),
            max_chunk_ms=1000,
            downmix_strategy=typing.cast(typing.Any, "average"),
        )


def test_range_is_split_into_chunks_timed_from_the_start_of_the_recording() -> None:
    pcm = _pcm(5000 * 16 * 2)
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000, start_ms=1500, end_ms=4000))

    assert [(c.index, c.start_ms, c.end_ms) for c in chunks] == [(0, 1500, 2333), (1, 2333, 3167), (2, 3167, 4000)]
    assert b"".join(_read(c.data)[1] for c in chunks) == pcm[1500 * 32 : 4000 * 32]


def test_range_cuts_between_samples_round_down_from_the_start_of_the_range() -> None:
    pcm = _pcm(22050 * 2)
    recording = wav_file(sample_rate=22050, channels=1, sample_width=2, data=pcm)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=10, start_ms=5, end_ms=45, downmix_strategy=None))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(5, 15), (15, 25), (25, 35), (35, 45)]
    # 5 ms is 110.25 samples and 45 ms is 992.25 samples.
    assert b"".join(_read(c.data)[1] for c in chunks) == pcm[110 * 2 : 992 * 2]


def test_range_may_end_at_the_duration_the_final_chunk_reports() -> None:
    # 1004 samples at 22,050 Hz last 45.533 ms, which the final chunk reports as 46 ms.
    pcm = _pcm(1004 * 2)
    recording = wav_file(sample_rate=22050, channels=1, sample_width=2, data=pcm)
    duration_ms = list(iter_audio_chunks(recording, max_chunk_ms=1000, downmix_strategy=None))[-1].end_ms

    chunks = list(
        iter_audio_chunks(
            recording, max_chunk_ms=1000, start_ms=duration_ms - 1, end_ms=duration_ms, downmix_strategy=None
        )
    )

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(45, 46)]
    assert _read(chunks[0].data)[1] == pcm[992 * 2 :]


@pytest.mark.parametrize(
    ("start_ms", "end_ms", "message"),
    [
        (-1, None, "start_ms must not be negative"),
        (1000, 1000, "end_ms must be greater than start_ms"),
        (None, 0, "end_ms must be greater than start_ms"),
        (2000, None, "start_ms must be less than the recording's duration of 2000 ms"),
        (None, 2001, "end_ms must not exceed the recording's duration of 2000 ms"),
    ],
)
def test_range_outside_the_recording_is_rejected(
    start_ms: typing.Optional[int], end_ms: typing.Optional[int], message: str
) -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(2000 * 16 * 2))

    with pytest.raises(ValueError, match=message):
        iter_audio_chunks(recording, max_chunk_ms=1000, start_ms=start_ms, end_ms=end_ms)


@pytest.mark.parametrize("name", ["start_ms", "end_ms"])
def test_range_bound_that_is_not_an_integer_is_rejected(name: str) -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(32))

    with pytest.raises(TypeError, match=f"{name} must be an integer, got 0.5"):
        iter_audio_chunks(recording, max_chunk_ms=1000, **{name: typing.cast(typing.Any, 0.5)})


@pytest.mark.parametrize(
    "wrap",
    [
        lambda recording: ("call.wav", recording),
        lambda recording: ("call.wav", recording, "audio/wav"),
        lambda recording: ("call.wav", io.BytesIO(recording), "audio/wav", {"X-Test": "1"}),
        lambda recording: (None, io.BytesIO(recording)),
    ],
)
def test_file_given_as_a_tuple_gives_the_same_chunks_as_its_bytes(
    wrap: typing.Callable[[bytes], typing.Any],
) -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(2500 * 16 * 2))

    chunks = list(iter_audio_chunks(wrap(recording), max_chunk_ms=1000))

    assert chunks == list(iter_audio_chunks(recording, max_chunk_ms=1000))


def test_file_object_is_read_from_its_start_and_left_open() -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(2500 * 16 * 2))
    file = io.BytesIO(recording)
    file.seek(100)

    chunks = list(iter_audio_chunks(file, max_chunk_ms=1000))

    assert chunks == list(iter_audio_chunks(recording, max_chunk_ms=1000))
    assert not file.closed


def test_file_object_that_cannot_seek_such_as_a_pipe_is_read() -> None:
    recording = wav_file(sample_rate=16000, channels=1, sample_width=2, data=_pcm(100 * 16 * 2))
    read_fd, write_fd = os.pipe()

    def write() -> None:
        with open(write_fd, "wb") as pipe:
            pipe.write(recording)

    writer = threading.Thread(target=write)
    writer.start()
    with open(read_fd, "rb") as pipe:
        chunks = list(iter_audio_chunks(pipe, max_chunk_ms=50))
    writer.join()

    assert chunks == list(iter_audio_chunks(recording, max_chunk_ms=50))


def test_path_given_as_a_str_is_rejected_with_an_explanation() -> None:
    with pytest.raises(ValueError, match="given as a str, which is read as the file's contents rather than a path"):
        iter_audio_chunks("call.wav", max_chunk_ms=1000)


def test_unsupported_bytes_are_rejected_without_mentioning_str() -> None:
    with pytest.raises(ValueError) as error:
        iter_audio_chunks(b"call.wav", max_chunk_ms=1000)

    assert "str" not in str(error.value)


def test_wav_is_mixed_down_to_16_khz_mono_floating_point_by_default_with_the_same_chunk_times() -> None:
    recording = wav_file(sample_rate=48000, channels=2, sample_width=2, data=_pcm(48000 * 4 * 3))

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    as_given = list(iter_audio_chunks(recording, max_chunk_ms=1000, downmix_strategy=None))
    assert [(c.start_ms, c.end_ms) for c in chunks] == [(c.start_ms, c.end_ms) for c in as_given]
    formats = {(a.sample_rate, a.channels, a.sample_width, a.is_float) for a in map(read_wav, (c.data for c in chunks))}
    assert formats == {(16000, 1, 4, True)}
    assert sum(len(c.data) for c in chunks) < sum(len(c.data) for c in as_given) / 2


def test_first_channel_keeps_one_channel_at_16_khz_as_floating_point() -> None:
    recording = wav_file(sample_rate=48000, channels=2, sample_width=2, data=_pcm(48000 * 4))

    (chunk,) = iter_audio_chunks(recording, max_chunk_ms=1000, downmix_strategy="first_channel")

    audio = read_wav(chunk.data)
    assert (audio.sample_rate, audio.channels, audio.sample_width, audio.is_float) == (16000, 1, 4, True)
