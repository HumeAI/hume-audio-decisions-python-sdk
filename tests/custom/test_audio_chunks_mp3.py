import dataclasses
import pathlib
import struct
import typing

import pytest

from hume_decisions.audio_helpers import ChunkSplit, iter_audio_chunks

_MPEG1 = 3
_MPEG2 = 2
_MPEG25 = 0

_ASSETS = pathlib.Path(__file__).parent / "assets"


def _frame(
    number: int,
    *,
    version: int = _MPEG1,
    bitrate_index: int = 9,
    sample_rate_index: int = 0,
    padding: int = 0,
    layer: int = 1,
    mono: bool = False,
    length: int = 417,
    reservoir: int = 0,
    checksum: bool = False,
) -> bytes:
    """
    Builds an MP3 frame of `length` bytes whose body identifies it by `number`, storing `reservoir`
    bytes of its audio data in the frames before it.
    """
    header = bytes(
        [
            0xFF,
            0xE0 | version << 3 | layer << 1 | (0 if checksum else 1),
            bitrate_index << 4 | sample_rate_index << 2 | padding << 1,
            (3 if mono else 0) << 6,
        ]
    )
    side_info = 6 if checksum else 4
    frame = bytearray(header + (struct.pack(">I", number) * (length // 4))[: length - 4])
    if version == _MPEG1:
        frame[side_info : side_info + 2] = bytes([reservoir >> 1, (reservoir & 1) << 7])
    else:
        frame[side_info] = reservoir
    return bytes(frame)


def _frames(count: int, **kwargs: typing.Any) -> typing.List[bytes]:
    return [_frame(number, **kwargs) for number in range(count)]


def _id3v2(size: int, *, footer: bool = False) -> bytes:
    syncsafe = bytes([size >> 21 & 0x7F, size >> 14 & 0x7F, size >> 7 & 0x7F, size & 0x7F])
    tag = b"ID3\x04\x00" + bytes([0x10 if footer else 0]) + syncsafe + b"\x00" * size
    return tag + (b"3DI\x04\x00\x10" + syncsafe if footer else b"")


def _vbr_frame(marker: bytes, offset: int, *, mono: bool = False) -> bytes:
    frame = bytearray(_frame(0, mono=mono))
    frame[offset : offset + 4] = marker
    return bytes(frame)


def _lame_frame(*, delay: int, padding: int, frames: int) -> bytes:
    """Builds an Info frame stating the frame count and, in a LAME tag, the samples the encoder added."""
    frame = bytearray(_vbr_frame(b"Info", 36))
    # Only the frame count is present, so the LAME tag follows it.
    frame[40:48] = struct.pack(">II", 1, frames)
    frame[48:52] = b"LAME"
    frame[69:72] = bytes([delay >> 4, (delay & 0x0F) << 4 | padding >> 8, padding & 0xFF])
    return bytes(frame)


def test_mp3_is_split_between_frames_into_the_fewest_even_chunks_no_longer_than_requested() -> None:
    # MPEG-1 frames at 44.1 kHz hold 1152 samples, about 26.12 ms, so 38 fit in a second. Each chunk
    # after the first repeats the frame before its first new frame, whose output is blended into it.
    # Chunks of 38 frames would need 3 chunks, the last of 26 frames, and 34 is the fewest that still
    # need only 3.
    frames = _frames(100)

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=1000))

    assert [(c.index, c.start_ms, c.end_ms) for c in chunks] == [(0, 0, 888), (1, 862, 1750), (2, 1724, 2612)]
    assert [c.data for c in chunks] == [b"".join(frames[:34]), b"".join(frames[33:67]), b"".join(frames[66:])]


def test_fixed_mp3_chunks_hold_as_many_frames_as_fit_and_the_last_holds_what_remains() -> None:
    frames = _frames(100)

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=1000, split="fixed"))

    assert [(c.index, c.start_ms, c.end_ms) for c in chunks] == [(0, 0, 993), (1, 967, 1959), (2, 1933, 2612)]
    assert [c.data for c in chunks] == [b"".join(frames[:38]), b"".join(frames[37:75]), b"".join(frames[74:])]


def test_mp3_chunks_of_the_maximum_duration_stay_within_five_minutes() -> None:
    # 32 kbit/s frames at 44.1 kHz are 104 bytes long. 11,484 frames fit in 5 minutes, and the
    # second chunk repeats one frame, so 22,967 frames fill exactly 2 chunks.
    frames = _frames(22967, bitrate_index=1, length=104)

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=300_000))

    assert [c.data for c in chunks] == [b"".join(frames[:11484]), b"".join(frames[11483:])]
    assert chunks[0].end_ms == 299_990


@pytest.mark.parametrize(
    ("version", "sample_rate_index", "length", "max_chunk_ms", "times"),
    [
        # MPEG-2 frames at 22,050 Hz hold 576 samples, about 26.12 ms.
        (_MPEG2, 0, 208, 262, [(0, 209), (183, 392), (366, 522)]),
        # MPEG-2.5 frames at 8 kHz hold 576 samples, 72 ms.
        (_MPEG25, 2, 576, 720, [(0, 576), (504, 1080), (1008, 1440)]),
    ],
)
def test_mpeg_2_and_2_5_frames_are_timed_at_576_samples(
    version: int, sample_rate_index: int, length: int, max_chunk_ms: int, times: typing.List[typing.Tuple[int, int]]
) -> None:
    # Bitrate index 8 is 64 kbit/s for MPEG-2 and 2.5.
    frames = _frames(20, version=version, bitrate_index=8, sample_rate_index=sample_rate_index, length=length)

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=max_chunk_ms))

    assert [(c.start_ms, c.end_ms) for c in chunks] == times
    assert [c.data for c in chunks] == [b"".join(frames[:8]), b"".join(frames[7:15]), b"".join(frames[14:])]


def test_padded_frames_are_one_byte_longer() -> None:
    frames = [_frame(n, padding=n % 2, length=417 + n % 2) for n in range(10)]

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=60))

    assert [c.data for c in chunks] == [b"".join(frames[:2])] + [b"".join(frames[n - 1 : n + 1]) for n in range(2, 10)]


@pytest.mark.parametrize("footer", [False, True])
def test_tags_are_left_out_of_the_chunks(footer: bool) -> None:
    frames = _frames(10)
    id3v1 = b"TAG" + b"\x00" * 125
    recording = _id3v2(300, footer=footer) + _id3v2(20) + b"".join(frames) + b"APETAGEX" + b"\x00" * 24 + id3v1

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    assert [c.data for c in chunks] == [b"".join(frames)]


@pytest.mark.parametrize(
    "header",
    [
        _vbr_frame(b"Xing", 36),
        _vbr_frame(b"Info", 36),
        _vbr_frame(b"VBRI", 36),
        # The side information before a Xing header is shorter in a mono frame.
        _vbr_frame(b"Xing", 21, mono=True),
    ],
)
def test_header_frame_stating_the_whole_files_length_is_left_out(header: bytes) -> None:
    frames = _frames(10)

    chunks = list(iter_audio_chunks(header + b"".join(frames), max_chunk_ms=1000))

    assert [c.data for c in chunks] == [b"".join(frames)]
    assert chunks[0].end_ms == 261


def test_mp3_times_are_measured_from_where_players_start_when_a_lame_tag_states_the_samples_added() -> None:
    # Players leave out 576 + 529 samples at the start and 1000 - 529 at the end, so the recording
    # starts 25.06 ms into the first frame and ends 225.48 ms after that.
    frames = _frames(10)
    recording = _lame_frame(delay=576, padding=1000, frames=10) + b"".join(frames)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(-25, 225)]
    assert chunks[0].data == b"".join(frames)


def test_mp3_range_from_0_to_the_duration_keeps_the_frames_holding_the_start_and_end() -> None:
    frames = _frames(10)
    recording = _lame_frame(delay=576, padding=1000, frames=10) + b"".join(frames)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000, start_ms=0, end_ms=225))

    assert [c.data for c in chunks] == [b"".join(frames)]

    with pytest.raises(ValueError, match="end_ms must not exceed the recording's duration of 225 ms"):
        iter_audio_chunks(recording, max_chunk_ms=1000, end_ms=226)


def test_mp3_range_with_a_lame_tag_starting_where_a_chunk_reported_it_ends_continues_from_that_chunk() -> None:
    recording = _lame_frame(delay=576, padding=1000, frames=100) + b"".join(_frames(100))
    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    rest = list(iter_audio_chunks(recording, max_chunk_ms=1000, start_ms=chunks[0].end_ms))

    assert rest == [dataclasses.replace(c, index=c.index - 1) for c in chunks[1:]]


def test_mp3_cut_off_before_the_samples_added_at_the_end_keeps_its_final_frame_whole() -> None:
    # The header counts 12 frames, so the samples added at the end were in the 2 that are missing.
    recording = _lame_frame(delay=576, padding=1000, frames=12) + b"".join(_frames(10))

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(-25, 236)]


def test_mp3_cut_off_partway_through_a_frame_is_read_to_the_last_whole_frame() -> None:
    frames = _frames(10)

    chunks = list(iter_audio_chunks(b"".join(frames) + frames[0][:200], max_chunk_ms=1000))

    assert [c.data for c in chunks] == [b"".join(frames)]


@pytest.mark.parametrize("tag", [b"", b"TAG" + b"\x00" * 125])
def test_bytes_after_the_last_frame_that_are_not_a_frame_are_left_out(tag: bytes) -> None:
    frames = _frames(10)

    chunks = list(iter_audio_chunks(b"".join(frames) + b"\x00" * 64 + b"\xff" + b"\x00" * 63 + tag, max_chunk_ms=1000))

    assert [c.data for c in chunks] == [b"".join(frames)]


def test_mp3_range_moves_to_the_nearest_frame_boundary() -> None:
    frames = _frames(100)

    # 990 ms is 37.9 frames in and 1990 ms is 76.2 frames in.
    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=2000, start_ms=990, end_ms=1990))

    assert [(c.start_ms, c.end_ms) for c in chunks] == [(967, 1985)]
    assert chunks[0].data == b"".join(frames[37:76])


def test_mp3_range_starting_where_a_chunk_reported_it_ends_continues_from_that_chunk() -> None:
    recording = b"".join(_frames(100))
    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    # The first chunk's end, 992.65 ms, is reported as 993 ms, which is past the frame boundary.
    rest = list(iter_audio_chunks(recording, max_chunk_ms=1000, start_ms=chunks[0].end_ms))

    assert rest == [dataclasses.replace(c, index=c.index - 1) for c in chunks[1:]]


def test_damaged_mp3_is_rejected_before_any_chunk_is_produced() -> None:
    frames = _frames(100)
    frames[90] = b"\x00" * len(frames[90])

    with pytest.raises(ValueError, match="no MPEG audio frame at byte 37530"):
        iter_audio_chunks(b"".join(frames), max_chunk_ms=1000)


@pytest.mark.parametrize(
    ("recording", "message"),
    [
        (b"".join(_frames(10, bitrate_index=0)), "free format"),
        (b"".join(_frames(10, layer=2)), "layer I and II"),
        (b"".join(_frames(5) + _frames(5, sample_rate_index=1, length=384)), "sample rate changes at byte 2085"),
        (_frame(0)[:2] + b"\xf0\x00" + b"\x00" * 413, "invalid frame header at byte 0"),
        (_id3v2(10), "no MPEG audio frame"),
    ],
)
def test_unsupported_mp3_is_rejected_with_a_conversion_command(recording: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message) as error:
        iter_audio_chunks(recording, max_chunk_ms=1000)

    assert "ffmpeg -i input.mp3" in str(error.value)


def test_mp3_holding_only_a_header_frame_is_rejected_as_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        iter_audio_chunks(_vbr_frame(b"Info", 36), max_chunk_ms=1000)


def test_chunk_duration_shorter_than_one_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_chunk_ms must be at least 27 ms"):
        iter_audio_chunks(b"".join(_frames(10)), max_chunk_ms=26)


def test_mp3_range_shorter_than_one_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="start_ms and end_ms must be at least 27 ms apart"):
        iter_audio_chunks(b"".join(_frames(10)), max_chunk_ms=1000, start_ms=100, end_ms=110)


def test_mp3_range_past_the_end_is_rejected() -> None:
    with pytest.raises(ValueError, match="end_ms must not exceed the recording's duration of 261 ms"):
        iter_audio_chunks(b"".join(_frames(10)), max_chunk_ms=1000, end_ms=262)


def test_mp3_chunk_starts_with_the_frames_its_first_new_frame_needs() -> None:
    # Each stereo MPEG-1 frame of 417 bytes holds 381 bytes of audio data after its header and side
    # information, so 300 bytes stored earlier are all in the frame before. A decoder may drop the
    # data of the first frame it reads, so another frame comes before that one. The frame before
    # the first new frame needs the same two frames before it.
    frames = [_frame(n, reservoir=0 if n == 0 else 300) for n in range(100)]

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=1000))

    assert [c.data for c in chunks] == [b"".join(frames[:36]), b"".join(frames[33:69]), b"".join(frames[66:])]
    assert [(c.start_ms, c.end_ms) for c in chunks] == [(0, 940), (862, 1802), (1724, 2612)]


@pytest.mark.parametrize(
    ("frame_options", "frame", "reservoir", "needed"),
    [
        # 381 bytes of audio data after a 4-byte header and 32 bytes of side information. One more
        # frame comes before the frames holding the data, because a decoder may drop the data of the
        # first frame it reads.
        ({}, 10, 381, 2),
        ({}, 10, 382, 3),
        # Mono frames have 17 bytes of side information, leaving 396.
        ({"mono": True}, 10, 396, 2),
        ({"mono": True}, 10, 397, 3),
        # A checksum takes 2 more bytes, leaving 379.
        ({"checksum": True}, 10, 379, 2),
        ({"checksum": True}, 10, 380, 3),
        # MPEG-2 stores the distance in 8 bits after 17 bytes of side information. 64 kbit/s frames
        # at 22,050 Hz are 208 bytes long, leaving 187.
        ({"version": _MPEG2, "bitrate_index": 8, "length": 208}, 10, 187, 2),
        ({"version": _MPEG2, "bitrate_index": 8, "length": 208}, 10, 188, 3),
        # The frame before the first new frame is always needed, along with the frames it needs.
        ({}, 10, 0, 1),
        ({}, 9, 381, 3),
        ({}, 9, 382, 4),
    ],
)
def test_frames_needed_count_only_the_audio_data_of_earlier_frames(
    frame_options: typing.Dict[str, typing.Any], frame: int, reservoir: int, needed: int
) -> None:
    frames = [_frame(n, reservoir=reservoir if n == frame else 0, **frame_options) for n in range(20)]
    recording = b"".join(frames)
    end_ms = list(iter_audio_chunks(recording, max_chunk_ms=300_000))[0].end_ms

    (chunk,) = iter_audio_chunks(recording, max_chunk_ms=300_000, start_ms=end_ms // 2)

    assert chunk.data == b"".join(frames[10 - needed :])


def test_mp3_range_starts_with_the_frames_its_first_new_frame_needs() -> None:
    # 32 kbit/s stereo frames at 44.1 kHz are 104 bytes long and hold 68 bytes of audio data, so
    # data stored 511 bytes back, the most MPEG-1 allows, spans 8 frames. With the frame a decoder
    # may drop, the frame before the first new frame needs 9 frames before it.
    frames = [_frame(n, bitrate_index=1, length=104, reservoir=511 if n >= 8 else 0) for n in range(100)]

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=1000, start_ms=993))

    assert chunks[0].data == b"".join(frames[28:59])
    assert chunks[0].start_ms == 731
    assert chunks[1].data == b"".join(frames[49:80])


def test_chunk_duration_too_short_for_the_frames_needed_is_rejected() -> None:
    frames = [_frame(n, bitrate_index=1, length=104, reservoir=511 if n >= 8 else 0) for n in range(100)]

    # 11 frames last 287.3 ms, enough for 10 needed frames and one new frame.
    with pytest.raises(ValueError, match="max_chunk_ms must be at least 288 ms .* up to 10 frames before them"):
        iter_audio_chunks(b"".join(frames), max_chunk_ms=287)


def test_chunk_duration_just_long_enough_for_the_frames_needed_advances_a_frame_at_a_time() -> None:
    frames = [_frame(n, bitrate_index=1, length=104, reservoir=511 if n >= 8 else 0) for n in range(100)]

    chunks = list(iter_audio_chunks(b"".join(frames), max_chunk_ms=288))

    assert [c.data for c in chunks] == [b"".join(frames[:11])] + [
        b"".join(frames[n - 10 : n + 1]) for n in range(11, 100)
    ]


@pytest.mark.parametrize(
    ("name", "split", "times"),
    [
        pytest.param(
            "tone_mpeg1_32k.mp3",
            "even",
            [(-35, 829), (685, 1549), (1333, 2197), (1981, 2845), (2629, 3493), (3277, 4000)],
            id="MPEG-1 even",
        ),
        pytest.param(
            "tone_mpeg1_32k.mp3",
            "fixed",
            [(-35, 937), (757, 1729), (1513, 2485), (2269, 3241), (3025, 3997), (3781, 4000)],
            id="MPEG-1 fixed",
        ),
        pytest.param(
            "tone_mpeg25_8k.mp3",
            "even",
            [(-138, 798), (510, 1446), (1158, 2094), (1734, 2670), (2310, 3246), (2814, 3750), (3318, 4000)],
            id="MPEG-2.5 even",
        ),
        pytest.param(
            "tone_mpeg25_8k.mp3",
            "fixed",
            [(-138, 798), (510, 1446), (1158, 2094), (1734, 2670), (2310, 3246), (2814, 3750), (3318, 4000)],
            id="MPEG-2.5 fixed",
        ),
    ],
)
def test_encoder_written_mp3_is_cut_at_the_times_confirmed_by_decoding_its_chunks(
    name: str, split: ChunkSplit, times: typing.List[typing.Tuple[int, int]]
) -> None:
    # The files were encoded with ffmpeg's libmp3lame from a tone mixed with noise, at a bitrate low
    # enough that almost every frame stores audio data in earlier frames. Decoding each of these
    # chunks with ffmpeg gives exactly the samples of decoding the whole file, over the chunk's new
    # frames. That was checked by hand, because CI has no decoder, so the times are pinned here.
    # With fewer frames repeated, the times change and the decoded audio does not match.
    # Times are measured from where ffmpeg and mpg123 start playing the whole file, which both play
    # for 4000 ms, so a chunk's start plus a time within the decoded chunk gives that sample.
    recording = (_ASSETS / name).read_bytes()

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000, split=split))

    assert [(c.start_ms, c.end_ms) for c in chunks] == times
    first_frame = recording.index(chunks[0].data)
    assert b"Info" in recording[:first_frame]
    # The file ends with a 128-byte ID3v1 tag.
    assert recording.index(chunks[-1].data) + len(chunks[-1].data) == len(recording) - 128
