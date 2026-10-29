import shutil
import struct
import subprocess
import typing

import numpy as np
import numpy.typing as npt
import pytest

from hume_decisions.audio_helpers._downmix import DownmixStrategy, downmix_wav
from hume_decisions.audio_helpers._wav import parse_wav, wav_chunk

_Samples = npt.NDArray[np.float64]

_FFMPEG = shutil.which("ffmpeg")
_FORMATS = [(1, False), (2, False), (3, False), (4, False), (4, True), (8, True)]


def _wav(samples: _Samples, *, sample_rate: int, sample_width: int = 2, is_float: bool = False) -> bytes:
    """A WAV file holding `samples`, shaped (frames, channels) and scaled to -1 to 1."""
    if is_float:
        data = samples.astype(f"<f{sample_width}").tobytes()
    else:
        full_scale = 1 << (sample_width * 8 - 1)
        values = np.clip(np.rint(samples * full_scale), -full_scale, full_scale - 1).astype(np.int64)
        if sample_width == 1:
            data = (values + 128).astype(np.uint8).tobytes()
        else:
            data = values.astype("<i8").view(np.uint8).reshape(*values.shape, 8)[..., :sample_width].tobytes()
    channels = samples.shape[1]
    frame_bytes = sample_width * channels
    fmt = struct.pack(
        "<HHIIHH", 3 if is_float else 1, channels, sample_rate, sample_rate * frame_bytes, frame_bytes, sample_width * 8
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


class _Audio(typing.NamedTuple):
    sample_rate: int
    sample_width: int
    is_float: bool
    # Shaped (frames, channels) and scaled to -1 to 1.
    samples: _Samples


def _read(file: bytes) -> _Audio:
    offset = 12
    fmt = b""
    while True:
        chunk_id, size = file[offset : offset + 4], struct.unpack_from("<I", file, offset + 4)[0]
        body = file[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        if chunk_id == b"data":
            break
        offset += 8 + size + size % 2
    tag, channels, sample_rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    width = bits // 8
    if tag == 3:
        samples = np.frombuffer(body, f"<f{width}").astype(np.float64)
    else:
        raw = np.frombuffer(body, np.uint8).reshape(-1, width).astype(np.int64)
        values = np.zeros(len(raw), np.int64)
        for byte in range(width):
            values |= raw[:, byte] << (8 * byte)
        if width == 1:
            values -= 128
        else:
            values[values >= 1 << (bits - 1)] -= 1 << bits
        samples = values / (1 << (bits - 1))
    return _Audio(sample_rate, width, tag == 3, samples.reshape(-1, channels))


def _downmix(file: bytes, strategy: DownmixStrategy = "mix") -> typing.Optional[bytes]:
    data = memoryview(file)
    wav = parse_wav(data)
    samples, converted = downmix_wav(data, wav, strategy)
    if converted is wav:
        return None
    return wav_chunk(samples, converted, 0, converted.sample_count)


def _noise(frames: int, channels: int) -> _Samples:
    return np.random.default_rng(frames).uniform(-0.5, 0.5, (frames, channels))


@pytest.mark.parametrize(
    "strategy, sample_rate, channels, sample_width, is_float",
    [
        ("mix", 16000, 1, 2, False),
        ("first_channel", 16000, 1, 2, False),
        ("mix", 16000, 1, 4, True),
        ("first_channel", 22050, 1, 2, False),
        ("mix", 8000, 2, 1, False),
        ("mix", 16000, 2, 2, False),
    ],
)
def test_audio_the_api_mixes_the_same_way_is_left_unchanged_when_it_would_not_get_smaller(
    strategy: DownmixStrategy, sample_rate: int, channels: int, sample_width: int, is_float: bool
) -> None:
    file = _wav(_noise(1000, channels), sample_rate=sample_rate, sample_width=sample_width, is_float=is_float)

    assert _downmix(file, strategy) is None


@pytest.mark.parametrize("sample_rate", [16000, 48000])
def test_mixing_more_than_two_channels_leaves_the_audio_unchanged(sample_rate: int) -> None:
    assert _downmix(_wav(_noise(1000, 4), sample_rate=sample_rate)) is None


def test_audio_above_16_khz_is_resampled_to_16_khz_covering_every_input_sample() -> None:
    audio = _read(typing.cast(bytes, _downmix(_wav(_noise(44107, 1), sample_rate=44100))))

    assert audio.sample_rate == 16000
    # 44107 samples at 44.1 kHz last 16002.5 samples at 16 kHz.
    assert len(audio.samples) == 16003


def test_resampling_keeps_a_constant_level() -> None:
    audio = _read(typing.cast(bytes, _downmix(_wav(np.full((4800, 1), 0.25), sample_rate=48000))))

    assert np.allclose(audio.samples, 0.25, rtol=0, atol=1e-7)


def test_resampling_keeps_the_overshoot_of_loud_audio_instead_of_clipping_it() -> None:
    # The filter overshoots the edges of a full-scale square wave.
    square = np.where(np.arange(4800) // 240 % 2 == 0, 1.0, -1.0)[:, None]

    audio = _read(typing.cast(bytes, _downmix(_wav(square, sample_rate=48000))))

    assert audio.samples.max() > 1
    assert audio.samples.min() < -1


@pytest.mark.parametrize("sample_width, is_float", _FORMATS)
def test_mixing_two_channels_adds_them_at_minus_3_db_as_32_bit_float(sample_width: int, is_float: bool) -> None:
    stereo = np.column_stack([np.full(300, 0.5), np.full(300, -0.25)])

    audio = _read(
        typing.cast(bytes, _downmix(_wav(stereo, sample_rate=48000, sample_width=sample_width, is_float=is_float)))
    )

    assert (audio.sample_rate, audio.sample_width, audio.is_float, audio.samples.shape) == (16000, 4, True, (100, 1))
    assert np.allclose(audio.samples, 0.25 * np.sqrt(0.5), rtol=0, atol=1e-7)


@pytest.mark.parametrize("sample_width, is_float", _FORMATS)
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_first_channel_at_16_khz_or_below_keeps_every_sample_of_the_first_channel(
    sample_width: int, is_float: bool, sample_rate: int
) -> None:
    file = _wav(_noise(1000, 2), sample_rate=sample_rate, sample_width=sample_width, is_float=is_float)

    audio = _read(typing.cast(bytes, _downmix(file, "first_channel")))

    expected = _read(file).samples[:, :1]
    # 64-bit floating point is halved by writing it as 32-bit, the format the API converts it to.
    if sample_width == 8:
        expected = expected.astype(np.float32)
    assert (audio.sample_rate, audio.sample_width, audio.is_float) == (sample_rate, min(sample_width, 4), is_float)
    assert np.array_equal(audio.samples, expected)


def test_first_channel_copies_the_first_channel_above_16_khz_when_that_is_smaller_than_resampling() -> None:
    file = _wav(_noise(1000, 2), sample_rate=22050, sample_width=1)

    audio = _read(typing.cast(bytes, _downmix(file, "first_channel")))

    assert (audio.sample_rate, audio.sample_width) == (22050, 1)
    assert np.array_equal(audio.samples, _read(file).samples[:, :1])


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("sample_width, is_float", _FORMATS)
@pytest.mark.parametrize(
    "sample_rate, channels, strategy, loud",
    [
        (48000, 2, "mix", False),
        (44100, 2, "mix", False),
        (44100, 1, "mix", False),
        (8000, 2, "mix", False),
        (48000, 2, "first_channel", False),
        (8000, 2, "first_channel", False),
        (16000, 2, "first_channel", False),
        (22050, 2, "first_channel", False),
        (22050, 6, "mix", False),
        (48000, 1, "mix", True),
        (48000, 2, "mix", True),
        (48000, 2, "first_channel", True),
    ],
)
def test_audio_reaches_the_model_as_it_does_through_the_apis_ffmpeg_conversion(
    sample_width: int, is_float: bool, sample_rate: int, channels: int, strategy: DownmixStrategy, loud: bool
) -> None:
    frames = sample_rate * 2 + 3
    if loud:
        # A square wave just below full scale, which the resampling filter overshoots.
        signal = np.repeat(np.where(np.arange(frames) // 240 % 2 == 0, 0.999, -0.999)[:, None], channels, axis=1)
    else:
        signal = 0.3 * np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate)[:, None] + 0.2 * _noise(
            frames, channels
        )
    file = _wav(signal, sample_rate=sample_rate, sample_width=sample_width, is_float=is_float)

    uploaded = _downmix(file, strategy) or file

    # The API converts every file with `ffmpeg -i <file> -f f32le -ac 1 -ar 16000 -`.
    pick = ["-af", "pan=mono|c0=c0"] if strategy == "first_channel" else []
    expected = _ffmpeg(file, *pick, "-ac", "1", "-ar", "16000")
    heard = _ffmpeg(uploaded, "-ac", "1", "-ar", "16000")
    # ffmpeg can leave out the last output sample.
    assert len(heard) - len(expected) in (0, 1)
    # Both are rounded to 32-bit floating point, which may differ by a few of its steps at the loudest sample.
    steps = np.spacing(np.float32(np.abs(expected).max()))
    assert np.abs(heard[: len(expected)] - expected).max() <= 5 * steps


def _ffmpeg(file: bytes, *options: str) -> _Samples:
    output = subprocess.run(
        [typing.cast(str, _FFMPEG), "-nostdin", "-v", "error", "-i", "-", *options, "-f", "f64le", "-"],
        input=file,
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(output, "<f8")
