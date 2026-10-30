import pathlib
import shutil
import subprocess
import typing

import numpy as np
import pytest
from _wav_files import Samples, read_wav, samples_wav

from hume_decisions.audio_helpers import DownmixStrategy, iter_audio_chunks

_ASSETS = pathlib.Path(__file__).parent / "assets"
_FFMPEG = shutil.which("ffmpeg")
_FORMATS = [(1, False), (2, False), (3, False), (4, False), (4, True), (8, True)]
_FFMPEG_SCENARIOS = [
    (48000, 2, "mix", False),
    (44100, 2, "mix", False),
    (44100, 1, "mix", False),
    # Rates whose ratio to 16 kHz needs more filter phases than ffmpeg uses, so it interpolates.
    (44101, 2, "mix", False),
    (22051, 2, "mix", False),
    (8000, 2, "mix", False),
    (48000, 2, "first_channel", False),
    (8000, 2, "first_channel", False),
    (16000, 2, "first_channel", False),
    (22050, 2, "first_channel", False),
    (22050, 6, "mix", False),
    (48000, 1, "mix", True),
    (48000, 2, "mix", True),
    (48000, 2, "first_channel", True),
]


def _downmix(file: bytes, strategy: DownmixStrategy = "mix") -> bytes:
    (chunk,) = iter_audio_chunks(file, max_chunk_ms=300_000, downmix_strategy=strategy)
    return chunk.data


def _noise(frames: int, channels: int) -> Samples:
    return np.random.default_rng(frames).uniform(-0.5, 0.5, (frames, channels))


def test_wav_is_mixed_down_to_16_khz_mono_floating_point_by_default_with_the_same_chunk_times() -> None:
    recording = samples_wav(_noise(48000 * 3, 2), sample_rate=48000)

    chunks = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    as_given = list(iter_audio_chunks(recording, max_chunk_ms=1000, downmix_strategy=None))
    assert [(c.start_ms, c.end_ms) for c in chunks] == [(c.start_ms, c.end_ms) for c in as_given]
    formats = {(a.sample_rate, a.channels, a.sample_width, a.is_float) for a in map(read_wav, (c.data for c in chunks))}
    assert formats == {(16000, 1, 4, True)}
    assert sum(len(c.data) for c in chunks) < sum(len(c.data) for c in as_given) / 2


def test_mp3_is_cut_as_it_is_whatever_the_downmix_strategy() -> None:
    recording = (_ASSETS / "tone_mpeg1_32k.mp3").read_bytes()

    mixed = list(iter_audio_chunks(recording, max_chunk_ms=1000))

    assert [c.data for c in mixed] == [
        c.data for c in iter_audio_chunks(recording, max_chunk_ms=1000, downmix_strategy=None)
    ]


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
    file = samples_wav(_noise(1000, channels), sample_rate=sample_rate, sample_width=sample_width, is_float=is_float)

    audio = read_wav(_downmix(file, strategy))

    assert (audio.sample_rate, audio.channels, audio.sample_width, audio.is_float) == (
        sample_rate,
        channels,
        sample_width,
        is_float,
    )
    assert np.array_equal(audio.samples, read_wav(file).samples)


@pytest.mark.parametrize("sample_rate", [16000, 48000])
def test_mixing_more_than_two_channels_leaves_the_audio_unchanged(sample_rate: int) -> None:
    file = samples_wav(_noise(1000, 4), sample_rate=sample_rate)

    audio = read_wav(_downmix(file))

    assert (audio.sample_rate, audio.channels, audio.sample_width, audio.is_float) == (sample_rate, 4, 2, False)
    assert np.array_equal(audio.samples, read_wav(file).samples)


def test_audio_above_16_khz_is_resampled_to_16_khz_covering_every_input_sample() -> None:
    audio = read_wav(_downmix(samples_wav(_noise(44107, 1), sample_rate=44100)))

    assert audio.sample_rate == 16000
    # 44107 samples at 44.1 kHz last 16002.5 samples at 16 kHz.
    assert len(audio.samples) == 16003


def test_resampling_keeps_a_constant_level() -> None:
    audio = read_wav(_downmix(samples_wav(np.full((4800, 1), 0.25), sample_rate=48000)))

    assert np.allclose(audio.samples, 0.25, rtol=0, atol=1e-7)


def test_resampling_keeps_the_overshoot_of_loud_audio_instead_of_clipping_it() -> None:
    # The filter overshoots the edges of a full-scale square wave.
    square = np.where(np.arange(4800) // 240 % 2 == 0, 1.0, -1.0)[:, None]

    audio = read_wav(_downmix(samples_wav(square, sample_rate=48000)))

    assert audio.samples.max() > 1
    assert audio.samples.min() < -1


@pytest.mark.parametrize("sample_width, is_float", _FORMATS)
def test_mixing_two_channels_adds_them_at_minus_3_db_as_32_bit_float(sample_width: int, is_float: bool) -> None:
    stereo = np.column_stack([np.full(300, 0.5), np.full(300, -0.25)])

    audio = read_wav(_downmix(samples_wav(stereo, sample_rate=48000, sample_width=sample_width, is_float=is_float)))

    assert (audio.sample_rate, audio.sample_width, audio.is_float, audio.samples.shape) == (16000, 4, True, (100, 1))
    assert np.allclose(audio.samples, 0.25 * np.sqrt(0.5), rtol=0, atol=1e-7)


@pytest.mark.parametrize("sample_width, is_float", _FORMATS)
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_first_channel_at_16_khz_or_below_keeps_every_sample_of_the_first_channel(
    sample_width: int, is_float: bool, sample_rate: int
) -> None:
    file = samples_wav(_noise(1000, 2), sample_rate=sample_rate, sample_width=sample_width, is_float=is_float)

    audio = read_wav(_downmix(file, "first_channel"))

    expected = read_wav(file).samples[:, :1]
    # 64-bit floating point is halved by writing it as 32-bit, the format the API converts it to.
    if sample_width == 8:
        expected = expected.astype(np.float32)
    assert (audio.sample_rate, audio.sample_width, audio.is_float) == (sample_rate, min(sample_width, 4), is_float)
    assert np.array_equal(audio.samples, expected)


def test_first_channel_keeps_one_channel_at_16_khz_as_floating_point() -> None:
    audio = read_wav(_downmix(samples_wav(_noise(48000, 2), sample_rate=48000), "first_channel"))

    assert (audio.sample_rate, audio.channels, audio.sample_width, audio.is_float) == (16000, 1, 4, True)


def test_first_channel_copies_the_first_channel_above_16_khz_when_that_is_smaller_than_resampling() -> None:
    file = samples_wav(_noise(1000, 2), sample_rate=22050, sample_width=1)

    audio = read_wav(_downmix(file, "first_channel"))

    assert (audio.sample_rate, audio.sample_width) == (22050, 1)
    assert np.array_equal(audio.samples, read_wav(file).samples[:, :1])


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize(
    "sample_width, is_float, sample_rate, channels, strategy, loud",
    # Each scenario in 16-bit, and each other sample format in the first scenario. How the samples
    # are read is the same in every scenario, and the tests above cover it for every format.
    [(2, False, *scenario) for scenario in _FFMPEG_SCENARIOS]
    + [(*sample_format, *_FFMPEG_SCENARIOS[0]) for sample_format in _FORMATS if sample_format != (2, False)],
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
    file = samples_wav(signal, sample_rate=sample_rate, sample_width=sample_width, is_float=is_float)

    uploaded = _downmix(file, strategy)

    # The API converts every file with `ffmpeg -i <file> -f f32le -ac 1 -ar 16000 -`.
    pick = ["-af", "pan=mono|c0=c0"] if strategy == "first_channel" else []
    expected = _ffmpeg(file, *pick, "-ac", "1", "-ar", "16000")
    heard = _ffmpeg(uploaded, "-ac", "1", "-ar", "16000")
    # ffmpeg can leave out the last output sample.
    assert len(heard) - len(expected) in (0, 1)
    # Both are rounded to 32-bit floating point, which may differ by a few of its steps at the loudest sample.
    steps = np.spacing(np.float32(np.abs(expected).max()))
    assert np.abs(heard[: len(expected)] - expected).max() <= 5 * steps


def _ffmpeg(file: bytes, *options: str) -> Samples:
    # ffmpeg's x86 code reads one more filter tap than its C code when it interpolates between filter
    # phases, so its output at such rates depends on the CPU. Clearing the CPU flags runs the C code.
    output = subprocess.run(
        [
            typing.cast(str, _FFMPEG),
            "-nostdin",
            "-v",
            "error",
            "-cpuflags",
            "0",
            "-i",
            "-",
            *options,
            "-f",
            "f64le",
            "-",
        ],
        input=file,
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(output, "<f8")
