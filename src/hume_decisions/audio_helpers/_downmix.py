# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import math
import typing

import numpy as np
import numpy.typing as npt
from ._integers import ceil_div
from ._wav import WavAudio, samples_wav

DownmixStrategy = typing.Literal["mix", "first_channel"]

# The model hears audio at this rate, in one channel.
MODEL_SAMPLE_RATE = 16000

# The settings of ffmpeg's default resampler, which the API uses before the model hears the audio, so
# audio resampled here reaches the model as it would have without this step.
_FILTER_SIZE = 32
_CUTOFF = 0.97
_KAISER_BETA = 9.0

# Outputs computed together. Each needs a filter's length of input, so this bounds the memory used.
_BLOCK = 8192

_Samples = npt.NDArray[np.float64]


def check_downmix_strategy(strategy: typing.Optional[DownmixStrategy]) -> None:
    if strategy not in (*typing.get_args(DownmixStrategy), None):
        raise ValueError(f'downmix_strategy must be "mix", "first_channel" or None, got {strategy!r}.')


def downmix_wav(
    data: memoryview, wav: WavAudio, strategy: typing.Optional[DownmixStrategy]
) -> typing.Tuple[memoryview, WavAudio]:
    """
    The samples of `wav` mixed down by `strategy` into one channel, with a description of them, or
    `data` and `wav` themselves when the API would mix the file the same way and converting it would
    not make it smaller. Resampled audio is at most MODEL_SAMPLE_RATE, in 32-bit floating point.
    """
    # ffmpeg weighs more than two channels by the speaker positions in the header's channel mask and
    # leaves out the LFE channel, so they are sent as they are for the API to mix.
    if strategy is None or (strategy == "mix" and wav.channels > 2):
        return data, wav
    sample_rate = min(wav.sample_rate, MODEL_SAMPLE_RATE)
    count = ceil_div(wav.sample_count * sample_rate, wav.sample_rate)
    # The API converts every sample to 32-bit floating point, so writing that loses nothing and cannot
    # clip, but it is larger than some recordings that the API would convert just the same.
    converted = samples_wav(1, sample_rate, 4, True, count)
    unchanged_bytes = wav.sample_count * wav.frame_bytes
    copy_bytes = wav.sample_count * wav.sample_width
    converted_bytes = count * converted.frame_bytes
    # The API would mix every channel, so the first is taken out even when that is not smaller.
    picks_channel = strategy == "first_channel" and wav.channels > 1
    if not picks_channel and converted_bytes >= unchanged_bytes:
        return data, wav
    frames = _frames(data, wav)
    # A copy of the channel's samples is exact, so it is preferred whenever it is no larger.
    if picks_channel and copy_bytes <= converted_bytes:
        picked = samples_wav(1, wav.sample_rate, wav.sample_width, wav.is_float, wav.sample_count)
        return memoryview(frames[:, 0].tobytes()), picked
    mixes = strategy == "mix" and wav.channels == 2

    def read(start: int, stop: int) -> _Samples:
        if mixes:
            # ffmpeg adds the two channels at -3 dB each when it writes floating point, as the API does.
            return _decode(frames[start:stop], wav).sum(axis=1, keepdims=True) * math.sqrt(0.5)
        return _decode(frames[start:stop, :1], wav)

    blocks: typing.Iterator[_Samples]
    if sample_rate == wav.sample_rate:
        blocks = (read(start, min(start + _BLOCK, count)) for start in range(0, count, _BLOCK))
    else:
        blocks = _resample(read, wav.sample_count, wav.sample_rate, sample_rate, count)
    out = bytearray(converted_bytes)
    offset = 0
    for samples in blocks:
        encoded = _encode(samples, wav)
        out[offset : offset + len(encoded)] = encoded
        offset += len(encoded)
    return memoryview(out), converted


def _resample(
    read: typing.Callable[[int, int], _Samples], length: int, from_rate: int, to_rate: int, count: int
) -> typing.Iterator[_Samples]:
    divisor = math.gcd(from_rate, to_rate)
    up, down = to_rate // divisor, from_rate // divisor
    factor = to_rate * _CUTOFF / from_rate
    # ffmpeg rounds the filter length up to an even number.
    taps = math.ceil(_FILTER_SIZE / factor)
    taps += taps % 2
    center = (taps - 1) // 2
    bank = _filter_bank(up, factor, taps)
    for start in range(0, count, _BLOCK):
        outputs = np.arange(start, min(start + _BLOCK, count))
        # Output k lies at input position k * down / up, between input `first` and the next one, at
        # the given phase of the filter bank.
        first = outputs * down // up
        phase = outputs * down % up
        positions = np.arange(first[0] - center, first[-1] - center + taps)
        # ffmpeg reflects the audio at both ends, so the first and last outputs match its own.
        positions = np.abs(positions)
        positions = np.clip(np.where(positions >= length, 2 * length - 1 - positions, positions), 0, length - 1)
        low = int(positions.min())
        segment = read(low, int(positions.max()) + 1)[positions - low]
        windows = np.lib.stride_tricks.sliding_window_view(segment, taps, axis=0)
        yield np.einsum("kct,kt->kc", windows[first - first[0]], bank[phase])


def _filter_bank(phases: int, factor: float, taps: int) -> _Samples:
    """A Kaiser-windowed sinc low-pass filter for each phase, built the way swresample builds its own."""
    x = np.pi * ((np.arange(taps) - (taps - 1) // 2)[None, :] - np.arange(phases)[:, None] / phases) * factor
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(x == 0, 1.0, np.sin(x) / x)
    window = np.i0(_KAISER_BETA * np.sqrt(np.maximum(1 - (2 * x / (factor * taps * np.pi)) ** 2, 0)))
    bank: _Samples = sinc * window
    # Every phase is scaled by the first phase's sum, as swresample does, so a constant level is kept.
    return bank / bank[0].sum()


def _frames(data: memoryview, wav: WavAudio) -> npt.NDArray[typing.Any]:
    raw = np.frombuffer(data, np.uint8, wav.sample_count * wav.frame_bytes, wav.data_start)
    if wav.sample_width == 3:
        return raw.reshape(wav.sample_count, wav.channels, 3)
    kind = "f" if wav.is_float else "u" if wav.sample_width == 1 else "i"
    return raw.view(f"<{kind}{wav.sample_width}").reshape(wav.sample_count, wav.channels)


def _decode(frames: npt.NDArray[typing.Any], wav: WavAudio) -> _Samples:
    if wav.sample_width == 3:
        wide = frames.astype(np.int32)
        values = wide[..., 0] | wide[..., 1] << 8 | wide[..., 2] << 16
        return np.where(values >= 1 << 23, values - (1 << 24), values).astype(np.float64)
    if wav.sample_width == 1:
        # 8-bit WAV samples are unsigned, centered on 128.
        return frames.astype(np.float64) - 128
    return frames.astype(np.float64)


def _encode(samples: _Samples, source: WavAudio) -> bytes:
    if not source.is_float:
        # Integer samples are scaled to the range -1 to 1, as ffmpeg does.
        samples = samples / (1 << (source.sample_width * 8 - 1))
    return samples.astype("<f4").tobytes()
