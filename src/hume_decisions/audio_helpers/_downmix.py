# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import math
import typing

import numpy as np
import numpy.typing as npt
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


def downmix_wav(
    data: memoryview, wav: WavAudio, strategy: DownmixStrategy
) -> typing.Optional[typing.Tuple[memoryview, WavAudio]]:
    """
    The samples of `wav` resampled to at most MODEL_SAMPLE_RATE and mixed down by `strategy`, with a
    description of them, or None when the audio needs no change.
    """
    # ffmpeg weighs more than two channels by their speaker positions and leaves out the LFE channel,
    # so mixing them here would change what the model hears. Resampling each one changes nothing,
    # because mixing and resampling can be done in either order.
    channels = wav.channels if strategy == "mix" and wav.channels > 2 else 1
    sample_rate = min(wav.sample_rate, MODEL_SAMPLE_RATE)
    if channels == wav.channels and sample_rate == wav.sample_rate:
        return None
    mixes = strategy == "mix" and wav.channels == 2
    frames = _frames(data, wav)

    def read(start: int, stop: int) -> _Samples:
        samples = _decode(frames[start:stop], wav)
        if mixes:
            # ffmpeg adds the two channels at -3 dB each when it writes floating point, as the API does.
            return samples.sum(axis=1, keepdims=True) * math.sqrt(0.5)
        return samples[:, :channels]

    count = -(-wav.sample_count * sample_rate // wav.sample_rate)
    if mixes:
        # The sum can exceed the range of an integer format, so it is kept as floating point, which is
        # how the model receives every sample.
        converted = samples_wav(channels, sample_rate, 4, True, count)
    else:
        converted = samples_wav(channels, sample_rate, wav.sample_width, wav.is_float, count)
    blocks: typing.Iterator[typing.Tuple[int, _Samples]]
    if sample_rate == wav.sample_rate:
        blocks = ((start, read(start, min(start + _BLOCK, count))) for start in range(0, count, _BLOCK))
    else:
        blocks = _resample(read, wav.sample_count, wav.sample_rate, sample_rate, count)
    out = bytearray(count * converted.frame_bytes)
    for start, samples in blocks:
        encoded = _encode(samples, wav, converted)
        offset = start * converted.frame_bytes
        out[offset : offset + len(encoded)] = encoded
    return memoryview(out), converted


def _resample(
    read: typing.Callable[[int, int], _Samples], length: int, from_rate: int, to_rate: int, count: int
) -> typing.Iterator[typing.Tuple[int, _Samples]]:
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
        yield start, np.einsum("kct,kt->kc", windows[first - first[0]], bank[phase])


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


def _encode(samples: _Samples, source: WavAudio, wav: WavAudio) -> bytes:
    if wav.is_float:
        if not source.is_float:
            # Integer samples are scaled to the range -1 to 1, as ffmpeg does.
            samples = samples / (1 << (source.sample_width * 8 - 1))
        return samples.astype(f"<f{wav.sample_width}").tobytes()
    bits = wav.sample_width * 8
    # Rounded to the nearest integer without dither and clipped, as ffmpeg does by default.
    values = np.clip(np.rint(samples), -(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    if wav.sample_width == 1:
        return (values + 128).astype(np.uint8).tobytes()
    if wav.sample_width == 3:
        return values.astype("<i4").view(np.uint8).reshape(*values.shape, 4)[..., :3].tobytes()
    return values.astype(f"<i{wav.sample_width}").tobytes()
