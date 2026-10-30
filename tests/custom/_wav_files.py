import struct
import typing

import numpy as np
import numpy.typing as npt

Samples = npt.NDArray[np.float64]

_PCM = 0x0001
_IEEE_FLOAT = 0x0003


class WavFile(typing.NamedTuple):
    sample_rate: int
    channels: int
    sample_width: int
    is_float: bool
    # Shaped (frames, channels) and scaled to -1 to 1.
    samples: Samples


def wav_file(*, data: bytes, sample_rate: int, channels: int, sample_width: int = 2, is_float: bool = False) -> bytes:
    """A WAV file with a basic header holding `data`, samples already in the given format."""
    frame_bytes = sample_width * channels
    fmt = struct.pack(
        "<HHIIHH",
        _IEEE_FLOAT if is_float else _PCM,
        channels,
        sample_rate,
        sample_rate * frame_bytes,
        frame_bytes,
        sample_width * 8,
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def samples_wav(samples: Samples, *, sample_rate: int, sample_width: int = 2, is_float: bool = False) -> bytes:
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
    return wav_file(
        data=data, sample_rate=sample_rate, channels=samples.shape[1], sample_width=sample_width, is_float=is_float
    )


def read_wav(file: bytes) -> WavFile:
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
    if tag == _IEEE_FLOAT:
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
    return WavFile(sample_rate, channels, width, tag == _IEEE_FLOAT, samples.reshape(-1, channels))
