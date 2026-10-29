# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import dataclasses
import struct
import typing

_CONVERT_COMMAND = "ffmpeg -i input.wav -c:a pcm_s16le output.wav"

_PCM = 0x0001
_IEEE_FLOAT = 0x0003
_EXTENSIBLE = 0xFFFE
# An extensible header's subformat GUID is the format tag it stands for followed by these bytes.
_SUBFORMAT_GUID_SUFFIX = bytes.fromhex("000000001000800000aa00389b71")
# Writers that cannot seek back to fill in the size, such as ffmpeg writing to a pipe, leave this.
_UNKNOWN_SIZE = 0xFFFFFFFF
_SAMPLE_WIDTHS = {_PCM: (1, 2, 3, 4), _IEEE_FLOAT: (4, 8)}
_EXTENSIBLE_FMT_BYTES = 40

# Chunks are at least 1 ms long, so at this rate every chunk holds at least one sample.
MIN_SAMPLE_RATE = 1000
# The resampling filter grows with the sample rate, so without a limit a few bytes of header could
# demand gigabytes. No recording format in common use goes above this rate.
_MAX_SAMPLE_RATE = 768_000


@dataclasses.dataclass(frozen=True)
class WavAudio:
    # The fmt chunk body written into every chunk.
    fmt: bytes
    channels: int
    sample_rate: int
    # Bytes in one sample of one channel.
    sample_width: int
    is_float: bool
    data_start: int
    sample_count: int

    @property
    def frame_bytes(self) -> int:
        """Bytes in one sample of every channel."""
        return self.sample_width * self.channels


class _Format(typing.NamedTuple):
    fmt: bytes
    channels: int
    sample_rate: int
    sample_width: int
    is_float: bool


def is_wav(data: memoryview) -> bool:
    return bytes(data[:4]) in (b"RIFF", b"RF64")


def parse_wav(data: memoryview) -> WavAudio:
    if bytes(data[:4]) == b"RF64":
        raise _unsupported("RF64 files are not supported")
    if len(data) < 12 or bytes(data[:4]) != b"RIFF" or bytes(data[8:12]) != b"WAVE":
        raise _unsupported("no RIFF WAVE header")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    offset = 12
    fmt: typing.Optional[_Format] = None
    while True:
        if offset + 8 > len(data):
            raise _unsupported("no data chunk")
        chunk_id = bytes(data[offset : offset + 4])
        size = struct.unpack_from("<I", data, offset + 4)[0]
        offset += 8
        if chunk_id == b"data":
            if fmt is None:
                raise _unsupported("the data chunk comes before the fmt chunk")
            available = len(data) - offset
            # Writers such as libsndfile fill in the sizes only when they close the file, so a recording
            # that was never closed declares no audio in a RIFF chunk that ends before it. Any other file
            # that declares no audio has none, including one whose RIFF size is unknown, because chunks
            # such as metadata can follow the data.
            unclosed = size == 0 and 8 + riff_size <= offset
            # A file cut short is read as far as it goes, up to its last whole sample.
            size = available if unclosed or size == _UNKNOWN_SIZE else min(size, available)
            return WavAudio(
                fmt=fmt.fmt,
                channels=fmt.channels,
                sample_rate=fmt.sample_rate,
                sample_width=fmt.sample_width,
                is_float=fmt.is_float,
                data_start=offset,
                sample_count=size // (fmt.sample_width * fmt.channels),
            )
        if chunk_id == b"fmt ":
            fmt = _parse_format(bytes(data[offset : offset + min(size, _EXTENSIBLE_FMT_BYTES)]))
        # Chunks with an odd size are followed by a padding byte.
        offset += size + size % 2


def wav_chunk(data: memoryview, audio: WavAudio, start: int, end: int) -> bytes:
    """Builds a WAV file holding samples `start` to `end` and none of the source's other chunks."""
    first = audio.data_start + start * audio.frame_bytes
    size = (end - start) * audio.frame_bytes
    padding = b"\x00" * (size % 2)
    header = (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(audio.fmt) + 8 + size + len(padding))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(audio.fmt))
        + audio.fmt
        + b"data"
        + struct.pack("<I", size)
    )
    return b"".join((header, data[first : first + size], padding))


def samples_wav(channels: int, sample_rate: int, sample_width: int, is_float: bool, sample_count: int) -> WavAudio:
    """Describes samples that start at offset 0 of their own buffer."""
    fmt = _fmt_body(_IEEE_FLOAT if is_float else _PCM, channels, sample_rate, sample_width, sample_width * 8, b"")
    return WavAudio(
        fmt=fmt,
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        is_float=is_float,
        data_start=0,
        sample_count=sample_count,
    )


def _parse_format(body: bytes) -> _Format:
    if len(body) < 16:
        raise _unsupported("the fmt chunk is too short")
    header_tag, channels, sample_rate, _, block_align, bits = struct.unpack("<HHIIHH", body[:16])
    tag = header_tag
    if tag == _EXTENSIBLE:
        if len(body) < _EXTENSIBLE_FMT_BYTES or body[26:40] != _SUBFORMAT_GUID_SUFFIX:
            raise _unsupported("unknown extensible subformat")
        tag = struct.unpack("<H", body[24:26])[0]
    if tag not in _SAMPLE_WIDTHS:
        raise _unsupported(f"format tag 0x{tag:04X} is compressed or unknown")
    sample_width = (bits + 7) // 8
    if sample_width not in _SAMPLE_WIDTHS[tag] or (tag == _IEEE_FLOAT and bits != sample_width * 8):
        raise _unsupported(f"{bits}-bit {'PCM' if tag == _PCM else 'floating point'}")
    # Some writers leave the block size at 0 or 1, so it is recomputed for the chunks. A block larger
    # than one sample per channel means padded samples, which would be cut in the wrong places. The
    # recomputed block size must fit the header's 16 bits, and the byte rate its 32 bits.
    frame_bytes = sample_width * channels
    if channels < 1 or block_align > frame_bytes or frame_bytes > 0xFFFF or sample_rate * frame_bytes > 0xFFFFFFFF:
        raise _unsupported("the fmt chunk is inconsistent")
    if sample_rate < MIN_SAMPLE_RATE:
        raise _unsupported(f"{sample_rate} Hz is below the minimum of {MIN_SAMPLE_RATE} Hz")
    if sample_rate > _MAX_SAMPLE_RATE:
        raise _unsupported(f"{sample_rate} Hz is above the maximum of {_MAX_SAMPLE_RATE} Hz")
    extension = body[16:_EXTENSIBLE_FMT_BYTES] if header_tag == _EXTENSIBLE else b""
    fmt = _fmt_body(header_tag, channels, sample_rate, sample_width, bits, extension)
    return _Format(fmt, channels, sample_rate, sample_width, tag == _IEEE_FLOAT)


def _fmt_body(
    header_tag: int, channels: int, sample_rate: int, sample_width: int, bits: int, extension: bytes
) -> bytes:
    frame_bytes = sample_width * channels
    fmt = struct.pack("<HHIIHH", header_tag, channels, sample_rate, sample_rate * frame_bytes, frame_bytes, bits)
    if header_tag == _EXTENSIBLE:
        fmt += extension
    elif header_tag != _PCM:
        # Formats other than PCM declare the size of their header extension, which is empty here.
        fmt += struct.pack("<H", 0)
    return fmt


def _unsupported(reason: str) -> ValueError:
    return ValueError(
        f"The audio is not a supported WAV file ({reason}). Convert it first, for example with: {_CONVERT_COMMAND}"
    )
