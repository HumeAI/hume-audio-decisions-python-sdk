# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import typing

_CONVERT_COMMAND = "ffmpeg -i input.mp3 -c:a libmp3lame -q:a 2 output.mp3"

# Keyed by the version bits of a frame header. Version 1 is reserved.
_SAMPLE_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
_MPEG1 = 3
# Layer III bitrates in kbit/s, keyed by whether the frame is MPEG-1. Index 0 is free format.
_BITRATES = {
    True: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    False: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_LAYER_III = 1
_MONO = 3
# Tags that some writers append after the last frame. Nothing after one is audio.
_TRAILING_TAGS = (b"TAG", b"APETAGEX", b"LYRICSBEGIN", b"ID3")


class _Header(typing.NamedTuple):
    version: int
    sample_rate: int
    length: int
    mono: bool


class Mp3Audio(typing.NamedTuple):
    sample_rate: int
    samples_per_frame: int
    # Where each frame starts, followed by where the last one ends. Frames are contiguous.
    offsets: typing.List[int]

    @property
    def frame_count(self) -> int:
        return len(self.offsets) - 1


def is_mp3(data: memoryview) -> bool:
    return bytes(data[:3]) == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)


def parse_mp3(data: memoryview) -> Mp3Audio:
    offset = _skip_id3v2(data, 0)
    first = _read_header(data, offset)
    offsets: typing.List[int] = []
    if offset + first.length <= len(data):
        # The first frame can be a Xing, Info or VBRI header, which holds the length of the whole
        # file instead of audio. Left in a chunk, it would tell players the chunk is that long.
        if not _is_vbr_header(data, offset, first):
            offsets.append(offset)
        offset += first.length
    while offset + 4 <= len(data) and not any(bytes(data[offset : offset + len(tag)]) == tag for tag in _TRAILING_TAGS):
        header = _read_header(data, offset)
        if header.version != first.version or header.sample_rate != first.sample_rate:
            raise _unsupported(f"the sample rate changes at byte {offset}")
        # A file cut short is read up to its last whole frame.
        if offset + header.length > len(data):
            break
        offsets.append(offset)
        offset += header.length
    offsets.append(offset)
    return Mp3Audio(first.sample_rate, 1152 if first.version == _MPEG1 else 576, offsets)


def mp3_chunk(data: memoryview, audio: Mp3Audio, start: int, end: int) -> bytes:
    """Builds an MP3 file holding frames `start` to `end` and none of the source's tags."""
    return bytes(data[audio.offsets[start] : audio.offsets[end]])


def _skip_id3v2(data: memoryview, offset: int) -> int:
    while bytes(data[offset : offset + 3]) == b"ID3" and offset + 10 <= len(data):
        flags = data[offset + 5]
        # ID3v2 sizes store 7 bits in each byte.
        size = 0
        for byte in data[offset + 6 : offset + 10]:
            size = size << 7 | byte & 0x7F
        offset += 10 + size + (10 if flags & 0x10 else 0)
    return offset


def _read_header(data: memoryview, offset: int) -> _Header:
    if offset + 4 > len(data):
        raise _unsupported("no MPEG audio frame")
    b0, b1, b2, b3 = data[offset : offset + 4]
    if b0 != 0xFF or b1 & 0xE0 != 0xE0:
        raise _unsupported(f"no MPEG audio frame at byte {offset}")
    version = b1 >> 3 & 3
    layer = b1 >> 1 & 3
    bitrate_index = b2 >> 4
    sample_rate_index = b2 >> 2 & 3
    if version not in _SAMPLE_RATES or layer == 0 or bitrate_index == 15 or sample_rate_index == 3:
        raise _unsupported(f"invalid frame header at byte {offset}")
    if layer != _LAYER_III:
        raise _unsupported("MPEG layer I and II audio is not supported")
    if bitrate_index == 0:
        raise _unsupported("free format MP3 files are not supported")
    mpeg1 = version == _MPEG1
    sample_rate = _SAMPLE_RATES[version][sample_rate_index]
    length = (144000 if mpeg1 else 72000) * _BITRATES[mpeg1][bitrate_index] // sample_rate + (b2 >> 1 & 1)
    return _Header(version, sample_rate, length, b3 >> 6 == _MONO)


def _is_vbr_header(data: memoryview, offset: int, header: _Header) -> bool:
    if header.version == _MPEG1:
        side_info = 17 if header.mono else 32
    else:
        side_info = 9 if header.mono else 17
    xing = offset + 4 + side_info
    return bytes(data[xing : xing + 4]) in (b"Xing", b"Info") or bytes(data[offset + 36 : offset + 40]) == b"VBRI"


def _unsupported(reason: str) -> ValueError:
    return ValueError(
        f"The audio is not a supported MP3 file ({reason}). Convert it first, for example with: {_CONVERT_COMMAND}"
    )
