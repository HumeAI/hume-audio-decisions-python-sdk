# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import dataclasses
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
# Samples a layer III decoder outputs before the first sample the encoder was given.
_DECODER_DELAY = 529
# Encoders whose header frame carries a LAME tag, which states the samples added at each end.
_LAME_TAG_ENCODERS = (b"LAME", b"Lavf", b"Lavc")
# Tags that some writers append after the last frame. Nothing after one is audio.
_TRAILING_TAGS = (b"TAG", b"APETAGEX", b"LYRICSBEGIN", b"ID3")


class _Header(typing.NamedTuple):
    version: int
    sample_rate: int
    length: int
    # Bytes from the start of the frame to its side information, which follows the header and an
    # optional checksum.
    side_info_start: int
    side_info_size: int


class _HeaderFrame(typing.NamedTuple):
    leading_samples: int
    trailing_samples: int
    stated_frames: typing.Optional[int]


@dataclasses.dataclass(frozen=True)
class Mp3Audio:
    sample_rate: int
    samples_per_frame: int
    # Where each frame starts, followed by where the last one ends. Frames are contiguous.
    offsets: typing.Tuple[int, ...]
    # For each frame, how many bytes of its audio data are stored in the frames before it, and how
    # many bytes of audio data, its own or later frames', it holds itself.
    reservoir_bytes: typing.Tuple[int, ...]
    main_data_bytes: typing.Tuple[int, ...]
    # Samples decoded from the frames before the recording starts and after it ends, which players
    # leave out when a LAME tag states them. Both are 0 without one.
    leading_samples: int
    trailing_samples: int

    @property
    def frame_count(self) -> int:
        return len(self.offsets) - 1

    @property
    def end_sample(self) -> int:
        """The sample after the recording's last, counted from the start of the first frame."""
        return self.frame_count * self.samples_per_frame - self.trailing_samples

    def frames_needed_before(self, frame: int) -> int:
        """Counts the frames that must come before `frame` in a chunk for it to decode exactly."""
        if frame == 0:
            return 0
        # Each frame's output is blended with the end of the previous frame's, so the previous frame
        # must decode exactly too, and so needs its own stored audio data.
        return max(self._frames_holding_data_of(frame), 1 + self._frames_holding_data_of(frame - 1))

    def _frames_holding_data_of(self, frame: int) -> int:
        needed = self.reservoir_bytes[frame]
        if needed == 0:
            return 0
        count = 0
        while needed > 0 and count < frame:
            count += 1
            needed -= self.main_data_bytes[frame - count]
        # Decoders such as ffmpeg and mpg123 do not keep the audio data of the first frame they read
        # when they cannot decode it, so the data must start after the chunk's first frame.
        return min(count + 1, frame)


def is_mp3(data: memoryview) -> bool:
    return bytes(data[:3]) == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)


def parse_mp3(data: memoryview) -> Mp3Audio:
    offset = _skip_id3v2(data)
    first = _read_header(data, offset)
    # The first frame can be a Xing, Info or VBRI header, which holds the length of the whole file
    # instead of audio. Left in a chunk, it would tell players the chunk is that long.
    header_frame = _read_header_frame(data, offset, first) if offset + first.length <= len(data) else None
    if header_frame is not None:
        offset += first.length
    offsets: typing.List[int] = []
    reservoir_bytes: typing.List[int] = []
    main_data_bytes: typing.List[int] = []
    while offset + 4 <= len(data) and not any(bytes(data[offset : offset + len(tag)]) == tag for tag in _TRAILING_TAGS):
        try:
            header = _read_header(data, offset)
        except ValueError:
            # Players skip bytes after the last frame that are neither a frame nor a tag, such as
            # padding. When frames follow them, the bytes are damage instead.
            if _frames_follow(data, offset + 1, first):
                raise
            break
        if header.version != first.version or header.sample_rate != first.sample_rate:
            raise _unsupported(f"the sample rate changes at byte {offset}")
        # A file cut short is read up to its last whole frame.
        if offset + header.length > len(data):
            break
        offsets.append(offset)
        reservoir_bytes.append(_reservoir_bytes(data, offset, header))
        main_data_bytes.append(header.length - header.side_info_start - header.side_info_size)
        offset += header.length
    offsets.append(offset)
    leading, trailing, stated_frames = header_frame or _HeaderFrame(0, 0, None)
    # The samples added at the end close the frames the header counts. A file cut short has lost
    # them, and players then leave nothing out at the end.
    if stated_frames != len(offsets) - 1:
        trailing = 0
    return Mp3Audio(
        sample_rate=first.sample_rate,
        samples_per_frame=1152 if first.version == _MPEG1 else 576,
        offsets=tuple(offsets),
        reservoir_bytes=tuple(reservoir_bytes),
        main_data_bytes=tuple(main_data_bytes),
        leading_samples=leading,
        trailing_samples=trailing,
    )


def mp3_chunk(data: memoryview, audio: Mp3Audio, start: int, end: int) -> bytes:
    """Builds an MP3 file holding frames `start` to `end` and none of the source's tags."""
    return bytes(data[audio.offsets[start] : audio.offsets[end]])


def _reservoir_bytes(data: memoryview, offset: int, header: _Header) -> int:
    # Layer III frames may store audio data in the unused space of the frames before them. The side
    # information starts with how far back that data begins: 9 bits in MPEG-1 and 8 bits otherwise.
    side_info = offset + header.side_info_start
    if header.version == _MPEG1:
        return data[side_info] << 1 | data[side_info + 1] >> 7
    return data[side_info]


def _skip_id3v2(data: memoryview) -> int:
    offset = 0
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
    mono = b3 >> 6 == _MONO
    side_info_size = (17 if mono else 32) if mpeg1 else (9 if mono else 17)
    # A protection bit of 0 means a 16-bit checksum follows the header.
    side_info_start = 4 if b1 & 1 else 6
    if length < side_info_start + side_info_size:
        raise _unsupported(f"invalid frame header at byte {offset}")
    return _Header(version, sample_rate, length, side_info_start, side_info_size)


def _read_header_frame(data: memoryview, offset: int, header: _Header) -> typing.Optional[_HeaderFrame]:
    """
    Reads a Xing, Info or VBRI header frame: the samples added before and after the recording, and
    the frame count it states. None when the frame holds audio instead.
    """
    # Encoders place the Xing tag after the side information as if the frame had no checksum.
    xing = offset + 4 + header.side_info_size
    if bytes(data[xing : xing + 4]) not in (b"Xing", b"Info"):
        return _HeaderFrame(0, 0, None) if bytes(data[offset + 36 : offset + 40]) == b"VBRI" else None
    flags = int.from_bytes(data[xing + 4 : xing + 8], "big")
    position = xing + 8
    stated_frames = None
    if flags & 1:
        stated_frames = int.from_bytes(data[position : position + 4], "big")
        position += 4
    # The byte count, table of contents and quality follow when their flags are set.
    position += (4 if flags & 2 else 0) + (100 if flags & 4 else 0) + (4 if flags & 8 else 0)
    if position + 24 > offset + header.length or bytes(data[position : position + 4]) not in _LAME_TAG_ENCODERS:
        return _HeaderFrame(0, 0, stated_frames)
    # Two 12-bit counts after the encoder's name and other settings: the samples the encoder added
    # before the recording, and after it. The decoder's own delay precedes both.
    b0, b1, b2 = data[position + 21 : position + 24]
    delay = b0 << 4 | b1 >> 4
    padding = (b1 & 0x0F) << 8 | b2
    return _HeaderFrame(delay + _DECODER_DELAY, max(0, padding - _DECODER_DELAY), stated_frames)


def _frames_follow(data: memoryview, offset: int, first: _Header) -> bool:
    """Whether two frames in a row, of the same version and sample rate as `first`, start at or after `offset`."""
    # Requiring two frames in a row keeps bytes that happen to look like a frame header from counting.
    tail = bytes(data[offset:])
    position = tail.find(0xFF)
    while position != -1:
        header = _matching_header(data, offset + position, first)
        if header is not None and _matching_header(data, offset + position + header.length, first) is not None:
            return True
        position = tail.find(0xFF, position + 1)
    return False


def _matching_header(data: memoryview, offset: int, first: _Header) -> typing.Optional[_Header]:
    try:
        header = _read_header(data, offset)
    except ValueError:
        return None
    return header if (header.version, header.sample_rate) == (first.version, first.sample_rate) else None


def _unsupported(reason: str) -> ValueError:
    return ValueError(
        f"The audio is not a supported MP3 file ({reason}). Convert it first, for example with: {_CONVERT_COMMAND}"
    )
