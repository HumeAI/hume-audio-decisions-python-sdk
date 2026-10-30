# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import io
import typing

from ..core.file import File, FileContent
from ._downmix import DownmixStrategy, check_downmix_strategy, downmix_wav
from ._wav import is_wav, parse_wav, wav_chunk


def read_file(file: File) -> typing.Tuple[memoryview, bool]:
    """The file's contents, and whether they were given as a str."""
    content = _content(file)
    if not isinstance(content, (str, bytes)):
        # Read the way httpx reads a file it uploads, so the chunks cover what `create` would send.
        try:
            content.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        content = content.read()
    if isinstance(content, str):
        return memoryview(content.encode("utf-8")), True
    return memoryview(content), False


def with_content(file: File, data: bytes) -> File:
    if not isinstance(file, tuple):
        return data
    return (file[0], data, *file[2:])


def downmix_file(file: File, downmix_strategy: typing.Optional[DownmixStrategy]) -> File:
    """`file` with a WAV recording mixed down as `downmix_strategy` asks, to send with `create`."""
    check_downmix_strategy(downmix_strategy)
    if downmix_strategy is None:
        return file
    data, _ = read_file(file)
    if is_wav(data):
        try:
            wav = parse_wav(data)
        except ValueError:
            # A WAV file this parser cannot read may still be one the API can decode, so it is left to the API.
            pass
        else:
            samples, converted = downmix_wav(data, wav, downmix_strategy)
            if converted is not wav:
                return with_content(file, wav_chunk(samples, converted, 0, converted.sample_count))
    if isinstance(_content(file), (str, bytes)):
        return file
    # A stream that has been read cannot always be read again, so what was read is sent instead.
    return with_content(file, bytes(data))


def _content(file: File) -> FileContent:
    return file[1] if isinstance(file, tuple) else file
