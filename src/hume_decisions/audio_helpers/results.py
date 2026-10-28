# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import dataclasses
import typing

from ..core.api_error import ApiError
from ..core.file import File
from ..types.decisions_response import DecisionsResponse
from ..types.error import Error
from .chunks import iter_audio_chunks

ChunkStatus = typing.Literal["succeeded", "failed", "not_attempted"]


@dataclasses.dataclass(frozen=True)
class ChunkResult:
    """
    The outcome of one chunk's request.

    Attributes
    ----------
    index : int
        Position of the chunk in the recording, starting at 0.

    start_ms : int
        Where the chunk starts in the recording, in milliseconds. A moment's `audio_start_ms` and
        `audio_end_ms` are measured from the start of the chunk, so add this to place the moment in
        the recording.

    end_ms : int
        Where the chunk ends in the recording, in milliseconds.

    response : typing.Optional[DecisionsResponse]
        The API's response for the chunk, unchanged, when the request succeeded.

    error : typing.Optional[Exception]
        The error the request raised, when it failed. None when the request succeeded or was never
        sent.
    """

    index: int
    start_ms: int
    end_ms: int
    response: typing.Optional[DecisionsResponse]
    error: typing.Optional[Exception]

    @property
    def status(self) -> ChunkStatus:
        if self.response is not None:
            return "succeeded"
        if self.error is not None:
            return "failed"
        return "not_attempted"


@dataclasses.dataclass(frozen=True)
class ChunkedDecisionsResult:
    """
    The outcome of every chunk's request, in the order of the chunks.

    Attributes
    ----------
    chunks : typing.Tuple[ChunkResult, ...]
        One result for each chunk of the requested range.
    """

    chunks: typing.Tuple[ChunkResult, ...]

    @property
    def responses(self) -> typing.List[typing.Optional[DecisionsResponse]]:
        """Each chunk's response, or None for a chunk without one, in the order of the chunks."""
        return [chunk.response for chunk in self.chunks]

    @property
    def complete(self) -> bool:
        """Whether every chunk's request succeeded."""
        return all(chunk.response is not None for chunk in self.chunks)


def send_chunks(
    create: typing.Callable[[File], DecisionsResponse],
    file: File,
    *,
    chunk_ms: int,
    start_ms: typing.Optional[int],
    end_ms: typing.Optional[int],
) -> ChunkedDecisionsResult:
    results = []
    stopped = False
    for chunk in iter_audio_chunks(file, chunk_ms=chunk_ms, start_ms=start_ms, end_ms=end_ms):
        response: typing.Optional[DecisionsResponse] = None
        error: typing.Optional[Exception] = None
        if not stopped:
            try:
                response = create(_with_content(file, chunk.data))
            except Exception as raised:
                error = raised
                stopped = not _fails_only_this_chunk(raised)
        results.append(ChunkResult(chunk.index, chunk.start_ms, chunk.end_ms, response=response, error=error))
    return ChunkedDecisionsResult(tuple(results))


def _fails_only_this_chunk(error: Exception) -> bool:
    # Any other error, including codes the API adds later, is assumed to fail every remaining chunk
    # the same way, so sending them would only spend requests. The generated client has already
    # retried rate limits and server errors before raising.
    return isinstance(error, ApiError) and isinstance(error.body, Error) and error.body.code == "audio_decode_failed"


def _with_content(file: File, data: bytes) -> File:
    # Each chunk is uploaded with the filename, content type and headers the caller gave the file.
    if not isinstance(file, tuple):
        return data
    if len(file) == 2:
        return (file[0], data)
    if len(file) == 3:
        return (file[0], data, file[2])
    return (file[0], data, file[2], file[3])
