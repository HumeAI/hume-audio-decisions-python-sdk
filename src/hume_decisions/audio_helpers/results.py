# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import asyncio
import dataclasses
import typing

import httpx
from ..core.api_error import ApiError
from ..core.file import File
from ..types.decisions_response import DecisionsResponse
from ..types.error import Error
from .chunks import ChunkSplit, PlannedChunk, plan_audio_chunks

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
        the recording. The first chunk of an MP3 file can start a few tens of milliseconds before 0.

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

    start_ms : int
        The `start_ms` the job was given, or 0 when it was given none.
    """

    chunks: typing.Tuple[ChunkResult, ...]
    start_ms: int = 0

    @property
    def responses(self) -> typing.List[typing.Optional[DecisionsResponse]]:
        """Each chunk's response, or None for a chunk without one, in the order of the chunks."""
        return [chunk.response for chunk in self.chunks]

    @property
    def complete(self) -> bool:
        """Whether every chunk's request succeeded."""
        return all(chunk.response is not None for chunk in self.chunks)

    @property
    def resume_ms(self) -> typing.Optional[int]:
        """
        Where the chunk before the one that stopped the job ends, or the job's `start_ms` when the
        first chunk stopped it, to pass as `start_ms`, with the same `end_ms`, to a new job that
        sends the audio of the chunk that stopped the job and the chunks after it again. None when no
        chunk stopped the job.
        """
        for position, chunk in enumerate(self.chunks):
            if chunk.error is not None and not _fails_only_this_chunk(chunk.error, self.chunks[:position]):
                # An MP3 chunk's start_ms includes frames repeated from before it, so resuming there
                # would send those again as new audio.
                return self.chunks[position - 1].end_ms if position > 0 else self.start_ms
        return None


def iter_send_chunks(
    create: typing.Callable[[File], DecisionsResponse],
    file: File,
    *,
    max_chunk_ms: int,
    start_ms: typing.Optional[int],
    end_ms: typing.Optional[int],
    split: ChunkSplit,
) -> typing.Iterator[ChunkResult]:
    # Planning reads and checks the whole file, so a file that cannot be split raises here, when the
    # caller asks for the iterator, rather than when iteration starts.
    chunks = plan_audio_chunks(file, max_chunk_ms=max_chunk_ms, start_ms=start_ms, end_ms=end_ms, split=split)
    return _send(create, file, chunks)


def _send(
    create: typing.Callable[[File], DecisionsResponse], file: File, chunks: typing.Iterable[PlannedChunk]
) -> typing.Iterator[ChunkResult]:
    results: typing.List[ChunkResult] = []
    stopped = False
    # A chunk's data is copied only when it is sent, so the chunks after a stop cost nothing.
    for chunk in chunks:
        response: typing.Optional[DecisionsResponse] = None
        error: typing.Optional[Exception] = None
        if not stopped:
            data = chunk.build()
            try:
                response = create(_with_content(file, data))
            except (ApiError, httpx.HTTPError) as raised:
                error = raised
                stopped = not _fails_only_this_chunk(raised, results)
        result = ChunkResult(chunk.index, chunk.start_ms, chunk.end_ms, response=response, error=error)
        results.append(result)
        yield result


async def iter_send_chunks_async(
    create: typing.Callable[[File], typing.Awaitable[DecisionsResponse]],
    file: File,
    *,
    max_chunk_ms: int,
    start_ms: typing.Optional[int],
    end_ms: typing.Optional[int],
    split: ChunkSplit,
) -> typing.AsyncIterator[ChunkResult]:
    # Reading and parsing the whole file happens here, before the first chunk, so it runs in a worker
    # thread to keep the event loop free.
    chunks = await asyncio.to_thread(
        plan_audio_chunks, file, max_chunk_ms=max_chunk_ms, start_ms=start_ms, end_ms=end_ms, split=split
    )
    results: typing.List[ChunkResult] = []
    stopped = False
    for chunk in chunks:
        response: typing.Optional[DecisionsResponse] = None
        error: typing.Optional[Exception] = None
        if not stopped:
            data = chunk.build()
            try:
                response = await create(_with_content(file, data))
            except (ApiError, httpx.HTTPError) as raised:
                error = raised
                stopped = not _fails_only_this_chunk(raised, results)
        result = ChunkResult(chunk.index, chunk.start_ms, chunk.end_ms, response=response, error=error)
        results.append(result)
        yield result


def _fails_only_this_chunk(error: Exception, earlier: typing.Sequence[ChunkResult]) -> bool:
    # A decode failure can come from one damaged stretch of the recording, but until some chunk has
    # decoded, the whole file is the likelier cause and every remaining chunk would fail the same way.
    # Any other error, including codes the API adds later, is assumed to fail every remaining chunk,
    # so sending them would only spend requests. The generated client has already retried rate limits
    # and server errors before raising.
    decode_failed = (
        isinstance(error, ApiError) and isinstance(error.body, Error) and error.body.code == "audio_decode_failed"
    )
    return decode_failed and any(result.response is not None for result in earlier)


def _with_content(file: File, data: bytes) -> File:
    # Each chunk is uploaded with the filename, content type and headers the caller gave the file.
    if not isinstance(file, tuple):
        return data
    return (file[0], data, *file[2:])
