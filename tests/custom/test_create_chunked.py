import asyncio
import email.message
import email.parser
import email.policy
import io
import json
import pathlib
import random
import typing
import wave

import httpx
import pytest

from hume_decisions import (
    AsyncDecisionsClient,
    BadRequestError,
    DecisionsClient,
    DecisionsRequest,
    Question_Noul,
    TooManyRequestsError,
)
from hume_decisions.audio_helpers import ChunkedDecisionsResult, ChunkResult, chunks, iter_audio_chunks
from hume_decisions.core import ApiError

_REQUEST = DecisionsRequest(
    context="A test call.",
    questions={"talks_over_caller": Question_Noul(instructions="Does the agent talk over the caller?")},
)


def _wav(milliseconds: int, *, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(random.Random(0).randbytes(sample_rate * milliseconds // 1000 * 2))
    return buffer.getvalue()


def _parts(request: httpx.Request) -> typing.Dict[str, email.message.Message]:
    header = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(header + request.content)
    return {typing.cast(str, part.get_param("name", header="content-disposition")): part for part in message.walk()}


def _succeeded(number: int) -> httpx.Response:
    return httpx.Response(
        200, json={"run_id": f"run-{number}", "answers": {"talks_over_caller": {"type": "noul", "noul": 0.5}}}
    )


def _failed(status: int, code: str, headers: typing.Optional[typing.Dict[str, str]] = None) -> httpx.Response:
    return httpx.Response(
        status, headers=headers, json={"type": "error", "code": code, "message": code, "request_id": "req"}
    )


def _rate_limited() -> httpx.Response:
    # Hume's API gateway answers a rate limit before the service does, so the body has no error code.
    return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"message": "Too many requests"})


_Respond = typing.Callable[[int], httpx.Response]
_CreateChunked = typing.Callable[..., ChunkedDecisionsResult]
_IterCreateChunked = typing.Callable[..., typing.List[typing.Tuple[ChunkResult, int]]]


def _transport(requests: typing.List[httpx.Request], respond: _Respond) -> httpx.MockTransport:
    # `respond` receives the number of the request, starting at 1, counting retries.
    def handle(http_request: httpx.Request) -> httpx.Response:
        requests.append(http_request)
        return respond(len(requests))

    return httpx.MockTransport(handle)


@pytest.fixture(params=["sync", "async"])
def create_chunked(request: pytest.FixtureRequest) -> _CreateChunked:
    """Calls create_chunked on the sync or async client, which must behave the same."""

    def call(
        requests: typing.List[httpx.Request], respond: _Respond = _succeeded, **kwargs: typing.Any
    ) -> ChunkedDecisionsResult:
        transport = _transport(requests, respond)
        if request.param == "sync":
            client = DecisionsClient(api_key="test", httpx_client=httpx.Client(transport=transport))
            return client.decisions.create_chunked(request=_REQUEST, **kwargs)
        async_client = AsyncDecisionsClient(api_key="test", httpx_client=httpx.AsyncClient(transport=transport))
        return asyncio.run(async_client.decisions.create_chunked(request=_REQUEST, **kwargs))

    return call


@pytest.fixture(params=["sync", "async"])
def iter_create_chunked(request: pytest.FixtureRequest) -> _IterCreateChunked:
    """
    Iterates iter_create_chunked on the sync or async client, pairing each result with the number of
    requests sent when it was yielded.
    """

    def call(
        requests: typing.List[httpx.Request], respond: _Respond = _succeeded, **kwargs: typing.Any
    ) -> typing.List[typing.Tuple[ChunkResult, int]]:
        transport = _transport(requests, respond)
        if request.param == "sync":
            client = DecisionsClient(api_key="test", httpx_client=httpx.Client(transport=transport))
            return [
                (chunk, len(requests)) for chunk in client.decisions.iter_create_chunked(request=_REQUEST, **kwargs)
            ]
        async_client = AsyncDecisionsClient(api_key="test", httpx_client=httpx.AsyncClient(transport=transport))

        async def collect() -> typing.List[typing.Tuple[ChunkResult, int]]:
            results = async_client.decisions.iter_create_chunked(request=_REQUEST, **kwargs)
            return [(chunk, len(requests)) async for chunk in results]

        return asyncio.run(collect())

    return call


def test_sends_each_chunk_in_order_and_returns_each_response(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []
    audio = _wav(2500)

    result = create_chunked(requests, file=audio, max_chunk_ms=1000)

    expected = list(iter_audio_chunks(audio, max_chunk_ms=1000))
    assert [_parts(request)["file"].get_payload(decode=True) for request in requests] == [c.data for c in expected]
    for request in requests:
        assert json.loads(typing.cast(bytes, _parts(request)["request"].get_payload(decode=True))) == {
            "context": "A test call.",
            "questions": {
                "talks_over_caller": {"type": "noul", "instructions": "Does the agent talk over the caller?"}
            },
        }
    assert [(c.index, c.start_ms, c.end_ms) for c in result.chunks] == [(0, 0, 833), (1, 833, 1667), (2, 1667, 2500)]
    assert [response.run_id for response in result.responses if response is not None] == ["run-1", "run-2", "run-3"]
    assert [c.status for c in result.chunks] == ["succeeded"] * 3
    assert [c.error for c in result.chunks] == [None] * 3
    assert result.complete


def test_sends_only_the_requested_range(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    result = create_chunked(requests, file=_wav(5000), max_chunk_ms=1000, start_ms=1500, end_ms=3000)

    assert len(requests) == 2
    assert [(c.start_ms, c.end_ms) for c in result.chunks] == [(1500, 2250), (2250, 3000)]


def test_sends_chunks_of_max_chunk_ms_when_split_is_fixed(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    result = create_chunked(requests, file=_wav(2500), max_chunk_ms=1000, split="fixed")

    assert len(requests) == 3
    assert [(c.start_ms, c.end_ms) for c in result.chunks] == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_uploads_each_chunk_with_the_filename_content_type_and_headers_given(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    create_chunked(
        requests,
        file=("call.wav", io.BytesIO(_wav(2000)), "audio/wav", {"X-Recording-Id": "rec-1"}),
        max_chunk_ms=1000,
    )

    assert len(requests) == 2
    for request in requests:
        part = _parts(request)["file"]
        assert (part.get_filename(), part.get_content_type(), part["X-Recording-Id"]) == (
            "call.wav",
            "audio/wav",
            "rec-1",
        )


def test_applies_request_options_to_every_request(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    create_chunked(
        requests,
        file=_wav(2000),
        max_chunk_ms=1000,
        request_options={"additional_headers": {"X-Trace": "abc"}},
    )

    assert [request.headers.get("X-Trace") for request in requests] == ["abc", "abc"]


def test_sends_nothing_when_the_recording_cannot_be_split(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    with pytest.raises(ValueError, match="end_ms must not exceed"):
        create_chunked(requests, file=_wav(2000), max_chunk_ms=1000, end_ms=2500)

    assert requests == []


def test_continues_after_a_chunk_whose_audio_cannot_be_decoded(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    result = create_chunked(
        requests,
        lambda number: _failed(400, "audio_decode_failed") if number == 2 else _succeeded(number),
        file=_wav(3000),
        max_chunk_ms=1000,
    )

    assert len(requests) == 3
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "succeeded"]
    assert isinstance(result.chunks[1].error, BadRequestError)
    assert [response.run_id if response else None for response in result.responses] == ["run-1", None, "run-3"]
    assert not result.complete


def test_stops_when_no_chunk_has_decoded_before_a_decode_failure(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    result = create_chunked(
        requests, lambda number: _failed(400, "audio_decode_failed"), file=_wav(3000), max_chunk_ms=1000
    )

    assert len(requests) == 1
    assert [c.status for c in result.chunks] == ["failed", "not_attempted", "not_attempted"]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(lambda: _failed(402, "payment_required"), id="out of tokens"),
        pytest.param(lambda: _failed(400, "question_invalid"), id="invalid request"),
        pytest.param(lambda: _failed(400, "a_code_added_later"), id="unknown code"),
        pytest.param(
            lambda: httpx.Response(401, json={"message": "Invalid API key", "request_id": "req"}),
            id="API key rejected by the gateway",
        ),
    ],
)
def test_stops_at_any_other_error_and_records_the_rest_as_not_attempted(
    create_chunked: _CreateChunked, failure: typing.Callable[[], httpx.Response]
) -> None:
    requests: typing.List[httpx.Request] = []

    result = create_chunked(
        requests,
        lambda number: failure() if number == 2 else _succeeded(number),
        file=_wav(4000),
        max_chunk_ms=1000,
    )

    assert len(requests) == 2
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "not_attempted", "not_attempted"]
    assert isinstance(result.chunks[1].error, ApiError)
    assert [(c.start_ms, c.end_ms, c.error) for c in result.chunks[2:]] == [(2000, 3000, None), (3000, 4000, None)]
    assert result.responses[0] is not None
    assert not result.complete


def test_copies_no_chunk_after_a_stop(create_chunked: _CreateChunked, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: typing.List[httpx.Request] = []
    copied: typing.List[int] = []

    def wav_chunk(data: memoryview, audio: typing.Any, start: int, end: int) -> bytes:
        copied.append(start)
        return original(data, audio, start, end)

    original = chunks.wav_chunk
    monkeypatch.setattr(chunks, "wav_chunk", wav_chunk)

    result = create_chunked(
        requests,
        lambda number: _failed(402, "payment_required") if number == 2 else _succeeded(number),
        file=_wav(4000),
        max_chunk_ms=1000,
    )

    assert [c.status for c in result.chunks] == ["succeeded", "failed", "not_attempted", "not_attempted"]
    assert copied == [0, 8000]


def test_stops_when_the_connection_drops(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    def respond(number: int) -> httpx.Response:
        if number == 2:
            raise httpx.RemoteProtocolError("Server disconnected")
        return _succeeded(number)

    result = create_chunked(requests, respond, file=_wav(3000), max_chunk_ms=1000)

    assert len(requests) == 2
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "not_attempted"]
    assert isinstance(result.chunks[1].error, httpx.RemoteProtocolError)


def test_raises_an_error_that_is_neither_from_the_api_nor_the_connection(create_chunked: _CreateChunked) -> None:
    def respond(number: int) -> httpx.Response:
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        create_chunked([], respond, file=_wav(2000), max_chunk_ms=1000)


def test_retries_a_rate_limited_chunk_before_moving_on(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    result = create_chunked(
        requests,
        lambda number: _rate_limited() if number == 2 else _succeeded(number),
        file=_wav(2000),
        max_chunk_ms=1000,
    )

    assert len(requests) == 3
    assert _parts(requests[1])["file"].get_payload(decode=True) == _parts(requests[2])["file"].get_payload(decode=True)
    assert [response.run_id if response else None for response in result.responses] == ["run-1", "run-3"]
    assert result.complete


def test_stops_when_rate_limit_retries_run_out(create_chunked: _CreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    result = create_chunked(
        requests,
        lambda number: _rate_limited() if number >= 2 else _succeeded(number),
        file=_wav(3000),
        max_chunk_ms=1000,
        request_options={"max_retries": 1},
    )

    assert len(requests) == 3
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "not_attempted"]
    assert isinstance(result.chunks[1].error, TooManyRequestsError)


def test_iterator_yields_each_result_before_sending_the_next_chunk(iter_create_chunked: _IterCreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    yielded = iter_create_chunked(
        requests,
        lambda number: _failed(402, "payment_required") if number == 3 else _succeeded(number),
        file=_wav(4000),
        max_chunk_ms=1000,
    )

    assert [(chunk.index, chunk.status, sent) for chunk, sent in yielded] == [
        (0, "succeeded", 1),
        (1, "succeeded", 2),
        (2, "failed", 3),
        (3, "not_attempted", 3),
    ]


def test_iterator_sends_nothing_when_the_recording_cannot_be_split(iter_create_chunked: _IterCreateChunked) -> None:
    requests: typing.List[httpx.Request] = []

    with pytest.raises(ValueError, match="end_ms must not exceed"):
        iter_create_chunked(requests, file=_wav(2000), max_chunk_ms=1000, end_ms=2500)

    assert requests == []


def test_resuming_at_resume_ms_sends_the_chunk_that_stopped_the_job_and_those_after_it(
    create_chunked: _CreateChunked,
) -> None:
    audio = _wav(4000)
    stopped = create_chunked(
        [],
        lambda number: _failed(402, "payment_required") if number == 2 else _succeeded(number),
        file=audio,
        max_chunk_ms=1000,
    )
    requests: typing.List[httpx.Request] = []

    resumed = create_chunked(requests, file=audio, max_chunk_ms=1000, start_ms=stopped.resume_ms)

    assert stopped.resume_ms == 1000
    assert [(c.start_ms, c.end_ms) for c in resumed.chunks] == [(1000, 2000), (2000, 3000), (3000, 4000)]
    assert resumed.complete


def test_resuming_an_mp3_job_at_resume_ms_sends_the_same_chunks_as_the_job_that_stopped(
    create_chunked: _CreateChunked,
) -> None:
    # Each chunk of this file starts with frames repeated from the chunk before it, so resuming where
    # the stopped chunk starts would send those frames again in a chunk of its own.
    audio = (pathlib.Path(__file__).parent / "assets" / "tone_mpeg25_8k.mp3").read_bytes()
    stopped = create_chunked(
        [],
        lambda number: _failed(402, "payment_required") if number == 3 else _succeeded(number),
        file=audio,
        max_chunk_ms=1000,
    )

    resumed = create_chunked([], file=audio, max_chunk_ms=1000, start_ms=stopped.resume_ms)

    assert stopped.resume_ms == 1446
    assert [(c.start_ms, c.end_ms) for c in resumed.chunks] == [(c.start_ms, c.end_ms) for c in stopped.chunks[2:]]
    assert resumed.complete


def test_resume_ms_passes_over_a_chunk_that_could_not_be_decoded_before_the_chunk_that_stopped_the_job(
    create_chunked: _CreateChunked,
) -> None:
    def respond(number: int) -> httpx.Response:
        if number == 2:
            return _failed(400, "audio_decode_failed")
        if number == 3:
            return _failed(402, "payment_required")
        return _succeeded(number)

    result = create_chunked([], respond, file=_wav(5000), max_chunk_ms=1000)

    assert [c.status for c in result.chunks] == ["succeeded", "failed", "failed", "not_attempted", "not_attempted"]
    assert result.resume_ms == 2000


@pytest.mark.parametrize(
    "respond",
    [
        pytest.param(_succeeded, id="complete"),
        pytest.param(
            lambda number: _failed(400, "audio_decode_failed") if number == 2 else _succeeded(number),
            id="only a chunk that could not be decoded failed",
        ),
    ],
)
def test_resume_ms_is_none_when_no_chunk_stopped_the_job(create_chunked: _CreateChunked, respond: _Respond) -> None:
    result = create_chunked([], respond, file=_wav(3000), max_chunk_ms=1000)

    assert result.resume_ms is None


@pytest.mark.parametrize("start_ms, resume_ms", [(None, 0), (1500, 1500)])
def test_resume_ms_is_the_job_start_when_an_mp3_job_stops_at_its_first_chunk(
    create_chunked: _CreateChunked, start_ms: typing.Optional[int], resume_ms: int
) -> None:
    # The first chunk starts with frames from before the job's start, and before 0 when the encoder
    # added samples before the recording, so its start_ms is earlier than where the job started.
    audio = (pathlib.Path(__file__).parent / "assets" / "tone_mpeg25_8k.mp3").read_bytes()
    stopped = create_chunked(
        [], lambda number: _failed(402, "payment_required"), file=audio, max_chunk_ms=1000, start_ms=start_ms
    )

    resumed = create_chunked([], file=audio, max_chunk_ms=1000, start_ms=stopped.resume_ms)

    assert stopped.chunks[0].start_ms < resume_ms
    assert stopped.resume_ms == resume_ms
    assert [(c.start_ms, c.end_ms) for c in resumed.chunks] == [(c.start_ms, c.end_ms) for c in stopped.chunks]


def test_resume_ms_is_0_for_a_result_built_from_iterator_results_with_no_start_ms_when_the_first_chunk_stopped_the_job(
    iter_create_chunked: _IterCreateChunked,
) -> None:
    start_ms: typing.Optional[int] = None
    yielded = iter_create_chunked(
        [], lambda number: _failed(402, "payment_required"), file=_wav(3000), max_chunk_ms=1000, start_ms=start_ms
    )

    result = ChunkedDecisionsResult(tuple(chunk for chunk, _ in yielded), start_ms=start_ms)

    assert result.start_ms == 0
    assert result.resume_ms == 0


class _Index:
    # Stands in for a NumPy integer, which is an integer through __index__ without subclassing int.
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def test_resume_ms_is_an_int_when_start_ms_is_an_integer_that_is_not_an_int(create_chunked: _CreateChunked) -> None:
    result = create_chunked(
        [], lambda number: _failed(402, "payment_required"), file=_wav(3000), max_chunk_ms=1000, start_ms=_Index(1500)
    )

    assert type(result.start_ms) is int
    assert type(result.resume_ms) is int
    assert result.resume_ms == 1500
