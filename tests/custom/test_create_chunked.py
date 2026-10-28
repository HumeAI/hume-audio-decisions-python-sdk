import email.message
import email.parser
import email.policy
import io
import json
import random
import typing
import wave

import httpx
import pytest

from hume_decisions import BadRequestError, DecisionsClient, DecisionsRequest, Question_Noul, TooManyRequestsError
from hume_decisions.audio_helpers import iter_audio_chunks
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


def _client(
    requests: typing.List[httpx.Request],
    respond: typing.Callable[[int], httpx.Response] = _succeeded,
) -> DecisionsClient:
    # `respond` receives the number of the request, starting at 1, counting retries.
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(len(requests))

    return DecisionsClient(api_key="test", httpx_client=httpx.Client(transport=httpx.MockTransport(handle)))


def test_sends_each_chunk_in_order_and_returns_each_response() -> None:
    requests: typing.List[httpx.Request] = []
    audio = _wav(2500)

    result = _client(requests).decisions.create_chunked(request=_REQUEST, file=audio, chunk_ms=1000)

    chunks = list(iter_audio_chunks(audio, chunk_ms=1000))
    assert [_parts(request)["file"].get_payload(decode=True) for request in requests] == [c.data for c in chunks]
    for request in requests:
        assert json.loads(typing.cast(bytes, _parts(request)["request"].get_payload(decode=True))) == {
            "context": "A test call.",
            "questions": {
                "talks_over_caller": {"type": "noul", "instructions": "Does the agent talk over the caller?"}
            },
        }
    assert [(c.index, c.start_ms, c.end_ms) for c in result.chunks] == [(0, 0, 1000), (1, 1000, 2000), (2, 2000, 2500)]
    assert [response.run_id for response in result.responses if response is not None] == ["run-1", "run-2", "run-3"]
    assert [c.status for c in result.chunks] == ["succeeded"] * 3
    assert [c.error for c in result.chunks] == [None] * 3
    assert result.complete


def test_sends_only_the_requested_range() -> None:
    requests: typing.List[httpx.Request] = []

    result = _client(requests).decisions.create_chunked(
        request=_REQUEST, file=_wav(5000), chunk_ms=1000, start_ms=1500, end_ms=3000
    )

    assert len(requests) == 2
    assert [(c.start_ms, c.end_ms) for c in result.chunks] == [(1500, 2500), (2500, 3000)]


def test_uploads_each_chunk_with_the_filename_and_content_type_given() -> None:
    requests: typing.List[httpx.Request] = []

    _client(requests).decisions.create_chunked(
        request=_REQUEST, file=("call.wav", io.BytesIO(_wav(2000)), "audio/wav"), chunk_ms=1000
    )

    assert len(requests) == 2
    for request in requests:
        part = _parts(request)["file"]
        assert (part.get_filename(), part.get_content_type()) == ("call.wav", "audio/wav")


def test_applies_request_options_to_every_request() -> None:
    requests: typing.List[httpx.Request] = []

    _client(requests).decisions.create_chunked(
        request=_REQUEST,
        file=_wav(2000),
        chunk_ms=1000,
        request_options={"additional_headers": {"X-Trace": "abc"}},
    )

    assert [request.headers.get("X-Trace") for request in requests] == ["abc", "abc"]


def test_sends_nothing_when_the_recording_cannot_be_split() -> None:
    requests: typing.List[httpx.Request] = []

    with pytest.raises(ValueError, match="end_ms must not exceed"):
        _client(requests).decisions.create_chunked(request=_REQUEST, file=_wav(2000), chunk_ms=1000, end_ms=2500)

    assert requests == []


def test_continues_after_a_chunk_whose_audio_cannot_be_decoded() -> None:
    requests: typing.List[httpx.Request] = []

    result = _client(
        requests, lambda number: _failed(400, "audio_decode_failed") if number == 2 else _succeeded(number)
    ).decisions.create_chunked(request=_REQUEST, file=_wav(3000), chunk_ms=1000)

    assert len(requests) == 3
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "succeeded"]
    assert isinstance(result.chunks[1].error, BadRequestError)
    assert [response.run_id if response else None for response in result.responses] == ["run-1", None, "run-3"]
    assert not result.complete


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(_failed(402, "payment_required"), id="out of tokens"),
        pytest.param(_failed(400, "question_invalid"), id="invalid request"),
        pytest.param(_failed(400, "a_code_added_later"), id="unknown code"),
    ],
)
def test_stops_at_any_other_error_and_records_the_rest_as_not_attempted(failure: httpx.Response) -> None:
    requests: typing.List[httpx.Request] = []

    result = _client(requests, lambda number: failure if number == 2 else _succeeded(number)).decisions.create_chunked(
        request=_REQUEST, file=_wav(4000), chunk_ms=1000
    )

    assert len(requests) == 2
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "not_attempted", "not_attempted"]
    assert isinstance(result.chunks[1].error, ApiError)
    assert [(c.start_ms, c.end_ms, c.error) for c in result.chunks[2:]] == [(2000, 3000, None), (3000, 4000, None)]
    assert result.responses[0] is not None
    assert not result.complete


def test_stops_when_the_connection_drops() -> None:
    requests: typing.List[httpx.Request] = []

    def respond(number: int) -> httpx.Response:
        if number == 2:
            raise httpx.RemoteProtocolError("Server disconnected")
        return _succeeded(number)

    result = _client(requests, respond).decisions.create_chunked(request=_REQUEST, file=_wav(3000), chunk_ms=1000)

    assert len(requests) == 2
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "not_attempted"]
    assert isinstance(result.chunks[1].error, httpx.RemoteProtocolError)


def test_retries_a_rate_limited_chunk_before_moving_on() -> None:
    requests: typing.List[httpx.Request] = []
    rate_limited = _failed(429, "rate_limited", headers={"retry-after-ms": "1"})

    result = _client(
        requests, lambda number: rate_limited if number == 2 else _succeeded(number)
    ).decisions.create_chunked(request=_REQUEST, file=_wav(2000), chunk_ms=1000)

    assert len(requests) == 3
    assert _parts(requests[1])["file"].get_payload(decode=True) == _parts(requests[2])["file"].get_payload(decode=True)
    assert [response.run_id if response else None for response in result.responses] == ["run-1", "run-3"]
    assert result.complete


def test_stops_when_rate_limit_retries_run_out() -> None:
    requests: typing.List[httpx.Request] = []
    rate_limited = _failed(429, "rate_limited", headers={"retry-after-ms": "1"})

    result = _client(
        requests, lambda number: rate_limited if number >= 2 else _succeeded(number)
    ).decisions.create_chunked(request=_REQUEST, file=_wav(3000), chunk_ms=1000, request_options={"max_retries": 1})

    assert len(requests) == 3
    assert [c.status for c in result.chunks] == ["succeeded", "failed", "not_attempted"]
    assert isinstance(result.chunks[1].error, TooManyRequestsError)
