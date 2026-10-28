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

from hume_decisions import DecisionsClient, DecisionsRequest, Question_Noul
from hume_decisions.audio_helpers import iter_audio_chunks

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


def _client(requests: typing.List[httpx.Request]) -> DecisionsClient:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"run_id": f"run-{len(requests)}", "answers": {"talks_over_caller": {"type": "noul", "noul": 0.5}}},
        )

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
