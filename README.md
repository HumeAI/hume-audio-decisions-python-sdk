# Hume Decisions Python SDK

The official Python client for the [Hume Decisions API](https://dev.hume.ai/decisions/docs/overview). The API answers questions about audio that you write in plain language. Send a recording with your questions, and each answer comes back as probabilities that estimate how human raters would respond, which your code can threshold, rank, or combine.

The SDK provides synchronous and asynchronous clients for the decisions and run endpoints, typed models for every question, answer, and response, and API key authentication.

1. [Documentation](https://dev.hume.ai/decisions/docs/overview)
2. [Python quickstart](https://dev.hume.ai/decisions/docs/quickstart/python)
3. [Writing questions](https://dev.hume.ai/decisions/docs/writing-questions)
4. [Primitives](https://dev.hume.ai/decisions/docs/primitives)
5. [Interpreting answers](https://dev.hume.ai/decisions/docs/interpreting-answers)
6. [Runs guide](https://dev.hume.ai/decisions/docs/runs)
7. [Cookbooks](https://dev.hume.ai/decisions/cookbooks/overview)
8. [API reference](https://dev.hume.ai/decisions/reference)

## Requirements

Python 3.10 or later.

## Installation

```bash
pip install hume-decisions
```

## Authentication

Every request carries your API key in the `X-Hume-Api-Key` header. Pass the key to the client, or set the `HUME_API_KEY` environment variable and construct the client without arguments.

```python
from hume_decisions import DecisionsClient

client = DecisionsClient(api_key="YOUR_API_KEY")
```

Keep API keys on a server. To use the API from a web page, relay through your server. See [Authentication](https://dev.hume.ai/decisions/docs/authentication).

## Preparing audio

Each request takes one MP3 or WAV file up to 5 minutes long. Send one request per recording, and split a longer recording as described in [Long recordings](#long-recordings).

The model hears audio at 16 kHz in one channel: the API resamples every file and mixes its channels into one before the model processes it. To make uploads smaller, the SDK does the same to a WAV file before sending it, in `create` and in the chunked methods. Resampled audio is written as 32-bit floating point, the format the API converts every file to, so loud audio is not clipped. `downmix_strategy` sets how:

1. `"mix"`, the default, resamples audio above 16 kHz to 16 kHz and mixes two channels into one the way the API does, so the model hears the same audio apart from rounding. A file that this would not make smaller, such as 16-bit mono at 16 kHz, is sent as it is, and so is a file with more than two channels, because the API weighs those channels by the speaker positions its header declares.
2. `"first_channel"` keeps only the first channel of a file with more than one. The channel is copied exactly in the file's own sample format when that is no larger than resampling it, as it is at 16 kHz or below for every sample format but 64-bit floating point, and is otherwise converted the same way as by `"mix"`. Use it to judge one speaker of a call who is on the first channel; for any other channel, send that channel as its own file.
3. `None` sends the file as it is.

MP3 files and WAV files the SDK cannot read, such as compressed WAV, are always sent as they are, and so is every file sent with `with_raw_response.create`.

## Quickstart

This request asks two questions about a simulated call with a voice AI agent: whether the agent talks over the caller, and how natural the agent sounds on a five-point scale.

```python
from hume_decisions import Answer_Noul, Answer_Score, DecisionsClient, DecisionsRequest, Question_Noul, Question_Score

client = DecisionsClient()

with open("test_call.wav", "rb") as audio:
    response = client.decisions.create(
        file=("test_call.wav", audio, "audio/wav"),
        request=DecisionsRequest(
            context="A simulated test call in which a voice AI agent helps a caller reschedule an appointment. Both voices are synthetic, and the agent speaks first.",
            questions={
                "talks_over_caller": Question_Noul(instructions="Does the agent talk over the caller?"),
                "agent_naturalness": Question_Score(
                    instructions="How natural does the agent sound?",
                    criteria=["Not at all", "2", "3", "4", "Extremely"],
                ),
            },
        ),
    )

for question_id, answer in response.answers.items():
    if isinstance(answer, Answer_Noul):
        print(f"{question_id}: {answer.noul:.0%} of raters would answer yes")
    elif isinstance(answer, Answer_Score):
        most_likely, _ = max(answer.probabilities.items(), key=lambda item: item[1])
        print(f"{question_id}: mean level {answer.score:.2f}, most often {answer.legend[most_likely]!r}")
```

The tuple sets the file part's filename and content type; for an MP3 file, pass `audio/mpeg`. Each answer comes back under the ID you gave its question, with the `run_id` of the model run that produced the answers. A probability estimates the share of human raters who would give that answer, so a `noul` near 0.5 means raters would split, not that the model is unsure. `score` is the mean level, counted from 0, and `legend` maps each level index to the text you gave it. The [Python quickstart](https://dev.hume.ai/decisions/docs/quickstart/python) walks through the request and the response.

## Questions

Each question is keyed by an ID you choose, and its model names its primitive, which sets the `criteria` it takes and the form of its answer. One request can mix primitives.

| Primitive | Model | Asks | `criteria` |
|---|---|---|---|
| `noul` | `Question_Noul` | Whether something is true of the audio. | Optional. A `NoulCriteria` with what counts as `true` and as `false`. |
| `choice` | `Question_Choice` | Which of several options applies. | Two or more `Criterion` options, each with a unique `name` and an optional `description`. |
| `score` | `Question_Score` | Where the audio falls on an ordered scale. | Two or more levels, from lowest to highest. |
| `moment` | `Question_Moment` | When an event occurs. | None. |

```python
from hume_decisions import Criterion, DecisionsRequest, Question_Choice, Question_Moment, Question_Noul, Question_Score

request = DecisionsRequest(
    context="A simulated test call in which a voice AI agent helps a caller reschedule an appointment. Both voices are synthetic, and the agent speaks first.",
    questions={
        "talks_over_caller": Question_Noul(instructions="Does the agent talk over the caller?"),
        "interruptions": Question_Moment(instructions="When does the agent interrupt the caller?"),
        "agent_naturalness": Question_Score(
            instructions="How natural does the agent sound?",
            criteria=["Not at all", "2", "3", "4", "Extremely"],
        ),
        "agent_tone": Question_Choice(
            instructions="Which best describes the tone of the agent?",
            criteria=[
                Criterion(name="warm", description="Warm and friendly toward the caller"),
                Criterion(name="neutral", description="Polite but impersonal"),
                Criterion(name="curt", description="Short or dismissive with the caller"),
            ],
        ),
    },
)
```

All the questions in a request are answered in one model inference, so an answer can change when other questions are added, removed, or reordered. Ask every question about a recording in one request, and to compare answers across recordings, send the same questions in the same order. [Writing questions](https://dev.hume.ai/decisions/docs/writing-questions) covers how to word instructions and criteria.

## Reading answers

`response.answers` maps each question ID to an answer whose model matches the question's primitive. Check the model with `isinstance`, which also lets a type checker narrow the answer to that model's fields. Every model is exported from `hume_decisions`.

| Primitive | Model | Fields |
|---|---|---|
| `noul` | `Answer_Noul` | `noul`, the share of raters who would answer yes. |
| `choice` | `Answer_Choice` | `choice`, the option most raters would choose; `probabilities`, keyed by option name; and `confidence`. |
| `score` | `Answer_Score` | `score`, the mean level; `probabilities` and `legend`, keyed by level index as a string; and `confidence`. |
| `moment` | `Answer_Moment` | `moments`, each with `audio_start_ms`, `audio_end_ms`, and `probability`. An empty list means the event was not found. |

```python
from hume_decisions import Answer_Choice, Answer_Moment, Answer_Noul, Answer_Score

for question_id, answer in response.answers.items():
    if isinstance(answer, Answer_Noul):
        print(f"{question_id}: {answer.noul:.2f}")
    elif isinstance(answer, Answer_Choice):
        print(f"{question_id}: {answer.choice}, confidence {answer.confidence:.2f}")
    elif isinstance(answer, Answer_Score):
        print(f"{question_id}: {answer.score:.2f} of {len(answer.legend) - 1}, confidence {answer.confidence:.2f}")
    elif isinstance(answer, Answer_Moment):
        for moment in answer.moments:
            print(f"{question_id}: {moment.audio_start_ms} to {moment.audio_end_ms} ms, {moment.probability:.2f}")
```

`confidence` measures how much raters would agree, not how certain the model is. A `moment` answer gives times but no share of raters, so to ask whether an event occurs as well as when, add a `noul` question. [Primitives](https://dev.hume.ai/decisions/docs/primitives) defines every field, and [Interpreting answers](https://dev.hume.ai/decisions/docs/interpreting-answers) covers setting thresholds for acting on them.

## Async client

`AsyncDecisionsClient` has the same interface with `async` methods. Use it to send requests for several recordings concurrently.

```python
import asyncio

from hume_decisions import AsyncDecisionsClient


async def main() -> None:
    client = AsyncDecisionsClient()

    with open("test_call.wav", "rb") as audio:
        response = await client.decisions.create(file=("test_call.wav", audio, "audio/wav"), request=request)
    print(f"run {response.run_id}: {len(response.answers)} answers")


asyncio.run(main())
```

## Long recordings

`decisions.create_chunked` cuts a recording into chunks no longer than `max_chunk_ms` and sends each chunk in its own `create` request, one at a time and in order, with the same questions, context, and `request_options`. The model hears only the chunk it answers about, and each chunk gets its own response, `run_id`, and bill. Nothing combines the answers across chunks.

```python
from hume_decisions import Answer_Moment, DecisionsClient, DecisionsRequest, Question_Moment, Question_Noul

client = DecisionsClient()

with open("long_call.wav", "rb") as audio:
    result = client.decisions.create_chunked(
        file=("long_call.wav", audio, "audio/wav"),
        request=DecisionsRequest(
            context="A call in which a voice AI agent helps a caller reschedule an appointment.",
            questions={
                "agent_glitches": Question_Moment(instructions="Where does the agent glitch, slur or sound robotic?"),
                "talks_over_caller": Question_Noul(instructions="Does the agent talk over the caller?"),
            },
        ),
        max_chunk_ms=300_000,
    )

for chunk in result.chunks:
    if chunk.response is None:
        print(f"chunk {chunk.index} ({chunk.start_ms} to {chunk.end_ms} ms): {chunk.status}, {chunk.error}")
        continue
    answer = chunk.response.answers["agent_glitches"]
    if isinstance(answer, Answer_Moment):
        for moment in answer.moments:
            # Moments are measured from the start of the chunk.
            print(f"{chunk.start_ms + moment.audio_start_ms} to {chunk.start_ms + moment.audio_end_ms} ms")
```

Each chunk's `start_ms` and `end_ms` give where it starts and ends in the recording, so add its `start_ms` to a moment's times to place the moment in the recording. Pass `start_ms` and `end_ms` to `create_chunked` to send only part of the recording.

### Chunk length

`max_chunk_ms` has no default, because it decides how many requests the job sends, and each request is billed for its own audio, context, and questions.

1. Use 300000, the longest chunk the API accepts, for the fewest requests.
2. Use shorter chunks for answers about shorter stretches of the call. A `noul` or `score` answer describes its whole chunk, so with 60000 ms chunks each answer covers at most one minute.

By default, the recording is cut into the fewest chunks that fit, of equal length, so the last chunk is not a short remainder: a 32-minute call at 300000 ms becomes 7 chunks of about 4 minutes 34 seconds, not 6 of 5 minutes and 1 of 2 minutes. Pass `split="fixed"` when chunks must be the same length in every recording, for example to count per 5 minutes across calls. Every chunk is then `max_chunk_ms` long except the last, which holds what remains and is billed like any other.

A WAV recording takes its duration divided by `max_chunk_ms` requests, rounded up: a 32-minute call takes 7 at 300000 ms and 32 at 60000 ms. Each chunk must also fit the API's limit of 25,000,000 bytes per request. With the default `downmix_strategy`, 5 minutes of mono or stereo WAV always fits. A file that does not fit is cut into shorter chunks that do, so 16-bit stereo sent with `downmix_strategy=None` is cut into chunks of at most 136054 ms at 44.1 kHz and 124999 ms at 48 kHz. With `split="fixed"`, `create_chunked` instead raises a `ValueError` naming the largest `max_chunk_ms` that fits, before it sends anything.

An MP3 recording can take more requests, because each chunk repeats a little audio from the chunk before, as described under [Where cuts fall](#where-cuts-fall). At 300000 ms this adds at most one request for recordings shorter than several hours, and only when the recording is at or just under a multiple of 5 minutes. Shorter chunks lose a larger share to repeated audio, so they can add several.

### Writing questions for chunks

The model answers about each chunk as if it were the whole recording, so write the context and questions to hold for any part of the call:

1. Describe in the context what is true throughout the call, such as who is speaking and what the call is for. A detail about one part of the call, such as "the agent speaks first", is true of the first chunk only, and in every other chunk it describes audio the model cannot hear.
2. Ask about what can be heard within a single chunk. "Does the agent talk over the caller?" can be answered from any stretch of the call. "Is the caller's appointment rescheduled by the end of the call?" cannot, because the end of the call is in only one chunk.

### Combining answers

The answers are not combined for you, because the right way to combine them depends on the question. To ask whether something happens anywhere in the call, take the highest `noul` across the chunks:

```python
from hume_decisions import Answer_Noul

if result.complete:
    answers = [response.answers["talks_over_caller"] for response in result.responses if response is not None]
    highest = max(answer.noul for answer in answers if isinstance(answer, Answer_Noul))
    print(f"In the chunk most likely to show it, {highest:.0%} of raters would say the agent talks over the caller.")
```

To ask how much of the call something describes, count the chunks whose `noul` passes a threshold instead. A `moment` answer needs no combining once its times are placed in the recording.

### Where cuts fall

1. A WAV file is cut between samples, after it is mixed down.
2. An MP3 file is cut between frames, without being decoded, so chunks can differ in length by a frame or more; a frame lasts 24 to 72 ms. A frame can depend on the frames before it, so every chunk after the first starts with frames repeated from the end of the chunk before, usually a fraction of a second and up to about 2 seconds at the lowest bitrates. They count toward `max_chunk_ms` and are included in the chunk's `start_ms`, so a moment near a cut can appear in both chunks. Most MP3 files also start with a few tens of milliseconds of silence that the encoder added and players skip. Times are measured from where a player starts the file, so the first chunk's `start_ms` can be below 0.

In both formats, an event that crosses a cut, such as the agent talking over the caller for a second or two around it, is heard in part by each chunk. A `moment` question can return it as two shorter moments, as one that ends or starts at the cut, or not at all, and a `noul` answer about it can come out lower in both chunks than for audio that holds the whole event. The frames an MP3 chunk repeats are there so it can be decoded, and cannot be relied on to cover an event at a cut.

### Failures and resuming

A request the API rejects, or whose connection fails, does not raise. Its error is recorded on the chunk, and `create_chunked` returns the responses it received, so check `result.complete` before relying on the result.

1. A chunk whose audio cannot be decoded (`audio_decode_failed`) fails on its own, and the remaining chunks are still sent, as long as an earlier chunk succeeded. Until one has, the whole file is the likely cause, so the job stops.
2. Any other error stops the job, such as running out of tokens (402), an invalid request, or a dropped connection. The chunks after it have the status `not_attempted`: no request was sent for them, so they were not billed.
3. Rate limits and server errors are retried first, as for any request, so they stop the job only once the retries run out. Pass `max_retries` in `request_options` to allow more.

To resume a job that stopped, call `create_chunked` again with `start_ms=result.resume_ms` and the same `end_ms`. `resume_ms` is where the chunk before the one that stopped the job ends, or `None` when no chunk stopped the job. The new job sends the audio of the chunk that stopped the job and everything after it, in no more chunks than the stopped job had left. It does not resend earlier chunks that failed because their audio could not be decoded, so its `complete` covers only the range it sent. With the default `split="even"`, its cuts after the first chunk can fall a few MP3 frames or a millisecond of WAV from the stopped job's. A WAV chunk's `start_ms` is rounded to the millisecond, so the resumed job can skip or repeat a few samples where it starts.

### Memory

The whole recording is read into memory, and a WAV file is mixed down, before the first request, so a file that cannot be split is rejected before anything is billed. An hour of 16-bit stereo WAV at 48 kHz is about 690 MB, plus about 230 MB for the mixed-down copy.

### Saving results as they arrive

`create_chunked` returns only once every chunk is done, so if your process stops partway through a long job, the responses it received are lost, although they were billed. `decisions.iter_create_chunked` cuts and sends the chunks the same way, but yields each chunk's result as soon as its request is done, so you can save each response before the next chunk is sent:

```python
from hume_decisions import DecisionsClient, DecisionsRequest, Question_Moment

client = DecisionsClient()
request = DecisionsRequest(
    questions={"agent_glitches": Question_Moment(instructions="Where does the agent glitch, slur or sound robotic?")},
)

with open("long_call.wav", "rb") as audio:
    for chunk in client.decisions.iter_create_chunked(
        file=("long_call.wav", audio, "audio/wav"), request=request, max_chunk_ms=300_000
    ):
        if chunk.response is not None:
            with open(f"chunk_{chunk.index}_{chunk.start_ms}.json", "w") as saved:
                saved.write(chunk.response.json())
```

To resume such a job, keep the results it yields.

1. When a chunk stopped the job, `ChunkedDecisionsResult(results, start_ms=start_ms).resume_ms` gives where to resume, with `ChunkedDecisionsResult` from `hume_decisions.audio_helpers` and the `start_ms` the job was given.
2. When your process stopped instead, `resume_ms` is `None`. Resume at the `end_ms` of the last chunk you have a result for, not at the `start_ms` of the chunk after it, which for an MP3 file includes frames repeated from the chunk you already have.

In both cases, pass the same `end_ms` as the job that stopped. Chunks that failed because their audio could not be decoded are not sent again, including the last chunk you have a result for, so check your saved results for them.

With `AsyncDecisionsClient`, iterate with `async for`. The async iterator reads the recording when iteration starts, so keep the file open while iterating.

### Sending chunks yourself

`iter_audio_chunks` in `hume_decisions.audio_helpers` cuts a recording the same way, mixdown included, and yields each chunk's bytes with its `start_ms` and `end_ms`. Use it to send chunks some other way, such as several at a time with the async client:

```python
import asyncio

from hume_decisions import AsyncDecisionsClient
from hume_decisions.audio_helpers import iter_audio_chunks


async def main() -> None:
    client = AsyncDecisionsClient()
    limit = asyncio.Semaphore(4)

    async def send(chunk):
        async with limit:
            return await client.decisions.create(file=("chunk.wav", chunk.data, "audio/wav"), request=request)

    with open("long_call.wav", "rb") as audio:
        chunks = list(iter_audio_chunks(audio, max_chunk_ms=300_000))
    responses = await asyncio.gather(*(send(chunk) for chunk in chunks), return_exceptions=True)
    for chunk, response in zip(chunks, responses):
        print(chunk.index, chunk.start_ms, response)


asyncio.run(main())
```

Sending several chunks at once reaches rate limits sooner, and a failure does not stop the requests already in flight, which are still billed. `create_chunked` sends one chunk at a time for these reasons.

## Runs

Every successful decisions request is recorded as a run, identified by the `run_id` of its response. The run endpoints return a run's status, when it started and ended, and the request it evaluated. They do not return answers, so keep the answers you need from the response.

```python
from hume_decisions import DecisionsClient

client = DecisionsClient()

page = client.runs.list(status="ended", page_size=20)
for summary in page.runs:
    print(summary.run_id, summary.status, summary.started_at)

run = client.runs.get("019a1f3c-7d2e-7b41-9c58-2e6f4a1d8b03")
print(run.status, list(run.request.questions))
```

`runs.list` returns your organization's runs, newest first, one page at a time, and filters by `user_id`, `status`, `started_after`, and `started_before`. Pass a page's `next_cursor` back as `cursor`, with the same filters, to fetch the next one; the last page has none. The [runs guide](https://dev.hume.ai/decisions/docs/runs) covers run status and the filters.

## Errors

The client raises `hume_decisions.core.ApiError` when a request fails. `status_code` holds the HTTP status, and `body` holds the error body. When the service answers, `body` is an `Error` whose `code` says what went wrong and whose `param` points to where the error was found, when there is one. Hume's API gateway answers some requests before they reach the service: a 401 for a missing or invalid credential, a 429 for a rate limit, and a 404 for a path or method the API does not serve. Those carry only a `message`, and a `request_id` when an API key was rejected.

```python
from hume_decisions.core import ApiError

try:
    with open("test_call.wav", "rb") as audio:
        response = client.decisions.create(file=("test_call.wav", audio, "audio/wav"), request=request)
except ApiError as error:
    print(f"{error.status_code}: {error.body}")
```

The client retries a request that fails with a 408, 409, 429, or 5xx status up to 2 times, waiting as long as the response's `Retry-After` header asks when it has one. Pass `max_retries` to the client, or in `request_options` for one request, to change the limit. Requests time out after 60 seconds unless you pass a different `timeout` the same way.

[Errors](https://dev.hume.ai/decisions/docs/errors) lists every code and whether to retry.

## Contributing

Most of this SDK is generated by [Fern](https://buildwithfern.com) from the API definition, and edits to generated files are overwritten on the next generation. [CONTRIBUTING.md](https://github.com/HumeAI/hume-decisions-python-sdk/blob/main/CONTRIBUTING.md) explains how to build and test the project and how to add code that persists across regenerations. To report a problem, open an issue in this repository.

## License

[MIT](https://github.com/HumeAI/hume-decisions-python-sdk/blob/main/LICENSE)
