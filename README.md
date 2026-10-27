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

Each request takes one audio file, MP3 or WAV, up to 5 minutes long. To evaluate several recordings, send one request per file. The service mixes all channels into one, so to judge one speaker in a call alone, send that speaker's channel as its own file.

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

The tuple gives the file part a filename and a content type; for an MP3 file, pass `audio/mpeg`. The response returns each answer under the ID you gave its question, with the `run_id` of the model run that produced the answers. A probability estimates the share of human raters who would give that answer, so a `noul` near 0.5 means raters would split on the audio, not that the model is unsure. `score` is the mean level, counted from 0, and `legend` maps each level index back to the text you gave it. The [Python quickstart](https://dev.hume.ai/decisions/docs/quickstart/python) walks through the request and the response.

## Questions

Each question is keyed by an ID you choose, and its model names its primitive, which sets the `criteria` it takes and the form of its answer. One request can mix primitives.

| Primitive | Model | Asks | `criteria` |
|---|---|---|---|
| `noul` | `Question_Noul` | Whether something is true of the audio. | Optional. A `NoulCriteria` with what counts as `true` and as `false`. |
| `choice` | `Question_Choice` | Which of several options applies. | Two or more `Criterion` options, each with a unique `name` and an optional `description`. |
| `score` | `Question_Score` | Where the audio falls on an ordered scale. | Two or more levels, from lowest to highest. |
| `span` | `Question_Span` | When an event occurs. | None. |

```python
from hume_decisions import Criterion, DecisionsRequest, Question_Choice, Question_Noul, Question_Score, Question_Span

request = DecisionsRequest(
    context="A simulated test call in which a voice AI agent helps a caller reschedule an appointment. Both voices are synthetic, and the agent speaks first.",
    questions={
        "talks_over_caller": Question_Noul(instructions="Does the agent talk over the caller?"),
        "interruptions": Question_Span(instructions="When does the agent interrupt the caller?"),
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
| `span` | `Answer_Span` | `spans`, each with `audio_start_ms`, `audio_end_ms`, and `probability`. An empty list means the event was not found. |

```python
from hume_decisions import Answer_Choice, Answer_Noul, Answer_Score, Answer_Span

for question_id, answer in response.answers.items():
    if isinstance(answer, Answer_Noul):
        print(f"{question_id}: {answer.noul:.2f}")
    elif isinstance(answer, Answer_Choice):
        print(f"{question_id}: {answer.choice}, confidence {answer.confidence:.2f}")
    elif isinstance(answer, Answer_Score):
        print(f"{question_id}: {answer.score:.2f} of {len(answer.legend) - 1}, confidence {answer.confidence:.2f}")
    elif isinstance(answer, Answer_Span):
        for span in answer.spans:
            print(f"{question_id}: {span.audio_start_ms} to {span.audio_end_ms} ms, {span.probability:.2f}")
```

`confidence` measures how much raters would agree, not how certain the model is. A `span` answer gives times but no share of raters, so to ask whether an event occurs as well as when, add a `noul` question. [Primitives](https://dev.hume.ai/decisions/docs/primitives) defines every field, and [Interpreting answers](https://dev.hume.ai/decisions/docs/interpreting-answers) covers setting thresholds for acting on them.

## Counting input tokens

`decisions.input_tokens` takes the same file and request as `decisions.create` and returns the number of input tokens the request would be billed for, without evaluating the audio or recording a run. Hume counts them from the audio duration, the context, and the questions.

```python
with open("test_call.wav", "rb") as audio:
    count = client.decisions.input_tokens(file=("test_call.wav", audio, "audio/wav"), request=request)
print(count.input_tokens)
```

The request is validated the same way as a decisions request and fails with the same errors.

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

The client raises `hume_decisions.core.ApiError` when a request fails. `status_code` holds the HTTP status, and `body` holds the error body. For every status except 401, `body` is an `Error` whose `code` says what went wrong and whose `param` points to the field it concerns, when there is one. A 401 carries only a `message`, and a `request_id` when an API key was rejected.

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
