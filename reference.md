# Reference
## decisions
<details><summary><code>client.decisions.<a href="src/hume_audio_decisions/decisions/client.py">create</a>(...) -> DecisionsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Asks questions about one audio file and returns an answer to each, with the `run_id` of the model run that produced them. Each answer estimates how human raters would respond, so you choose the thresholds for acting on it.

All the questions in a request are answered in one model inference, so an answer can change when other questions are added, removed or reordered. To compare answers across recordings, send the same questions in the same order. To evaluate several files, send one request per file.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from hume_audio_decisions import DecisionsClient, DecisionsRequest, Question_Noul, Question_Choice, Criterion, Question_Score, Question_Span
from hume_audio_decisions.environment import DecisionsClientEnvironment

client = DecisionsClient(
    api_key="<value>",
    environment=DecisionsClientEnvironment.PRODUCTION,
)

client.decisions.create(
    file="example_file",
    request=DecisionsRequest(
        context="A simulated test call in which a voice AI agent helps a caller reschedule an appointment. Both voices are synthetic, and the agent speaks first.",
        questions={
            "talks_over_caller": Question_Noul(
                instructions="Does the agent talk over the caller?",
            ),
            "agent_tone": Question_Choice(
                instructions="Which best describes the tone of the agent?",
                criteria=[
                    Criterion(
                        name="warm",
                        description="Warm and friendly toward the caller",
                    ),
                    Criterion(
                        name="neutral",
                        description="Polite but impersonal",
                    ),
                    Criterion(
                        name="curt",
                        description="Short or dismissive with the caller",
                    )
                ],
            ),
            "agent_naturalness": Question_Score(
                instructions="How natural does the agent sound?",
                criteria=[
                    "Not at all",
                    "2",
                    "3",
                    "4",
                    "Extremely"
                ],
            ),
            "agent_glitches": Question_Span(
                instructions="Where does the agent glitch, slur or sound robotic?",
            )
        },
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `DecisionsRequest` 
    
</dd>
</dl>

<dl>
<dd>

**file:** `core.File` — The audio file, MP3 or WAV, in a part named `file`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.decisions.<a href="src/hume_audio_decisions/decisions/client.py">input_tokens</a>(...) -> InputTokensResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Estimates the input tokens of a decisions request from its questions and the duration of its audio. The body takes the same `context` and `questions` as the `request` part of a `POST /v0/decisions/file` request, plus `audio_duration_ms`, the duration of the audio file in milliseconds.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from hume_audio_decisions import DecisionsClient, Question_Noul, Question_Choice, Criterion, Question_Score, Question_Span
from hume_audio_decisions.environment import DecisionsClientEnvironment

client = DecisionsClient(
    api_key="<value>",
    environment=DecisionsClientEnvironment.PRODUCTION,
)

client.decisions.input_tokens(
    audio_duration_ms=135000,
    context="A simulated test call in which a voice AI agent helps a caller reschedule an appointment. Both voices are synthetic, and the agent speaks first.",
    questions={
        "talks_over_caller": Question_Noul(
            instructions="Does the agent talk over the caller?",
        ),
        "agent_tone": Question_Choice(
            instructions="Which best describes the tone of the agent?",
            criteria=[
                Criterion(
                    name="warm",
                    description="Warm and friendly toward the caller",
                ),
                Criterion(
                    name="neutral",
                    description="Polite but impersonal",
                ),
                Criterion(
                    name="curt",
                    description="Short or dismissive with the caller",
                )
            ],
        ),
        "agent_naturalness": Question_Score(
            instructions="How natural does the agent sound?",
            criteria=[
                "Not at all",
                "2",
                "3",
                "4",
                "Extremely"
            ],
        ),
        "agent_glitches": Question_Span(
            instructions="Where does the agent glitch, slur or sound robotic?",
        )
    },
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**audio_duration_ms:** `int` — The duration of the audio file the decisions request would send, in milliseconds. A duration under 1 second is rejected with `audio_too_short`, and one over 5 minutes with `audio_too_long`.
    
</dd>
</dl>

<dl>
<dd>

**questions:** `Questions` 
    
</dd>
</dl>

<dl>
<dd>

**context:** `typing.Optional[str]` — Background information the model considers alongside the audio, such as the setting or who is speaking.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## runs
<details><summary><code>client.runs.<a href="src/hume_audio_decisions/runs/client.py">list</a>(...) -> RunsPage</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Your organization's runs, newest first, one page at a time. The filters combine.

A run is one model inference. A successful decisions request is recorded as a run, which can take a moment to appear in this list, and under heavy load may not be recorded. A decisions request that fails before its input reaches the model, including every request rejected with a `4xx` status, has none.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from hume_audio_decisions import DecisionsClient
from hume_audio_decisions.environment import DecisionsClientEnvironment

client = DecisionsClient(
    api_key="<value>",
    environment=DecisionsClientEnvironment.PRODUCTION,
)

client.runs.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page_size:** `typing.Optional[int]` — How many runs to return.
    
</dd>
</dl>

<dl>
<dd>

**cursor:** `typing.Optional[str]` — The `next_cursor` of the previous page. Pass it back unchanged, with the same filters as the request that returned it.
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` — Only runs made by this user.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[RunStatus]` — Only runs with this status.
    
</dd>
</dl>

<dl>
<dd>

**started_before:** `typing.Optional[datetime.datetime]` — Only runs that started before this time, in RFC 3339 format.
    
</dd>
</dl>

<dl>
<dd>

**started_after:** `typing.Optional[datetime.datetime]` — Only runs that started at or after this time, in RFC 3339 format. Consecutive time ranges that share a boundary neither skip a run nor return one twice.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.runs.<a href="src/hume_audio_decisions/runs/client.py">get</a>(...) -> Run</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

The run's status, when it started and ended, the audio it received, and the request it evaluated. The run does not include the answers.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from hume_audio_decisions import DecisionsClient
from hume_audio_decisions.environment import DecisionsClientEnvironment

client = DecisionsClient(
    api_key="<value>",
    environment=DecisionsClientEnvironment.PRODUCTION,
)

client.runs.get(
    run_id="019a1f3c-7d2e-7b41-9c58-2e6f4a1d8b03",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**run_id:** `str` — The run's ID: the `run_id` of a decisions response. An ID that is not a UUID returns `404`, as an unknown one does.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

