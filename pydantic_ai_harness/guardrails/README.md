# Guardrails

Intercept unsafe user prompts before they reach the model, and unsafe model outputs before they reach the caller.

## The problem

Agents take unstructured input from users and return unstructured output to callers. Without a validation layer, a prompt injection attempt, PII-laden message, or off-topic question goes to the model as-is, and any output the model produces is returned verbatim. The framework does not reason about "this is unsafe to send" or "this is unsafe to show".

## The solution

Two capabilities, each backed by a callable you supply.

| Capability | Checks | When a guard returns `False` | When a guard raises |
|---|---|---|---|
| `InputGuard` | The user prompt before each model request | `SkipModelRequest` — the model call is skipped and `block_message` becomes the response for that step | The exception propagates out of the run |
| `OutputGuard` | The final run output | `OutputBlocked` is raised | The exception propagates out of the run |

The asymmetry is intentional. Blocking the input means no tokens are spent, so a graceful refusal is almost always what you want. Blocking the output means the model already generated a response you do not want exposed — raising forces the caller to decide what to do next.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import InputGuard, OutputGuard


def no_secrets(prompt: str) -> bool:
    return 'api_key' not in prompt.lower()


def no_pii(output: str) -> bool:
    return 'SSN' not in output


agent = Agent(
    'openai:gpt-4.1',
    capabilities=[
        InputGuard(guard=no_secrets),
        OutputGuard(guard=no_pii),
    ],
)
```

Both guards accept async callables too:

```python
async def check_with_moderation_api(prompt: str) -> bool:
    response = await client.moderations.create(input=prompt)
    return not response.results[0].flagged


agent = Agent(
    'openai:gpt-4.1',
    capabilities=[InputGuard(guard=check_with_moderation_api)],
)
```

## Parallel input guards

When a guard is slow (an LLM-based classifier or a network call), running it in sequence before every model request adds latency to every turn. Set `parallel=True` to race the guard against the model call. The model call is cancelled immediately if the guard reports a violation.

```python
InputGuard(guard=slow_async_classifier, parallel=True)
```

For fast local checks (regex, keyword lookup, a small classifier) sequential is usually fine — the overhead is measured in microseconds and the wiring is simpler.

## Customising the block message

```python
InputGuard(
    guard=no_secrets,
    block_message='This request looks like it contains credentials. Please rephrase.',
)
```

The text is returned as the model response for that step, so the caller sees a normal completion rather than an exception. Multi-turn agents can continue the conversation from there.

## Hard-fail path

Returning `False` from a guard is the graceful path. If you want the caller to see an exception instead, raise from the guard:

```python
from pydantic_ai_harness import InputBlocked


def strict_guard(prompt: str) -> bool:
    if contains_credentials(prompt):
        raise InputBlocked('credentials detected')
    return True
```

Any exception raised by the guard propagates as-is — you can use `InputBlocked` / `OutputBlocked` from this module or your own exception types.

## API

```python
InputGuard(
    guard: Callable[[str], bool | Awaitable[bool]],
    parallel: bool = False,
    block_message: str = 'Request blocked by input guardrail.',
)

OutputGuard(
    guard: Callable[[str], bool | Awaitable[bool]],
    block_message: str = 'Output blocked by output guardrail.',
)
```

## Relationship to `pydantic-ai-shields`

`pydantic-ai-shields` provides opinionated implementations on top of these primitives (prompt-injection detectors, PII scrubbers, keyword blocklists, etc.). Use the guardrails here when you want to plug in your own validation logic; reach for shields when you need a batteries-included detector.
