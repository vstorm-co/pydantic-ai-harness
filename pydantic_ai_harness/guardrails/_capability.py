"""Input and output guardrail capabilities.

`InputGuard` intercepts each model request and lets a user-supplied callable
decide whether the current user prompt is safe to send to the model. A guard
that returns `False` is treated as a graceful refusal: the LLM call is
skipped via [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest] and
a canned message becomes the model response for that step. A guard that raises
propagates the exception so the caller can observe a hard failure.

`OutputGuard` runs once the run completes and validates the final output.
A guard that returns `False` raises
[`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked].

Both guards accept sync or async callables.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.guardrails._exceptions import OutputBlocked

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.run import AgentRunResult


GuardrailFunc = Callable[[str], bool | Awaitable[bool]]
"""Signature of the callable passed to `InputGuard` / `OutputGuard`.

The callable receives the text to validate and returns `True` when safe.
It may be sync or async. Raising an exception is treated as a hard failure
and propagates up to the caller.
"""


async def _evaluate(guard: GuardrailFunc, value: str) -> bool:
    """Call `guard` and await it if it returned an awaitable."""
    result = guard(value)
    if inspect.isawaitable(result):
        return await result
    return result


def _extract_prompt(ctx: RunContext[AgentDepsT], messages: list[Any]) -> str | None:
    """Return the text of the most recent user prompt, or `None` if absent.

    Prefers `ctx.prompt` (set at run start) and falls back to scanning the
    message history for the last [`UserPromptPart`][pydantic_ai.messages.UserPromptPart]
    so that sub-agent calls or resumed runs without a fresh prompt still work.
    """
    if ctx.prompt is not None:
        return ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt)
    for message in reversed(messages):
        parts = getattr(message, 'parts', None)
        if not parts:
            continue
        for part in reversed(parts):
            if isinstance(part, UserPromptPart):
                return part.content if isinstance(part.content, str) else str(part.content)
    return None


@dataclass
class InputGuard(AbstractCapability[AgentDepsT]):
    """Validate the user prompt before it reaches the model.

    The `guard` callable receives the prompt text and returns `True` when
    the input is safe. Returning `False` triggers a graceful refusal: the
    current model request is short-circuited via
    [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest] with
    `block_message` as the response text, so the agent returns cleanly to
    the caller. Raising an exception from the guard propagates it as-is.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import InputGuard


    def no_secrets(prompt: str) -> bool:
        return 'api_key' not in prompt.lower()


    agent = Agent('openai:gpt-4.1', capabilities=[InputGuard(guard=no_secrets)])
    ```

    Set `parallel=True` to start the guard alongside the model call. The
    handler is cancelled as soon as the guard reports a violation, which saves
    tokens when the guard is slower than the provider round-trip.
    """

    guard: GuardrailFunc
    """Callable that returns `True` when the prompt is safe to send to the model."""

    parallel: bool = False
    """Run the guard concurrently with the model request and cancel the model call on failure."""

    block_message: str = 'Request blocked by input guardrail.'
    """Text returned as the model response when the guard trips gracefully."""

    def _blocked_response(self) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=self.block_message)])

    async def _run_guard(self, prompt: str) -> None:
        """Evaluate the guard and raise `SkipModelRequest` on failure."""
        if not await _evaluate(self.guard, prompt):
            raise SkipModelRequest(self._blocked_response())

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Check the prompt before the model call in sequential mode."""
        if self.parallel:
            return request_context
        prompt = _extract_prompt(ctx, list(request_context.messages))
        if prompt is None:
            return request_context
        await self._run_guard(prompt)
        return request_context

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Run the guard alongside the model call when `parallel=True`."""
        if not self.parallel:
            return await handler(request_context)
        prompt = _extract_prompt(ctx, list(request_context.messages))
        if prompt is None:
            return await handler(request_context)
        async def run_handler() -> ModelResponse:
            return await handler(request_context)

        guard_task: asyncio.Task[None] = asyncio.create_task(self._run_guard(prompt))
        handler_task: asyncio.Task[ModelResponse] = asyncio.create_task(run_handler())
        try:
            done, _ = await asyncio.wait(
                [guard_task, handler_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if guard_task in done:
                guard_exc = guard_task.exception()
                if guard_exc is not None:
                    handler_task.cancel()
                    raise guard_exc
                return await handler_task
            # Handler finished first: if it raised, propagate and cancel the guard.
            handler_exc = handler_task.exception()
            if handler_exc is not None:
                guard_task.cancel()
                raise handler_exc
            # Handler succeeded; still need the guard verdict before committing the response.
            await guard_task
            return handler_task.result()
        finally:
            if not guard_task.done():
                guard_task.cancel()
            if not handler_task.done():
                handler_task.cancel()


@dataclass
class OutputGuard(AbstractCapability[AgentDepsT]):
    """Validate the final agent output.

    The `guard` callable receives the stringified run output and returns
    `True` when the output is safe to expose to the caller. Returning
    `False` raises
    [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked] with
    `block_message`. Raising an exception from the guard propagates it.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import OutputGuard


    def no_pii(output: str) -> bool:
        return 'SSN' not in output


    agent = Agent('openai:gpt-4.1', capabilities=[OutputGuard(guard=no_pii)])
    ```
    """

    guard: GuardrailFunc
    """Callable that returns `True` when the output is safe."""

    block_message: str = 'Output blocked by output guardrail.'
    """Message attached to `OutputBlocked` when the guard trips."""

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Validate `result.output` and raise `OutputBlocked` on failure."""
        output = str(result.output)
        if not await _evaluate(self.guard, output):
            raise OutputBlocked(self.block_message)
        return result
