"""Tests for the `OutputGuard` capability."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import OutputBlocked, OutputGuard
from pydantic_ai_harness.guardrails import GuardrailError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_guard_allows_safe_output():
    agent = Agent(
        TestModel(custom_output_text='harmless reply'),
        capabilities=[OutputGuard[None](guard=lambda out: 'SSN' not in out)],
    )
    result = await agent.run('hello')
    assert result.output == 'harmless reply'


async def test_guard_blocks_unsafe_output():
    agent = Agent(
        TestModel(custom_output_text='leaks SSN 123-45-6789'),
        capabilities=[OutputGuard[None](guard=lambda out: 'SSN' not in out, block_message='contains SSN')],
    )
    with pytest.raises(OutputBlocked, match='contains SSN'):
        await agent.run('hello')


async def test_async_guard_awaited():
    async def guard(output: str) -> bool:
        await asyncio.sleep(0)
        return 'bad' not in output

    agent = Agent(
        TestModel(custom_output_text='ok reply'),
        capabilities=[OutputGuard[None](guard=guard)],
    )
    assert (await agent.run('prompt')).output == 'ok reply'

    agent_bad = Agent(
        TestModel(custom_output_text='bad reply'),
        capabilities=[OutputGuard[None](guard=guard)],
    )
    with pytest.raises(OutputBlocked):
        await agent_bad.run('prompt')


async def test_guard_raising_propagates():
    def guard(_: str) -> bool:
        raise RuntimeError('guard exploded')

    agent = Agent(
        TestModel(custom_output_text='anything'),
        capabilities=[OutputGuard[None](guard=guard)],
    )
    with pytest.raises(RuntimeError, match='guard exploded'):
        await agent.run('hello')


def test_output_blocked_is_guardrail_error():
    assert issubclass(OutputBlocked, GuardrailError)
