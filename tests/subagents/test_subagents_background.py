"""Tests for background sub-agent delegation and parent-child communication."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import CallDeferred, ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.subagents import SubAgent, SubAgents, SubAgentToolset

pytestmark = pytest.mark.anyio

_TASK_ID_RE = re.compile(r"task '([0-9a-f]{8})'")


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _echo_child(name: str, text: str = 'child result') -> Agent[object, str]:
    """A child agent that immediately replies with `text`."""
    return Agent(TestModel(custom_output_text=text), name=name, description=f'{name} agent')


def _gated_child(
    name: str, gate: asyncio.Event, started: asyncio.Event | None = None, reply: str = 'gated result'
) -> Agent[object, str]:
    """A child agent whose model blocks until `gate` is set, then replies with `reply`."""

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if started is not None:
            started.set()
        await gate.wait()
        return ModelResponse(parts=[TextPart(reply)])

    return Agent(FunctionModel(model_fn), name=name, description=f'{name} agent')


def _raising_child(name: str, exc: Exception) -> Agent[object, str]:
    """A child agent whose model raises `exc`."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise exc

    return Agent(FunctionModel(model_fn), name=name, description=f'{name} agent')


def _asking_child(name: str, question: str = 'which database?') -> Agent[object, str]:
    """A child agent that asks its parent one question, then echoes the answer."""
    asked = {'done': False}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if asked['done']:
            returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
            answer = returns[0].content if returns else 'no answer'
            return ModelResponse(parts=[TextPart(f'answer: {answer}')])
        asked['done'] = True
        return ModelResponse(parts=[ToolCallPart('ask_parent', {'question': question}, tool_call_id='q1')])

    return Agent(FunctionModel(model_fn), name=name, description=f'{name} agent')


def _driver_parent(capability: SubAgents[object], toolset: FunctionToolset[object]) -> Agent[object, str]:
    """A parent whose model calls the `drive` tool once, then finishes.

    The drive tool runs test orchestration code inside a live parent run, which is
    the supported way to exercise the task-management tools without importing
    private helpers: `capability.get_toolset()` shares its run-scoped state with
    the toolset instance the run itself registered.
    """
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(parts=[ToolCallPart('drive', {}, tool_call_id='d1')])
        return ModelResponse(parts=[TextPart('parent done')])

    return Agent(FunctionModel(model_fn), toolsets=[toolset], capabilities=[capability])


def _toolset_of(capability: SubAgents[object]) -> SubAgentToolset[object]:
    toolset = capability.get_toolset()
    assert isinstance(toolset, SubAgentToolset)
    return cast(SubAgentToolset[object], toolset)


async def _drive_test(
    capability: SubAgents[object],
    body: Any,
) -> AgentRunResult[str]:
    """Run `body(ctx)` inside a live parent run of an agent carrying `capability`."""
    driver: FunctionToolset[object] = FunctionToolset(id='test-driver')

    async def drive(ctx: RunContext[object]) -> str:
        """Drive the test.

        Args:
            ctx: The run context.
        """
        await body(ctx)
        return 'driven'

    driver.add_function(drive)
    parent = _driver_parent(capability, driver)
    return await parent.run('go')


def _injected_texts(result: AgentRunResult[Any]) -> list[str]:
    """The text of every injected (non-initial) user prompt part in the history."""
    texts: list[str] = []
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content != 'go':
                texts.append(part.content)
    return texts


async def _wait_for_status(
    toolset: SubAgentToolset[object], ctx: RunContext[object], task_id: str, status: str
) -> None:
    """Poll `check_task` until the task reports `status` (bounded)."""
    for _ in range(200):
        report = await toolset.check_task(ctx, task_id)
        if f'Status: {status}' in report:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f'task {task_id} never reached status {status}')  # pragma: no cover


# --- delegation modes -----------------------------------------------------------------


async def test_background_delegation_completes_and_reports():
    gate = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate))], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        long_task = 'do it very thoroughly: ' + 'x' * 100  # long enough for the listing preview to clip
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', long_task, background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        seen['running'] = await toolset.check_task(ctx, task_id)
        gate.set()
        await _wait_for_status(toolset, ctx, task_id, 'completed')
        seen['completed'] = await toolset.check_task(ctx, task_id)
        seen['listing'] = await toolset.check_task(ctx)

    await _drive_test(capability, body)
    assert '[...]' in seen['listing']
    assert 'Status: running' in seen['running']
    assert 'Running for:' in seen['running']
    assert 'Status: completed' in seen['completed']
    assert 'Result: gated result' in seen['completed']
    assert 'Usage:' in seen['completed']
    assert 'worker (completed)' in seen['listing']


async def test_backgroundable_tool_still_runs_synchronously_by_default():
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(_echo_child('worker'))], agent_folders=None)
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        seen['result'] = await toolset.delegate_task_backgroundable(ctx, 'worker', 'do it')
        seen['empty'] = await toolset.check_task(ctx)

    await _drive_test(capability, body)
    assert seen['result'] == 'child result'
    assert seen['empty'] == 'No background tasks in this run.'


async def test_background_unknown_agent_is_a_model_retry():
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(_echo_child('worker'))], agent_folders=None)
    toolset = _toolset_of(capability)

    async def body(ctx: RunContext[object]) -> None:
        with pytest.raises(ModelRetry, match='Unknown sub-agent'):
            await toolset.delegate_task_backgroundable(ctx, 'ghost', 't', background=True)

    await _drive_test(capability, body)


async def test_sub_agent_background_false_refuses_background():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_echo_child('worker'), background=False)], agent_folders=None
    )
    toolset = _toolset_of(capability)

    async def body(ctx: RunContext[object]) -> None:
        with pytest.raises(ModelRetry, match='does not support background'):
            await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)

    await _drive_test(capability, body)


async def test_sub_agent_background_true_forces_background():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_echo_child('worker'), background=True)], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        seen['started'] = await toolset.delegate_task_backgroundable(ctx, 'worker', 't')
        task_id = _TASK_ID_RE.search(seen['started']).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'completed')

    await _drive_test(capability, body)
    assert 'Started background task' in seen['started']


async def test_background_respects_max_calls_budget():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_echo_child('worker'), max_calls=1)], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        first = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(first).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'completed')
        seen['second'] = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)

    await _drive_test(capability, body)
    assert 'budget' in seen['second']
    assert 'exhausted' in seen['second']


async def test_background_respects_max_concurrent_tasks():
    gate = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate))],
        agent_folders=None,
        notify=None,
        max_concurrent_tasks=1,
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        first = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(first).group(1)  # type: ignore[union-attr]
        seen['second'] = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        gate.set()
        await _wait_for_status(toolset, ctx, task_id, 'completed')

    await _drive_test(capability, body)
    assert 'background task budget' in seen['second']


@dataclass
class _Deps:
    label: str


async def test_deps_factory_applies_to_both_modes():
    factory_calls: list[str] = []

    def deps_factory(ctx: RunContext[_Deps]) -> _Deps:
        factory_calls.append(ctx.deps.label)
        return _Deps(label=f'child-of-{ctx.deps.label}')

    child_deps: list[str] = []
    child_toolset: FunctionToolset[_Deps] = FunctionToolset(id='probe')

    async def probe(ctx: RunContext[_Deps]) -> str:
        """Record the deps this child received.

        Args:
            ctx: The run context.
        """
        child_deps.append(ctx.deps.label)
        return 'probed'

    child_toolset.add_function(probe)
    child_model = TestModel(call_tools=['probe'], custom_output_text='ok')
    child: Agent[_Deps, str] = Agent(child_model, name='worker', deps_type=_Deps, toolsets=[child_toolset])

    capability: SubAgents[_Deps] = SubAgents(
        agents=[SubAgent(child)], agent_folders=None, deps_factory=deps_factory, notify=None
    )
    toolset = cast(SubAgentToolset[_Deps], capability.get_toolset())
    driver: FunctionToolset[_Deps] = FunctionToolset(id='test-driver')

    async def drive(ctx: RunContext[_Deps]) -> str:
        """Drive the test.

        Args:
            ctx: The run context.
        """
        await toolset.delegate_task_backgroundable(ctx, 'worker', 'sync one')
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 'bg one', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(
            cast(SubAgentToolset[object], toolset), cast(RunContext[object], ctx), task_id, 'completed'
        )
        return 'driven'

    driver.add_function(drive)
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(parts=[ToolCallPart('drive', {}, tool_call_id='d1')])
        return ModelResponse(parts=[TextPart('parent done')])

    parent: Agent[_Deps, str] = Agent(
        FunctionModel(model_fn), deps_type=_Deps, toolsets=[driver], capabilities=[capability]
    )
    await parent.run('go', deps=_Deps(label='root'))
    assert factory_calls == ['root', 'root']
    assert child_deps == ['child-of-root', 'child-of-root']


# --- worker outcomes ------------------------------------------------------------------


async def test_background_failure_is_soft_and_notified():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_raising_child('worker', ValueError('boom')))], agent_folders=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'failed')
        seen['report'] = await toolset.check_task(ctx, task_id)
        seen['wait'] = await toolset.wait_tasks(ctx, [task_id])

    result = await _drive_test(capability, body)
    assert 'Error: ValueError: boom' in seen['report']
    assert 'FAILED - ValueError: boom' in seen['wait']
    injected = _injected_texts(result)
    assert any('failed' in text and 'ValueError: boom' in text for text in injected)


async def test_background_failure_notification_uses_on_failure_override():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_raising_child('worker', ValueError('boom')), on_failure='use the cached copy')],
        agent_folders=None,
    )
    toolset = _toolset_of(capability)

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'failed')

    result = await _drive_test(capability, body)
    assert any('use the cached copy' in text for text in _injected_texts(result))


async def test_background_model_retry_failure_is_recorded():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_raising_child('worker', UnexpectedModelBehavior('weird')))],
        agent_folders=None,
        notify=None,
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'failed')
        seen['report'] = await toolset.check_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'weird' in seen['report']


async def test_background_own_usage_budget_fails_softly():
    child_toolset: FunctionToolset[object] = FunctionToolset(id='busy')

    async def busy(ctx: RunContext[object]) -> str:
        """Do one step of work.

        Args:
            ctx: The run context.
        """
        return 'step done'

    child_toolset.add_function(busy)
    child: Agent[object, str] = Agent(
        TestModel(call_tools=['busy'], custom_output_text='ok'), name='worker', toolsets=[child_toolset]
    )
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(child, usage_limits=UsageLimits(request_limit=1))], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'failed')
        seen['report'] = await toolset.check_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'reached its usage budget' in seen['report']


async def test_background_shared_usage_limit_is_reported_as_tree_budget():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_raising_child('worker', UsageLimitExceeded('over')))], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'failed')
        seen['report'] = await toolset.check_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'shared usage limit exceeded' in seen['report']


async def test_background_human_in_the_loop_is_unsupported():
    @dataclass
    class _DeferOnRun(AbstractCapability[object]):
        async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
            raise CallDeferred()

    child: Agent[object, str] = Agent(TestModel(), name='worker', capabilities=[_DeferOnRun()])
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(child)], agent_folders=None, notify=None)
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'failed')
        seen['report'] = await toolset.check_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'not supported in background mode' in seen['report']


async def test_background_timeout_cancels_the_child():
    gate = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate), timeout_seconds=0.05)], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'cancelled')
        seen['report'] = await toolset.check_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'timed out after 0.05s' in seen['report']


async def test_wrap_run_short_circuit_completes_the_task():
    inner_child = _echo_child('inner', 'precomputed')
    precomputed = await inner_child.run('anything')

    @dataclass
    class _ShortCircuit(AbstractCapability[object]):
        async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
            return precomputed

    child: Agent[object, str] = Agent(TestModel(), name='worker', capabilities=[_ShortCircuit()])
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(child)], agent_folders=None, notify=None)
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'completed')
        seen['report'] = await toolset.check_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'Result: precomputed' in seen['report']


# --- task management tools ------------------------------------------------------------


async def test_check_task_unknown_id_is_a_model_retry():
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(_echo_child('worker'))], agent_folders=None)
    toolset = _toolset_of(capability)

    async def body(ctx: RunContext[object]) -> None:
        with pytest.raises(ModelRetry, match='Unknown background task'):
            await toolset.check_task(ctx, 'deadbeef')

    await _drive_test(capability, body)


async def test_wait_tasks_all_and_any_modes():
    fast_gate = asyncio.Event()
    slow_gate = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[
            SubAgent(_gated_child('fast', fast_gate, reply='fast done')),
            SubAgent(_gated_child('slow', slow_gate, reply='slow done')),
        ],
        agent_folders=None,
        notify=None,
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        fast = await toolset.delegate_task_backgroundable(ctx, 'fast', 't', background=True)
        slow = await toolset.delegate_task_backgroundable(ctx, 'slow', 't', background=True)
        fast_id = _TASK_ID_RE.search(fast).group(1)  # type: ignore[union-attr]
        slow_id = _TASK_ID_RE.search(slow).group(1)  # type: ignore[union-attr]
        fast_gate.set()
        seen['any'] = await toolset.wait_tasks(ctx, [fast_id, slow_id], mode='any')
        slow_gate.set()
        seen['all'] = await toolset.wait_tasks(ctx, [fast_id, slow_id], mode='all')

    await _drive_test(capability, body)
    assert 'mode=any' in seen['any']
    assert '1/2 finished' in seen['any']
    assert '1 still running' in seen['any']
    assert 'COMPLETED\nfast done' in seen['any']
    assert 'mode=all' in seen['all']
    assert '2/2 finished' in seen['all']
    assert 'COMPLETED\nslow done' in seen['all']


async def test_wait_tasks_times_out_on_a_stuck_task():
    gate = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate))], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        seen['zero'] = await toolset.wait_tasks(ctx, [task_id], timeout=0)
        seen['short'] = await toolset.wait_tasks(ctx, [task_id], timeout=0.05)
        with pytest.raises(ModelRetry, match='Unknown background task'):
            await toolset.wait_tasks(ctx, ['deadbeef'])
        gate.set()
        await _wait_for_status(toolset, ctx, task_id, 'completed')

    await _drive_test(capability, body)
    assert '0/1 finished' in seen['zero']
    assert '(running)' not in seen['zero']  # status line, not the listing format
    assert 'running' in seen['zero']
    assert '0/1 finished' in seen['short']


async def test_wait_tasks_returns_early_when_a_task_asks_a_question():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_asking_child('worker'), can_ask_parent=True)], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        seen['wait'] = await toolset.wait_tasks(ctx, [task_id])
        seen['check'] = await toolset.check_task(ctx, task_id)
        seen['answer'] = await toolset.send_message_to_task(ctx, task_id, 'postgres')
        await _wait_for_status(toolset, ctx, task_id, 'completed')
        seen['done'] = await toolset.check_task(ctx, task_id)

    result = await _drive_test(capability, body)
    assert 'WAITING FOR YOUR ANSWER: which database?' in seen['wait']
    assert 'Question: which database?' in seen['check']
    assert seen['answer'] == "Answer delivered to task '" + seen['check'].splitlines()[0].split(': ')[1] + "'."
    assert 'Result: answer: postgres' in seen['done']
    assert any('worker asks: which database?' in text for text in _injected_texts(result))


async def test_ask_parent_times_out_into_best_judgment():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_asking_child('worker'), can_ask_parent=True)],
        agent_folders=None,
        notify=None,
        ask_timeout_seconds=0.05,
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await _wait_for_status(toolset, ctx, task_id, 'completed')
        seen['report'] = await toolset.check_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'did not respond in time' in seen['report']


async def test_send_message_steers_a_running_task():
    gate = asyncio.Event()
    started_event = asyncio.Event()
    steered: list[str] = []

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                if (
                    isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and 'parent agent' in part.content
                ):
                    steered.append(part.content)
        if not steered:
            started_event.set()
            await gate.wait()
        return ModelResponse(parts=[TextPart(f'saw {len(steered)} steering message(s)')])

    child: Agent[object, str] = Agent(FunctionModel(model_fn), name='worker', description='steerable')
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(child)], agent_folders=None, notify=None)
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await started_event.wait()
        seen['sent'] = await toolset.send_message_to_task(ctx, task_id, 'focus on caching')
        gate.set()
        await _wait_for_status(toolset, ctx, task_id, 'completed')
        seen['report'] = await toolset.check_task(ctx, task_id)
        seen['late'] = await toolset.send_message_to_task(ctx, task_id, 'too late')

    await _drive_test(capability, body)
    assert 'it will be applied before its next model request' in seen['sent']
    assert steered == ['Message from your parent agent: focus on caching']
    assert 'saw 1 steering message(s)' in seen['report']
    assert 'already completed' in seen['late']


async def test_send_message_to_a_task_that_has_not_started_retries():
    gate = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate))], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        # No await between spawn and send: the worker has not entered its run yet.
        with pytest.raises(ModelRetry, match='still starting up'):
            await toolset.send_message_to_task(ctx, task_id, 'too early')
        gate.set()
        await _wait_for_status(toolset, ctx, task_id, 'completed')

    await _drive_test(capability, body)


async def test_soft_cancel_stops_at_a_step_boundary():
    gate = asyncio.Event()
    started_event = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate, started=started_event))], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await started_event.wait()
        seen['cancel'] = await toolset.cancel_task(ctx, task_id)
        gate.set()  # the child's in-flight model call finishes, then the boundary check fires
        await _wait_for_status(toolset, ctx, task_id, 'cancelled')
        seen['report'] = await toolset.check_task(ctx, task_id)
        seen['again'] = await toolset.cancel_task(ctx, task_id)

    await _drive_test(capability, body)
    assert 'stops at its next step boundary' in seen['cancel']
    assert 'Status: cancelled' in seen['report']
    assert 'already cancelled' in seen['again']


async def test_hard_cancel_interrupts_immediately():
    gate = asyncio.Event()
    started_event = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate, started=started_event))], agent_folders=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await started_event.wait()
        seen['cancel'] = await toolset.cancel_task(ctx, task_id, force=True)
        await _wait_for_status(toolset, ctx, task_id, 'cancelled')

    result = await _drive_test(capability, body)
    assert seen['cancel'].endswith('cancelled.')
    assert any('cancelled' in text for text in _injected_texts(result))


async def test_soft_cancel_unblocks_a_task_waiting_for_an_answer():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_asking_child('worker'), can_ask_parent=True)], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        seen['wait'] = await toolset.wait_tasks(ctx, [task_id])  # returns early: waiting for answer
        await toolset.cancel_task(ctx, task_id)
        await _wait_for_status(toolset, ctx, task_id, 'cancelled')

    await _drive_test(capability, body)
    assert 'WAITING FOR YOUR ANSWER' in seen['wait']


async def test_double_answer_becomes_steering():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_asking_child('worker'), can_ask_parent=True)], agent_folders=None, notify=None
    )
    toolset = _toolset_of(capability)
    seen: dict[str, str] = {}

    async def body(ctx: RunContext[object]) -> None:
        started = await toolset.delegate_task_backgroundable(ctx, 'worker', 't', background=True)
        task_id = _TASK_ID_RE.search(started).group(1)  # type: ignore[union-attr]
        await toolset.wait_tasks(ctx, [task_id])
        seen['first'] = await toolset.send_message_to_task(ctx, task_id, 'postgres')
        # The child has not resumed yet, so the future is resolved but the task is
        # still waiting: a second message falls through to the steering path.
        seen['second'] = await toolset.send_message_to_task(ctx, task_id, 'also add an index')
        await _wait_for_status(toolset, ctx, task_id, 'completed')

    await _drive_test(capability, body)
    assert seen['first'].startswith('Answer delivered')
    assert 'it will be applied before its next model request' in seen['second']


# --- end-of-run boundary --------------------------------------------------------------


def _delegate_bg_then_finish(agent_name: str, *, finish: str = 'parent done') -> FunctionModel:
    """A parent model that starts one background delegation, then always finishes."""
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'delegate_task', {'agent_name': agent_name, 'task': 't', 'background': True}, tool_call_id='c1'
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(finish)])

    return FunctionModel(model_fn)


async def test_on_parent_end_wait_delivers_the_result_before_finishing():
    gate = asyncio.Event()
    started_event = asyncio.Event()
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate, started=started_event, reply='late result'))],
        agent_folders=None,
    )
    parent: Agent[object, str] = Agent(_delegate_bg_then_finish('worker'), capabilities=[capability])

    async def open_gate() -> None:
        await started_event.wait()
        gate.set()

    opener = asyncio.create_task(open_gate())
    result = await parent.run('go')
    await opener
    injected = _injected_texts(result)
    assert any('completed: late result' in text for text in injected)
    # The parent got a redirect turn after its would-be-final response.
    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert len(responses) == 3


async def test_on_parent_end_wait_grace_expiry_cancels_stragglers():
    gate = asyncio.Event()  # never opened
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate))], agent_folders=None, end_grace_seconds=0.05
    )
    parent: Agent[object, str] = Agent(_delegate_bg_then_finish('worker'), capabilities=[capability])
    result = await parent.run('go')
    injected = _injected_texts(result)
    assert any('cancelled' in text for text in injected)


async def test_on_parent_end_cancel_kills_leftover_tasks_silently():
    gate = asyncio.Event()  # never opened
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_gated_child('worker', gate))], agent_folders=None, on_parent_end='cancel'
    )
    parent: Agent[object, str] = Agent(_delegate_bg_then_finish('worker'), capabilities=[capability])
    result = await parent.run('go')
    assert _injected_texts(result) == []
    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert len(responses) == 2  # no redirect turn


async def test_on_parent_end_wait_reminds_about_a_waiting_question_then_releases_it():
    answers: list[str] = []

    def child_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if returns:
            answers.append(cast(str, returns[0].content))
            return ModelResponse(parts=[TextPart('wrapped up')])
        return ModelResponse(parts=[ToolCallPart('ask_parent', {'question': 'proceed?'}, tool_call_id='q1')])

    child: Agent[object, str] = Agent(FunctionModel(child_fn), name='worker', description='asks')
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(child, can_ask_parent=True)], agent_folders=None)

    calls = {'n': 0}

    def parent_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'delegate_task', {'agent_name': 'worker', 'task': 't', 'background': True}, tool_call_id='c1'
                    )
                ]
            )
        # Ignores both the question and the reminder, tries to end every turn.
        return ModelResponse(parts=[TextPart('parent done')])

    parent: Agent[object, str] = Agent(FunctionModel(parent_fn), capabilities=[capability])
    result = await parent.run('go')
    injected = _injected_texts(result)
    assert any('is still waiting for your answer' in text for text in injected)
    assert answers == ['Your parent agent is finishing without answering. Proceed with your best judgment.']
    assert any('completed: wrapped up' in text for text in injected)


async def test_notify_when_idle_defers_the_notification_to_the_end():
    child = _echo_child('worker', 'quick result')
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(child)], agent_folders=None, notify='when_idle')
    parent: Agent[object, str] = Agent(_delegate_bg_then_finish('worker'), capabilities=[capability])
    result = await parent.run('go')
    injected = _injected_texts(result)
    assert any('completed: quick result' in text for text in injected)


# --- configuration surface ------------------------------------------------------------


async def test_allow_background_off_keeps_the_sync_surface():
    capability: SubAgents[object] = SubAgents(
        agents=[SubAgent(_echo_child('worker'))], agent_folders=None, allow_background=False
    )
    toolset = _toolset_of(capability)
    tools = await toolset.get_tools(cast(RunContext[object], _stub_ctx()))
    assert set(tools) == {'delegate_task'}
    instructions = capability.get_instructions()
    assert isinstance(instructions, str)
    assert 'background' not in instructions

    parent: Agent[object, str] = Agent(FunctionModel(_delegate_once_sync('worker')), capabilities=[capability])
    result = await parent.run('go')
    assert 'child result' in str(result.all_messages())


async def test_background_instructions_and_tools_are_exposed():
    capability: SubAgents[object] = SubAgents(agents=[SubAgent(_echo_child('worker'))], agent_folders=None)
    toolset = _toolset_of(capability)
    tools = await toolset.get_tools(cast(RunContext[object], _stub_ctx()))
    assert set(tools) == {'delegate_task', 'check_task', 'wait_tasks', 'send_message_to_task', 'cancel_task'}
    instructions = capability.get_instructions()
    assert isinstance(instructions, str)
    assert 'background=true' in instructions
    assert 'cancel_task' in instructions


def _delegate_once_sync(agent_name: str) -> Any:
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': agent_name, 'task': 't'}, tool_call_id='c1')]
            )
        return ModelResponse(parts=[TextPart('parent done')])

    return model_fn


def _stub_ctx() -> Any:
    """A minimal context stand-in for `get_tools` (which only reads retry state)."""
    from pydantic_ai.models.test import TestModel as _TestModel
    from pydantic_ai.usage import RunUsage

    return RunContext(deps=None, model=_TestModel(), usage=RunUsage())
