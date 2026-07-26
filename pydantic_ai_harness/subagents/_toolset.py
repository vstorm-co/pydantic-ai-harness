"""Sub-agent toolset: a delegate tool that runs named child agents, sync or in the background."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, Literal

from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

# Private import: pydantic-ai has no public way to tell capability-contributed
# toolsets apart from the agent's own in `agent.toolsets`.
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.usage import UsageLimits
from pydantic_graph import End

from pydantic_ai_harness.subagents._tasks import RunTasks, TaskState

logger = logging.getLogger(__name__)

DepsFactory = Callable[[RunContext[AgentDepsT]], AgentDepsT]
"""Derives the deps for one delegation from the parent's run context (e.g. to hand
sub-agents an isolated copy of the parent's deps). When unset, the parent's `deps`
are forwarded as-is."""

_RESULT_PREVIEW_CHARS = 1000
"""How much of a result rides in notifications and `wait_tasks` summaries; the full
result is always available via `check_task`."""

# Signals that must always reach the parent run, even when a delegate has
# `contain_errors` on. Containing the first five would break the agent graph
# (deferred/approval/skip control-flow); a `UserError` is a setup bug that no
# retry can fix, so masking it into a retry only delays and obscures it.
# Cancellation (`asyncio.CancelledError`, a `BaseException`) is out of `except
# Exception`'s reach already, and a shared `UsageLimitExceeded` has its own clause.
_ALWAYS_PROPAGATE: tuple[type[Exception], ...] = (
    CallDeferred,
    ApprovalRequired,
    SkipModelRequest,
    SkipToolValidation,
    SkipToolExecution,
    UserError,
)

# Control-flow signals a background child cannot deliver: they need a caller to hand
# the deferred/approval state back to, and the delegate tool returned long ago.
_HUMAN_IN_THE_LOOP: tuple[type[Exception], ...] = (CallDeferred, ApprovalRequired)


@dataclass(frozen=True)
class SubAgent(Generic[AgentDepsT]):
    """One delegate: a child agent plus its per-delegate run controls.

    Pass a sequence of these as `SubAgents(agents=[...])`. The delegate's name --
    how the parent model refers to it, and how it is listed in the system prompt --
    is `name` when set, otherwise the agent's own `name`. An agent with neither is
    rejected by `SubAgents`.

    Every control below is optional; an unset field leaves the corresponding
    behaviour at the `SubAgents` default.
    """

    agent: AbstractAgent[AgentDepsT, Any]
    """The agent that runs when this delegate is invoked."""

    name: str | None = None
    """Name the parent model uses to delegate to this agent. Defaults to the
    agent's own `name` when unset."""

    description: str | None = None
    """Description for the system-prompt listing. Defaults to the agent's own
    `description` when unset; a delegate with neither is listed by name alone."""

    usage_limits: UsageLimits | None = None
    """Request/token budget for one delegation. When set, the child runs with
    its own usage accounting so the budget counts only the child's own requests
    and tokens (not the parent's or siblings'), even when `forward_usage=True`.
    The tradeoff: that child's tokens no longer aggregate into the parent's
    `usage`. Hitting this budget is a soft outcome (steering message), not a
    run-stopping `UsageLimitExceeded`."""

    timeout_seconds: float | None = None
    """Wall-clock budget for one delegation. When the child exceeds it, the run
    is cancelled and the parent gets a soft steering message (sync) or a
    cancellation notification (background) instead of hanging on the child."""

    max_calls: int | None = None
    """Maximum number of delegations to this sub-agent per parent run. Once
    reached, further delegations return a soft budget-exhausted message without
    running the child."""

    on_failure: str | None = None
    """Steering message returned to the parent for any soft degradation of this
    delegate (timeout, child failure, usage budget reached, call budget
    exhausted), in place of the built-in default. Setting it also makes child
    failures soft: a child error returns this message as a normal tool result
    instead of raising a parent `ModelRetry`. For background delegations it
    replaces the body of failure/cancellation notifications."""

    contain_errors: bool | None = None
    """Whether an unexpected sub-agent crash is contained instead of aborting the
    parent run. When `True`, an exception the child raises that is not an expected
    soft degradation (a provider `ModelAPIError`/`FallbackExceptionGroup`, a plain
    `ValueError` from a bad tool argument, etc.) is caught and returned to the parent
    as a bounded `ModelRetry`, so one delegate crash cannot kill the whole run. It
    stays loud: the exception rides the retry message and is logged, and
    `tool_retries` still bounds consecutive crashes into an abort. Cancellation, a
    shared usage-limit, pydantic-ai control-flow signals, and `UserError` always
    propagate regardless. Unset inherits `SubAgents.contain_errors` (default off).
    Orthogonal to `on_failure`, which only sets the message for expected soft
    degradations; a contained crash always raises the loud `ModelRetry`. Only
    meaningful for synchronous delegations: a background failure is always soft --
    it reaches the parent as a notification, never as an exception."""

    background: bool | None = None
    """Forces this delegate's execution mode: `True` always runs it in the
    background, `False` refuses background delegation (the model gets a retry
    telling it to call again without `background`). Unset (default) lets the
    model choose per call. Only takes effect when the `SubAgents` background
    surface is enabled (`allow_background=True`)."""

    can_ask_parent: bool = False
    """Whether background delegations of this agent get an `ask_parent` tool to
    ask the parent a question mid-run (the parent sees the question as an
    injected message and answers via `send_message_to_task`). Off by default so
    a delegate can't stall on a parent that never answers unless the app opted
    in. Background mode only."""

    @property
    def resolved_name(self) -> str | None:
        """The delegate's name: `name` if set, else the agent's own `name`."""
        return self.name or self.agent.name


def _is_capability_contributed(toolset: AbstractToolset[AgentDepsT]) -> bool:
    """Whether `toolset`'s tree contains a `CapabilityOwnedToolset`."""
    found = False

    def visit(node: AbstractToolset[AgentDepsT]) -> None:
        nonlocal found
        if isinstance(node, CapabilityOwnedToolset):
            found = True

    toolset.apply(visit)
    return found


def _preview(text: str, limit: int = _RESULT_PREVIEW_CHARS) -> str:
    """`text` clipped to `limit` characters, with a marker when clipped."""
    if len(text) <= limit:
        return text
    return text[:limit] + ' [...]'


class SubAgentToolset(FunctionToolset[AgentDepsT]):
    """Exposes a delegate tool that dispatches a task to a named sub-agent.

    Each delegation runs the child agent in a fresh run with its own message
    history, so the sub-agent never sees the parent conversation. The parent's
    `deps` are forwarded (or derived via `deps_factory`); its `usage` is shared
    when enabled; its tools are inherited when enabled; any `shared_capabilities`
    are applied to every sub-agent run; and sub-agent events are streamed to
    `event_stream_handler` when one is set (synchronous delegations only).
    Per-delegate run controls come from each `SubAgent`.

    When `allow_background` is on, the delegate tool grows a `background` flag and
    four management tools (`check_task`, `wait_tasks`, `send_message_to_task`,
    `cancel_task`); see `SubAgents` for the background execution model.
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, SubAgent[AgentDepsT]],
        forward_usage: bool,
        inherit_tools: bool,
        shared_capabilities: Sequence[AgentCapability[AgentDepsT]],
        event_stream_handler: EventStreamHandler[AgentDepsT] | None,
        tool_name: str,
        tool_retries: int | None,
        contain_errors: bool,
        call_counts: dict[str, dict[str, int]],
        allow_background: bool = False,
        deps_factory: DepsFactory[AgentDepsT] | None = None,
        notify: Literal['asap', 'when_idle'] | None = 'asap',
        max_concurrent_tasks: int | None = None,
        ask_timeout_seconds: float = 300.0,
        run_tasks: dict[str, RunTasks[AgentDepsT]] | None = None,
    ) -> None:
        super().__init__()
        self._agents: dict[str, SubAgent[AgentDepsT]] = dict(agents)
        self._forward_usage = forward_usage
        self._inherit_tools = inherit_tools
        self._shared_capabilities = list(shared_capabilities)
        self._event_stream_handler = event_stream_handler
        self._tool_name = tool_name
        self._contain_errors = contain_errors
        # Run-scoped delegation counts, keyed by run_id then sub-agent name.
        # Shared with the capability, which clears each run's entry in wrap_run.
        self._call_counts = call_counts
        self._deps_factory = deps_factory
        self._notify: Literal['asap', 'when_idle'] | None = notify
        self._max_concurrent_tasks = max_concurrent_tasks
        self._ask_timeout_seconds = ask_timeout_seconds
        # Run-scoped background task state, keyed by run_id. Shared with the
        # capability, which finalizes and clears each run's entry in wrap_run.
        self._run_tasks: dict[str, RunTasks[AgentDepsT]] = run_tasks if run_tasks is not None else {}
        if allow_background:
            self.add_function(self.delegate_task_backgroundable, name=tool_name, retries=tool_retries)
            self.add_function(self.check_task)
            self.add_function(self.wait_tasks)
            self.add_function(self.send_message_to_task)
            self.add_function(self.cancel_task)
            self._owned_tool_names = {tool_name, 'check_task', 'wait_tasks', 'send_message_to_task', 'cancel_task'}
        else:
            self.add_function(self.delegate_task, name=tool_name, retries=tool_retries)
            self._owned_tool_names = {tool_name}

    def _inherited_toolsets(self, ctx: RunContext[AgentDepsT]) -> list[AbstractToolset[AgentDepsT]] | None:
        """The parent agent's own toolsets, excluding capability-contributed ones.

        Capability toolsets are bound to capability instances registered in the
        parent run; carrying them into the sub-agent's run (where their owner is
        not registered) fails `CapabilityOwnedToolset`'s ownership resolution, and
        the tools would arrive without the hooks and instructions that make them
        work. Use `shared_capabilities` to share a capability with sub-agents.
        The delegate and task-management tools are also filtered out by name, so
        delegation cannot recurse. When this toolset was registered via the
        `SubAgents` capability the capability filter already drops it; the name
        filter covers direct registration in `Agent(toolsets=[...])`, where
        nothing wraps it in `CapabilityOwnedToolset`.
        """
        agent = ctx.agent
        if agent is None:  # pragma: no cover - the running agent is always set during a run
            return None
        # Capability toolsets surface as `CombinedToolset(CapabilityOwnedToolset(...))`
        # entries, so ownership is detected by walking each tree. Only core's capability
        # assembly constructs `CapabilityOwnedToolset`, so a tree containing one is
        # capability-contributed in its entirety.
        return [
            toolset.filtered(lambda _ctx, tool_def: tool_def.name not in self._owned_tool_names)
            for toolset in agent.toolsets
            if not _is_capability_contributed(toolset)
        ]

    def _budget_exhausted(self, ctx: RunContext[AgentDepsT], agent_name: str, max_calls: int) -> bool:
        """Increment this run's delegation count for `agent_name` and report whether it is over budget.

        Runs synchronously before any await, so concurrent delegations in one run
        count without a lock.
        """
        counts = self._call_counts.setdefault(ctx.run_id or '', {})
        counts[agent_name] = counts.get(agent_name, 0) + 1
        return counts[agent_name] > max_calls

    def _tasks_for(self, ctx: RunContext[AgentDepsT]) -> RunTasks[AgentDepsT]:
        """This run's background-task registry, created on first use."""
        return self._run_tasks.setdefault(ctx.run_id or '', RunTasks())

    def _lookup(self, agent_name: str) -> SubAgent[AgentDepsT]:
        """The named delegate, or a `ModelRetry` listing what exists."""
        sub_agent = self._agents.get(agent_name)
        if sub_agent is None:
            available = ', '.join(sorted(self._agents))
            raise ModelRetry(f'Unknown sub-agent {agent_name!r}. Available sub-agents: {available}.')
        return sub_agent

    def _get_task(self, ctx: RunContext[AgentDepsT], task_id: str) -> TaskState[AgentDepsT]:
        """The named background task, or a `ModelRetry` listing what exists."""
        state = self._tasks_for(ctx).tasks.get(task_id)
        if state is None:
            known = ', '.join(self._tasks_for(ctx).tasks) or 'none'
            raise ModelRetry(f'Unknown background task {task_id!r}. Tasks in this run: {known}.')
        return state

    async def delegate_task(self, ctx: RunContext[AgentDepsT], agent_name: str, task: str) -> str:
        """Delegate a self-contained task to a named sub-agent and return its result.

        The sub-agent runs in its own fresh context and does not see this
        conversation, so `task` must contain everything it needs.

        Args:
            ctx: The run context (provides the parent's deps, usage, and tools).
            agent_name: Name of the sub-agent to run. Must be one of the agents
                listed in the instructions.
            task: The complete, self-contained instruction for the sub-agent.
        """
        sub_agent = self._lookup(agent_name)
        return await self._delegate_sync(ctx, agent_name, sub_agent, task)

    async def delegate_task_backgroundable(
        self, ctx: RunContext[AgentDepsT], agent_name: str, task: str, background: bool = False
    ) -> str:
        """Delegate a self-contained task to a named sub-agent.

        The sub-agent runs in its own fresh context and does not see this
        conversation, so `task` must contain everything it needs. By default the
        call blocks and returns the sub-agent's result. With `background=True` it
        returns a task id immediately and the sub-agent runs in the background:
        you will get a `[background task ...]` message when it finishes, and you
        can manage it with `check_task`, `wait_tasks`, `send_message_to_task`,
        and `cancel_task` in the meantime.

        Args:
            ctx: The run context (provides the parent's deps, usage, and tools).
            agent_name: Name of the sub-agent to run. Must be one of the agents
                listed in the instructions.
            task: The complete, self-contained instruction for the sub-agent.
            background: Run the delegation in the background and return a task id
                immediately instead of blocking on the result.
        """
        sub_agent = self._lookup(agent_name)
        if sub_agent.background is False and background:
            raise ModelRetry(
                f'Sub-agent {agent_name!r} does not support background delegation. '
                f'Call `{self._tool_name}` again without `background`.'
            )
        if sub_agent.background is True:
            background = True
        if not background:
            return await self._delegate_sync(ctx, agent_name, sub_agent, task)
        return self._start_background(ctx, agent_name, sub_agent, task)

    async def _delegate_sync(
        self, ctx: RunContext[AgentDepsT], agent_name: str, sub_agent: SubAgent[AgentDepsT], task: str
    ) -> str:
        """Run one blocking delegation and return the sub-agent's result."""
        if sub_agent.max_calls is not None and self._budget_exhausted(ctx, agent_name, sub_agent.max_calls):
            return self._steer(
                sub_agent.on_failure,
                f'Delegate budget for {agent_name!r} is exhausted for this run '
                f'({sub_agent.max_calls} call(s)). Synthesize from existing evidence and '
                f'choose the next action; do not delegate to {agent_name!r} again.',
            )

        toolsets = self._inherited_toolsets(ctx) if self._inherit_tools else None
        capabilities = self._shared_capabilities or None
        usage_limits: UsageLimits | None
        if sub_agent.usage_limits is not None:
            # Isolated accounting so the per-child budget counts only this child.
            own_budget = True
            usage = None
            usage_limits = sub_agent.usage_limits
        else:
            own_budget = False
            usage = ctx.usage if self._forward_usage else None
            usage_limits = None

        # A sub-agent with no model of its own (e.g. one loaded from disk) inherits
        # the parent run's model; one that brought its own keeps it.
        model = None if sub_agent.agent.model is not None else ctx.model
        run = sub_agent.agent.run(
            task,
            deps=self._deps_factory(ctx) if self._deps_factory is not None else ctx.deps,
            model=model,
            usage=usage,
            usage_limits=usage_limits,
            toolsets=toolsets,
            capabilities=capabilities,
            event_stream_handler=self._event_stream_handler,
        )
        timeout = sub_agent.timeout_seconds
        try:
            result = await (asyncio.wait_for(run, timeout) if timeout is not None else run)
        except asyncio.TimeoutError:
            return self._steer(
                sub_agent.on_failure,
                f'Sub-agent {agent_name!r} exceeded its {timeout}s time budget. '
                f'Treat this as a recoverable observation and decide from existing evidence.',
            )
        except UsageLimitExceeded:
            if own_budget:
                return self._steer(
                    sub_agent.on_failure,
                    f'Sub-agent {agent_name!r} reached its usage budget. '
                    f'Treat this as a recoverable observation and decide from existing evidence.',
                )
            # A shared/parent usage limit means the whole tree is out of budget.
            raise
        except (ModelRetry, UnexpectedModelBehavior) as exc:
            if sub_agent.on_failure is not None:
                return sub_agent.on_failure
            # Soft sub-agent failures come back to the parent as a retry it can react to.
            raise ModelRetry(f'Sub-agent {agent_name!r} failed: {exc}') from exc
        except _ALWAYS_PROPAGATE:
            raise
        except Exception as exc:
            contain = sub_agent.contain_errors if sub_agent.contain_errors is not None else self._contain_errors
            if not contain:
                raise
            # Contain the crash so it cannot abort the parent, but keep it loud: the
            # exception rides the retry message and is logged, and `tool_retries`
            # bounds consecutive crashes into an abort.
            logger.warning('Contained crash from sub-agent %r', agent_name, exc_info=exc)
            raise ModelRetry(
                f'Sub-agent {agent_name!r} crashed: {type(exc).__name__}: {exc}. '
                f'Treat this as a recoverable failure and decide from existing evidence.'
            ) from exc
        return str(result.output)

    def _start_background(
        self, ctx: RunContext[AgentDepsT], agent_name: str, sub_agent: SubAgent[AgentDepsT], task: str
    ) -> str:
        """Spawn one background delegation and return its task id immediately."""
        if sub_agent.max_calls is not None and self._budget_exhausted(ctx, agent_name, sub_agent.max_calls):
            return self._steer(
                sub_agent.on_failure,
                f'Delegate budget for {agent_name!r} is exhausted for this run '
                f'({sub_agent.max_calls} call(s)). Synthesize from existing evidence and '
                f'choose the next action; do not delegate to {agent_name!r} again.',
            )
        run_tasks = self._tasks_for(ctx)
        if self._max_concurrent_tasks is not None and len(run_tasks.live()) >= self._max_concurrent_tasks:
            return self._steer(
                sub_agent.on_failure,
                f'The background task budget ({self._max_concurrent_tasks} concurrent) is exhausted. '
                f'Wait for or cancel a running task before starting another.',
            )
        task_id = uuid.uuid4().hex[:8]
        state = TaskState(
            task_id=task_id,
            agent_name=agent_name,
            task=task,
            parent_ctx=ctx,
            attention=run_tasks.attention,
            on_failure=sub_agent.on_failure,
        )
        run_tasks.tasks[task_id] = state
        state.asyncio_task = asyncio.create_task(self._worker(state, sub_agent), name=f'subagents-{task_id}')
        return (
            f'Started background task {task_id!r} ({agent_name}). You will be notified when it '
            f'finishes; manage it with check_task, wait_tasks, send_message_to_task, and cancel_task.'
        )

    async def _worker(self, state: TaskState[AgentDepsT], sub_agent: SubAgent[AgentDepsT]) -> None:
        """Run one background delegation to a terminal status, then notify the parent.

        Failures never propagate to the parent run -- the parent already got its tool
        result (the task id), so a background outcome is always delivered softly, as
        a status transition plus an enqueued notification.
        """
        agent_name = state.agent_name
        try:
            timeout = sub_agent.timeout_seconds
            drive = self._drive(state, sub_agent)
            await (asyncio.wait_for(drive, timeout) if timeout is not None else drive)
        except asyncio.TimeoutError:
            state.finish('cancelled', error=f'timed out after {sub_agent.timeout_seconds}s')
        except asyncio.CancelledError:
            state.finish('cancelled', error='cancelled')
            raise
        except UsageLimitExceeded as exc:
            if sub_agent.usage_limits is not None:
                state.finish('failed', error=f'reached its usage budget: {exc}')
            else:
                # The budget is shared with the parent: the whole agent tree is out.
                state.finish('failed', error=f'shared usage limit exceeded (the whole run is out of budget): {exc}')
        except _HUMAN_IN_THE_LOOP as exc:
            state.finish(
                'failed',
                error=f'{type(exc).__name__}: sub-agents that require approval or deferred tools '
                f'are not supported in background mode; delegate synchronously instead.',
            )
        except (ModelRetry, UnexpectedModelBehavior) as exc:
            state.finish('failed', error=str(exc))
        except Exception as exc:
            logger.warning('Background sub-agent %r crashed', agent_name, exc_info=exc)
            state.finish('failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            # Notify BEFORE waking waiters: a waiter that observes the terminal
            # status may let the parent run end, and the notification must already
            # be in the queue for the end-of-run redirect to deliver it.
            self._notify_parent(state)
            state.attention.set()

    async def _drive(self, state: TaskState[AgentDepsT], sub_agent: SubAgent[AgentDepsT]) -> None:
        """Drive one child run node-by-node -- core `run()`'s own loop, plus a soft-cancel check.

        Manual iteration (`run.next`) keeps every capability hook firing exactly as in
        `agent.run()` (the run lifecycle protocol lives inside `agent.iter()`), while
        letting us stop at a clean node boundary when `cancel_event` is set and hold
        the live `AgentRun` handle that `send_message_to_task` steers through.
        """
        ctx = state.parent_ctx
        toolsets = self._inherited_toolsets(ctx) if self._inherit_tools else None
        if sub_agent.can_ask_parent:
            toolsets = [*(toolsets or []), self._ask_parent_toolset(state)]
        capabilities = self._shared_capabilities or None
        usage_limits: UsageLimits | None
        if sub_agent.usage_limits is not None:
            usage = None
            usage_limits = sub_agent.usage_limits
        else:
            usage = ctx.usage if self._forward_usage else None
            usage_limits = None
        model = None if sub_agent.agent.model is not None else ctx.model
        async with sub_agent.agent.iter(
            state.task,
            deps=self._deps_factory(ctx) if self._deps_factory is not None else ctx.deps,
            model=model,
            usage=usage,
            usage_limits=usage_limits,
            toolsets=toolsets,
            capabilities=capabilities,
        ) as run:
            state.run_handle = run
            try:
                node = run.next_node
                while not isinstance(node, End):
                    if run.result is not None:  # a wrap_run capability short-circuited the run
                        break
                    if state.cancel_event.is_set():
                        state.finish('cancelled', error='cancelled')
                        return
                    node = await run.next(node)
                result = run.result
                assert result is not None, 'the sub-agent run did not produce a result'
                state.finish('completed', result=str(result.output))
            finally:
                state.usage = run.usage  # partial usage survives cancellation and timeouts
                state.run_handle = None  # no steering once the run is over

    def _notify_parent(self, state: TaskState[AgentDepsT]) -> None:
        """Enqueue the task's outcome into the parent run.

        Delivered before the parent's next model request, or -- when the parent was
        about to finish -- through the pending-message drain's end-of-run redirect,
        which gives the parent another turn to react. If the parent run has already
        ended this is a silent no-op; the capability's `wrap_run` finalizer guarantees
        no task outlives its parent run, so nothing is lost silently.
        """
        notify = self._notify
        if notify is None:
            return
        if state.status == 'completed':
            body = _preview(state.result or '')
        else:
            body = state.on_failure if state.on_failure is not None else _preview(state.error or '')
        state.parent_ctx.enqueue(
            f'[background task {state.task_id}] {state.agent_name} {state.status}: {body} '
            f"(full details via check_task('{state.task_id}'))",
            priority=notify,
        )

    def _ask_parent_toolset(self, state: TaskState[AgentDepsT]) -> FunctionToolset[AgentDepsT]:
        """A per-delegation toolset with one `ask_parent` tool, closed over this task's state."""
        toolset: FunctionToolset[AgentDepsT] = FunctionToolset(id='ask_parent')
        ask_timeout = self._ask_timeout_seconds

        async def ask_parent(ctx: RunContext[AgentDepsT], question: str) -> str:
            """Ask the parent agent a question and wait for the answer.

            Use this when you are blocked on information only the parent has. Keep
            questions specific and self-contained.

            Args:
                ctx: The run context.
                question: The question for the parent agent.
            """
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            state.pending_question = question
            state.answer_future = future
            # Enqueue before the status transition: the transition wakes the parent,
            # which may act on the question immediately -- it must already be queued.
            state.parent_ctx.enqueue(
                f'[background task {state.task_id}] {state.agent_name} asks: {question} '
                f"(answer with send_message_to_task('{state.task_id}', ...), or cancel_task to stop it)",
                priority='asap',
            )
            state.set_status('waiting_for_answer')
            try:
                return await asyncio.wait_for(future, timeout=ask_timeout)
            except asyncio.TimeoutError:
                return 'Your parent agent did not respond in time. Proceed with your best judgment.'
            finally:
                # Nothing else can transition the task while its run sits inside this
                # tool call (worker outcomes land only after the run unwinds), so the
                # status here is always `waiting_for_answer`.
                state.pending_question = None
                state.answer_future = None
                state.set_status('running')

        toolset.add_function(ask_parent)
        return toolset

    async def check_task(self, ctx: RunContext[AgentDepsT], task_id: str | None = None) -> str:
        """Check background tasks: one task's full status and result, or a one-line listing of all.

        Args:
            ctx: The run context.
            task_id: The task to inspect. Omit to list every background task of
                this run with one status line each.
        """
        run_tasks = self._tasks_for(ctx)
        if task_id is None:
            if not run_tasks.tasks:
                return 'No background tasks in this run.'
            lines = ['Background tasks:']
            for state in run_tasks.tasks.values():
                lines.append(f'- {state.task_id}: {state.agent_name} ({state.status}) - {_preview(state.task, 80)}')
            return '\n'.join(lines)
        state = self._get_task(ctx, task_id)
        lines = [
            f'Task: {state.task_id}',
            f'Sub-agent: {state.agent_name}',
            f'Status: {state.status}',
            f'Task description: {state.task}',
        ]
        if state.status == 'completed':
            lines.append(f'Result: {state.result}')
        elif state.status == 'waiting_for_answer':
            lines.append(f'Question: {state.pending_question}')
        elif state.done:
            lines.append(f'Error: {state.error}')
        else:
            elapsed = (datetime.now(timezone.utc) - state.started_at).total_seconds()
            lines.append(f'Running for: {elapsed:.1f}s')
        if state.usage is not None:
            usage = state.usage
            lines.append(f'Usage: {usage.input_tokens + usage.output_tokens} tokens ({usage.requests} request(s))')
        return '\n'.join(lines)

    async def wait_tasks(
        self,
        ctx: RunContext[AgentDepsT],
        task_ids: list[str],
        timeout: float = 300.0,
        mode: Literal['all', 'any'] = 'all',
    ) -> str:
        """Wait for background tasks to finish and return their outcomes.

        Returns early when a waited task starts waiting for an answer from you --
        answer it (`send_message_to_task`) or cancel it, then wait again.

        Args:
            ctx: The run context.
            task_ids: The tasks to wait for.
            timeout: Maximum seconds to wait before reporting whatever state the
                tasks are in.
            mode: `'all'` waits for every task to finish; `'any'` returns as soon
                as one reaches a terminal state, so you can react to the first
                finisher without stalling on the slowest.
        """
        run_tasks = self._tasks_for(ctx)
        states = [self._get_task(ctx, task_id) for task_id in task_ids]

        def _ready() -> bool:
            if any(state.status == 'waiting_for_answer' for state in states):
                return True
            finished = sum(1 for state in states if state.done)
            return finished == len(states) if mode == 'all' else finished > 0

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            # The event is only a wake-up signal; task state is the source of truth,
            # so clearing before checking cannot lose a transition (one that lands
            # after the check re-sets the event and the wait returns immediately).
            run_tasks.attention.clear()
            if _ready():
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(run_tasks.attention.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break

        finished = sum(1 for state in states if state.done)
        header_parts = [f'mode={mode}', f'{finished}/{len(states)} finished']
        still_running = len(states) - finished
        if still_running:
            header_parts.append(f'{still_running} still running')
        lines = [f'Task results ({", ".join(header_parts)}):']
        for state in states:
            if state.status == 'completed':
                lines.append(f'- {state.task_id} ({state.agent_name}): COMPLETED\n{_preview(state.result or "")}')
            elif state.status == 'waiting_for_answer':
                lines.append(
                    f'- {state.task_id} ({state.agent_name}): WAITING FOR YOUR ANSWER: {state.pending_question}'
                )
            elif state.done:
                lines.append(f'- {state.task_id} ({state.agent_name}): {state.status.upper()} - {state.error}')
            else:
                lines.append(f'- {state.task_id} ({state.agent_name}): {state.status}')
        return '\n'.join(lines)

    async def send_message_to_task(self, ctx: RunContext[AgentDepsT], task_id: str, message: str) -> str:
        """Send a message to a running background task: steer it, or answer its question.

        If the task asked you a question (status `waiting_for_answer`), the message
        is delivered as the answer. Otherwise it is injected into the task's next
        model request as guidance, without losing its progress.

        Args:
            ctx: The run context.
            task_id: The background task to message.
            message: The steering instruction or answer to deliver.
        """
        state = self._get_task(ctx, task_id)
        if state.resolve_answer(message):
            return f'Answer delivered to task {task_id!r}.'
        if state.done:
            return (
                f'Task {task_id!r} already {state.status}; nothing to deliver. '
                f"See check_task('{task_id}') for its outcome."
            )
        run_handle = state.run_handle
        if run_handle is None:
            raise ModelRetry(f'Task {task_id!r} is still starting up; try again.')
        run_handle.enqueue(f'Message from your parent agent: {message}', priority='asap')
        return f'Message delivered to task {task_id!r}; it will be applied before its next model request.'

    async def cancel_task(self, ctx: RunContext[AgentDepsT], task_id: str, force: bool = False) -> str:
        """Cancel a background task.

        Args:
            ctx: The run context.
            task_id: The background task to cancel.
            force: By default the task stops cooperatively at its next step
                boundary (an in-flight tool call is not interrupted). Set `True`
                to cancel it immediately.
        """
        state = self._get_task(ctx, task_id)
        if state.done:
            return f'Task {task_id!r} already {state.status}.'
        if force:
            if state.asyncio_task is not None:  # pragma: no branch - live tasks always have a worker
                state.asyncio_task.cancel()
            return f'Task {task_id!r} cancelled.'
        state.cancel_event.set()
        # A task blocked on ask_parent sits inside a tool call, not at a node
        # boundary -- unblock it so the cooperative cancel can take effect now
        # rather than after the ask timeout.
        state.resolve_answer('Your parent agent cancelled this task. Stop working and wrap up immediately.')
        return (
            f'Cancellation requested for task {task_id!r}; it stops at its next step boundary '
            f'(an in-flight tool call finishes first; use force=True to interrupt).'
        )

    @staticmethod
    def _steer(on_failure: str | None, default: str) -> str:
        """A soft steering message: the delegate's `on_failure` override, else `default`."""
        if on_failure is not None:
            return on_failure
        return default
