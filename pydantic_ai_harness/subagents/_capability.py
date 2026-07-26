"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.agent import Agent, AgentRunResult, EventStreamHandler
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    AgentNode,
    NodeResult,
    WrapRunHandler,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_graph import End

from pydantic_ai_harness.subagents._disk import (
    AgentOverride,
    ParsedAgent,
    parse_agent_markdown,
    resolve_folders,
)
from pydantic_ai_harness.subagents._effort import clamp_effort
from pydantic_ai_harness.subagents._tasks import RunTasks
from pydantic_ai_harness.subagents._toolset import DepsFactory, SubAgent, SubAgentToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

ToolResolver = Callable[[str], 'Sequence[AgentToolset[object]] | None']
"""Maps one tool name from a disk definition's `tools` list to the toolsets that
provide it, or `None` when the name is unknown (the loader warns and skips it)."""


@dataclass
class SubAgents(AbstractCapability[AgentDepsT]):
    """Let an agent delegate self-contained tasks to named sub-agents.

    Exposes a single `delegate_task(agent_name, task)` tool. Each delegation
    runs the chosen sub-agent in a fresh, isolated run (it never sees the parent
    conversation), and the available sub-agents are listed in the system prompt
    as a static, cache-stable instruction.

    Sub-agents are passed as a sequence of `SubAgent` entries, each pairing an
    agent with its per-delegate run controls (a `usage_limits` budget, a
    wall-clock `timeout_seconds`, a per-run `max_calls` budget, an `on_failure`
    steering message, and optional `name`/`description` overrides). A delegate's
    name is its `SubAgent.name`, or the agent's own `name` when unset; two
    explicitly-passed delegates resolving to the same name is an error.

    Sub-agents are also loaded from disk by default: each markdown agent definition
    under `./.agents/agents/` and `~/.agents/agents/` (or the `.claude/` equivalent)
    becomes a delegate, built with the parent's model. Disk delegates get no tools
    by default (`inherit_tools` is `False`); set `inherit_tools=True` to expose the
    parent's tools, or pass a `tool_resolver` to map their frontmatter tool names.
    Disk delegates coexist with explicitly-passed ones; explicitly-passed agents take
    precedence, then the project folder, then the home folder. A disk delegate whose
    name is already taken is skipped with a warning. Configure or disable this with
    `agent_folders`; see also `agent_overrides` and `tool_resolver`.

    The parent's `deps` are forwarded to each sub-agent (sub-agents therefore
    share the parent's `AgentDepsT`; see `deps_factory` to derive per-delegation
    deps instead), and by default the parent's `usage` is shared so usage limits
    apply across the whole agent tree. Optionally, the parent's tools can be
    inherited (`inherit_tools`), extra capabilities can be applied to every
    sub-agent run (`shared_capabilities`), and sub-agent events can be streamed
    to a handler (`event_stream_handler`).

    Delegations can also run in the background (`allow_background`, on by
    default): the delegate tool grows a `background` flag, task-management tools
    are exposed (`check_task`, `wait_tasks`, `send_message_to_task`,
    `cancel_task`), completed tasks notify the parent by injecting a message
    into its conversation (`notify`), the parent can steer a running task and
    answer its `ask_parent` questions (`SubAgent.can_ask_parent`), and the
    end-of-run boundary waits for or cancels whatever is still running
    (`on_parent_end`, `end_grace_seconds`).

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.subagents import SubAgent, SubAgents

    researcher = Agent('anthropic:claude-sonnet-4-6', name='researcher', description='Researches topics')
    writer = Agent('anthropic:claude-sonnet-4-6', name='writer', description='Writes prose')

    orchestrator = Agent(
        'anthropic:claude-opus-4-7',
        capabilities=[SubAgents(agents=[SubAgent(researcher), SubAgent(writer)])],
    )
    ```
    """

    agents: Sequence[SubAgent[AgentDepsT]] = ()
    """The sub-agents to expose, each a `SubAgent` pairing an agent with its
    per-delegate run controls. See `SubAgent`. These take precedence over any
    disk-loaded agents of the same name."""

    agent_folders: str | Sequence[Path] | None = 'agents'
    """Where to load markdown agent definitions from, in addition to `agents`.
    Defaults to the conventional layout, so constructing the capability auto-loads
    a repo's agent files with no extra configuration.

    - a folder-name `str` (the default `'agents'` is the conventional layout): for
      the project root (cwd) then the home root, load from `<root>/.agents/<name>/`,
      falling back to `<root>/.claude/<name>/` when `<root>/.agents/` is absent.
    - a sequence of paths: load from exactly those folders, in order.
    - `None`: disable disk loading entirely (only `agents` are exposed).

    Missing folders are skipped. Within a folder every `*.md` file is a candidate."""

    agent_overrides: Mapping[str, AgentOverride] = field(default_factory=dict[str, AgentOverride])
    """Per-disk-agent overrides keyed by the agent's name. An entry can set the
    agent's `model` (otherwise the parent's model is inherited) and its `effort`
    (otherwise the minimum floor). Has no effect on explicitly-passed `agents`."""

    tool_resolver: ToolResolver | None = None
    """Optional override for how a disk agent gets its tools. When set, each tool
    name in a definition's `tools`/`allowed-tools` frontmatter is passed to this
    resolver and the returned toolsets are attached to that agent; an unknown name
    (resolver returns `None`) is skipped with a warning. When unset, the
    frontmatter tool list is ignored and disk agents inherit the parent's tools
    via `inherit_tools` (set `inherit_tools=True` to expose them)."""

    forward_usage: bool = True
    """If `True`, the parent run's `usage` is shared with each sub-agent run, so
    token usage aggregates and usage limits apply across the whole agent tree."""

    inherit_tools: bool = False
    """If `True`, the parent agent's tools are exposed to each sub-agent run (the
    delegate tool itself is filtered out, so sub-agents can't recurse into
    further delegation). Off by default to avoid silently widening sub-agent access."""

    shared_capabilities: Sequence[AgentCapability[AgentDepsT]] = ()
    """Capabilities applied to every sub-agent run, in addition to whatever each
    sub-agent already has."""

    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None
    """If set, this handler is passed to each sub-agent run, so the sub-agent's
    model-streaming and tool events surface to the caller. The handler receives
    the sub-agent's own `RunContext` and event stream."""

    tool_name: str = 'delegate_task'
    """Name of the delegate tool exposed to the model."""

    tool_retries: int | None = 2
    """Retries for the delegate tool -- how many extra attempts it gets after a
    sub-agent error before the parent run aborts. A sub-agent failure (e.g. it
    exhausts its own output retries) surfaces to the parent as a tool retry it
    can react to by re-delegating with a corrected task. The retry counter
    resets after any successful delegation, so this bounds consecutive failures,
    not total ones. Defaults to `2` (pydantic-ai's per-tool default is `1`) so a
    repeated flaky sub-agent does not abort the parent run on its first repeat;
    set `None` to inherit the parent agent's default tool retries instead."""

    contain_errors: bool = False
    """Default for `SubAgent.contain_errors`: whether an unexpected sub-agent crash
    is caught and returned to the parent as a bounded `ModelRetry` instead of
    aborting the parent run. Off by default, so a crash propagates. Any `SubAgent`
    can override this per delegate. See `SubAgent.contain_errors` for the
    containment contract and what always propagates regardless."""

    allow_background: bool = True
    """Whether the model may run delegations in the background. When on, the
    delegate tool grows a `background` flag and the task-management tools
    (`check_task`, `wait_tasks`, `send_message_to_task`, `cancel_task`) are
    exposed. A background delegation returns a task id immediately; the child
    runs concurrently with the parent, its completion (or failure) is injected
    into the parent's conversation as a `[background task ...]` message -- if
    the parent was about to finish, it gets another turn to react -- and the
    parent can steer it, answer its questions, and cancel it mid-run. Note:
    `event_stream_handler` does not apply to background delegations (driving a
    run and streaming it needs core API that `Agent.iter()` does not expose)."""

    deps_factory: DepsFactory[AgentDepsT] | None = None
    """Derives the deps for each delegation (sync and background) from the
    parent's run context, e.g. to hand every sub-agent an isolated copy of the
    parent's deps. When unset, the parent's `deps` are forwarded as-is. A factory
    returning the parent's deps unchanged means sub-agents share mutable state
    with the parent and with each other -- fine when intended, but background
    delegations then mutate it concurrently."""

    on_parent_end: Literal['wait', 'cancel'] = 'wait'
    """What happens to still-running background tasks when the parent run would
    end. `'wait'` (default): the run pauses at the boundary until the first task
    finishes (bounded by `end_grace_seconds`); its completion message then gives
    the model another turn to read the result, keep waiting, or cancel the rest --
    the model, not the harness, decides to abandon work, but it decides informed.
    A task waiting for an answer gets a reminder turn instead of a blind wait,
    and is released with a "proceed with your best judgment" answer if the model
    ignores the reminder. `'cancel'`: don't pause; whatever is still running is
    cancelled when the run ends. On every exit path (including errors and usage
    limits) any surviving tasks are cancelled before the run returns, so no
    background work outlives its parent run."""

    end_grace_seconds: float | None = 300.0
    """Bound on each `'wait'` pause at the end-of-run boundary. When it expires,
    the still-running tasks are cancelled and their cancellation notices give the
    model one final turn. `None` waits without bound (rely on each delegate's
    `timeout_seconds`)."""

    notify: Literal['asap', 'when_idle'] | None = 'asap'
    """When a background task's outcome is injected into the parent conversation:
    `'asap'` before the parent's next model request, `'when_idle'` only once the
    parent would otherwise finish its turn, `None` never (the model must poll
    with `check_task`/`wait_tasks`)."""

    max_concurrent_tasks: int | None = None
    """Cap on simultaneously live background tasks per parent run. Over the cap,
    a background delegation returns a soft budget message without running the
    child. `None` (default) means no cap."""

    ask_timeout_seconds: float = 300.0
    """How long a background child's `ask_parent` waits for an answer before the
    child is told to proceed with its best judgment."""

    _by_name: dict[str, SubAgent[AgentDepsT]] = field(
        default_factory=dict[str, 'SubAgent[AgentDepsT]'], init=False, repr=False, compare=False
    )
    """Sub-agents keyed by resolved name, built in `__post_init__` and passed to
    the toolset. Insertion order matches `agents` for a stable prompt listing."""

    _call_counts: dict[str, dict[str, int]] = field(
        default_factory=dict[str, 'dict[str, int]'], init=False, repr=False, compare=False
    )
    """Run-scoped delegation counts (run_id -> name -> count), shared with the
    toolset and cleared per run in `wrap_run`. Backs `SubAgent.max_calls`."""

    _tasks: dict[str, RunTasks[AgentDepsT]] = field(
        default_factory=dict[str, 'RunTasks[AgentDepsT]'], init=False, repr=False, compare=False
    )
    """Run-scoped background task registries (run_id -> RunTasks), shared with the
    toolset and finalized per run in `wrap_run`."""

    def __post_init__(self) -> None:
        by_name: dict[str, SubAgent[AgentDepsT]] = {}
        for sub_agent in self.agents:
            name = sub_agent.resolved_name
            if name is None:
                raise ValueError('Sub-agent has no name: give its `Agent` a `name`, or set `SubAgent(name=...)`.')
            if name in by_name:
                raise ValueError(
                    f'Duplicate sub-agent name {name!r}. Each sub-agent needs a distinct name; '
                    f'set `SubAgent(name=...)` to disambiguate.'
                )
            by_name[name] = sub_agent
        # Disk agents are lower precedence than explicit ones and than earlier
        # folders, so a name already taken is shadowed (a warning, not an error --
        # overriding a home agent from the project, or a disk agent from code, is
        # the intended path).
        for sub_agent in self._load_disk_agents():
            name = sub_agent.resolved_name
            if name is None:  # pragma: no cover - disk agents always get a name (frontmatter or stem)
                continue
            if name in by_name:
                warnings.warn(
                    f'Disk sub-agent {name!r} is shadowed by a higher-precedence definition; skipping it.',
                    stacklevel=2,
                )
                continue
            by_name[name] = sub_agent
        self._by_name = by_name

    def _load_disk_agents(self) -> list[SubAgent[AgentDepsT]]:
        """Build a `SubAgent` for every markdown definition in `agent_folders`.

        Folders are returned in precedence order (project before home); within a
        folder, files are loaded in sorted name order for a stable listing.
        """
        if self.agent_folders is None:
            return []
        result: list[SubAgent[AgentDepsT]] = []
        for folder in resolve_folders(self.agent_folders, Path.cwd(), Path.home()):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob('*.md')):
                try:
                    text = path.read_text(encoding='utf-8')
                except (OSError, UnicodeDecodeError) as exc:
                    warnings.warn(f'Skipping unreadable disk sub-agent file {str(path)!r}: {exc}', stacklevel=2)
                    continue
                parsed = parse_agent_markdown(text)
                result.append(self._build_disk_agent(parsed.name or path.stem, parsed))
        return result

    def _build_disk_agent(self, name: str, parsed: ParsedAgent) -> SubAgent[AgentDepsT]:
        """Build one disk-defined sub-agent: parent model + floored effort, tools resolved or inherited.

        The agent is constructed with `deps_type=object` so the parent's deps (of
        any type) flow through unused at delegation; this also lets a disk
        `SubAgent[object]` sit in the parent's `SubAgent[AgentDepsT]` roster.
        """
        override = self.agent_overrides.get(name)
        model = override.model if override is not None else None
        effort = override.effort if override is not None else None
        toolsets = self._resolve_disk_tools(parsed.tools) if self.tool_resolver is not None else None
        agent = Agent(
            model,
            deps_type=object,
            name=name,
            description=parsed.description,
            instructions=parsed.body or None,
            model_settings=ModelSettings(thinking=clamp_effort(effort)),
            toolsets=toolsets,
        )
        return SubAgent(agent)

    def _resolve_disk_tools(self, tool_names: Sequence[str]) -> list[AgentToolset[object]]:
        """Map a definition's tool names to toolsets via `tool_resolver`, warning on unknown names."""
        resolver = self.tool_resolver
        if resolver is None:  # pragma: no cover - only called when tool_resolver is set
            return []
        toolsets: list[AgentToolset[object]] = []
        for tool_name in tool_names:
            resolved = resolver(tool_name)
            if resolved is None:
                warnings.warn(f'Unknown tool {tool_name!r} in disk sub-agent definition; skipping it.', stacklevel=2)
                continue
            toolsets.extend(resolved)
        return toolsets

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run the parent agent, then finalize this run's background tasks and delegation counts.

        The finalizer runs on every exit path (normal end, error, usage limit,
        cancellation), so no background task ever outlives its parent run.
        """
        try:
            return await handler()
        finally:
            self._call_counts.pop(ctx.run_id or '', None)
            run_tasks = self._tasks.pop(ctx.run_id or '', None)
            if run_tasks is not None:
                await run_tasks.cancel_all()

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Hold the end-of-run boundary while background tasks are still live (`on_parent_end='wait'`).

        Runs before the pending-message drain's own `after_node_run` (the drain is
        ordered outermost, and after-hooks run in reverse), so anything a finishing
        task enqueues here is guaranteed to be picked up by the drain's end-of-run
        redirect in the same pass -- the model gets another turn with the outcome
        in front of it instead of the run silently ending or silently killing work.
        """
        if not isinstance(result, End) or self.on_parent_end != 'wait':
            return result
        run_tasks = self._tasks.get(ctx.run_id or '')
        if run_tasks is None:
            return result

        while True:
            # The attention event is only a wake-up signal; task state is the source
            # of truth. Clearing before reading the states cannot lose a transition:
            # one that lands after the clear re-sets the event, and the wait below
            # then returns immediately for another pass.
            run_tasks.attention.clear()
            live = run_tasks.live()
            if not live:
                return result

            # A task waiting for the parent's answer must not be blind-waited on:
            # only the parent can unblock it. Remind once (the redirect gives the
            # model a turn to answer or cancel); if the model ignored the reminder
            # and is ending again, release the child with a best-judgment answer
            # and go back to waiting for it like any other running task.
            fresh_waiting = [state for state in live if state.status == 'waiting_for_answer' and not state.reminded]
            if fresh_waiting:
                for state in fresh_waiting:
                    state.reminded = True
                    ctx.enqueue(
                        f'Background task {state.task_id!r} ({state.agent_name}) is still waiting for your answer: '
                        f"{state.pending_question} -- answer with send_message_to_task('{state.task_id}', ...) "
                        f"or cancel_task('{state.task_id}').",
                        priority='asap',
                    )
                return result
            for state in live:
                if state.status == 'waiting_for_answer':
                    state.resolve_answer(
                        'Your parent agent is finishing without answering. Proceed with your best judgment.'
                    )

            workers = [
                state.asyncio_task for state in live if state.asyncio_task is not None and not state.asyncio_task.done()
            ]
            if not workers:  # pragma: no cover - live tasks always have a live worker
                return result
            waker = asyncio.ensure_future(run_tasks.attention.wait())
            done, _pending = await asyncio.wait(
                [*workers, waker], timeout=self.end_grace_seconds, return_when=asyncio.FIRST_COMPLETED
            )
            waker.cancel()
            if done:
                # A worker finished, or a task changed status mid-pause (e.g. it
                # started waiting for an answer) -- re-evaluate everything.
                continue
            # Grace expired with nothing happening: cancel the stragglers and let
            # their cancellation notices give the model one final, informed turn.
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            return result

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable listing of the available sub-agents."""
        if not self._by_name:
            return None
        lines: list[str] = []
        for name, sub_agent in self._by_name.items():
            description = sub_agent.description or sub_agent.agent.description
            lines.append(f'- {name}: {description}' if description else f'- {name}')
        listing = '\n'.join(lines)
        instructions = (
            f'You can delegate self-contained tasks to these sub-agents using the `{self.tool_name}` '
            f'tool. Each runs in its own fresh context and does not see this conversation, so pass '
            f'everything it needs.\n\nAvailable sub-agents:\n{listing}'
        )
        if self.allow_background:
            instructions += (
                f'\n\nDelegations can also run in the background: call `{self.tool_name}` with '
                f'`background=true` to start one and keep working; a `[background task ...]` message '
                f'arrives when it finishes. Manage running tasks with `check_task`, `wait_tasks`, '
                f'`send_message_to_task` (steer a task, or answer a question it asked you), and '
                f'`cancel_task`. Cancel tasks you no longer need before finishing -- ending your reply '
                f'waits for (or cancels) whatever is still running.'
            )
        return instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the delegate tool, or `None` when no sub-agents are configured."""
        if not self._by_name:
            return None
        return SubAgentToolset(
            agents=self._by_name,
            forward_usage=self.forward_usage,
            inherit_tools=self.inherit_tools,
            shared_capabilities=self.shared_capabilities,
            event_stream_handler=self.event_stream_handler,
            tool_name=self.tool_name,
            tool_retries=self.tool_retries,
            contain_errors=self.contain_errors,
            call_counts=self._call_counts,
            allow_background=self.allow_background,
            deps_factory=self.deps_factory,
            notify=self.notify,
            max_concurrent_tasks=self.max_concurrent_tasks,
            ask_timeout_seconds=self.ask_timeout_seconds,
            run_tasks=self._tasks,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable -- the capability holds live `Agent` instances."""
        return None
