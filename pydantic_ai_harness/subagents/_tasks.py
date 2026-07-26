"""Run-scoped state for background sub-agent delegations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Generic, Literal

from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRun
    from pydantic_ai.usage import RunUsage

TaskStatus = Literal['running', 'waiting_for_answer', 'completed', 'failed', 'cancelled']
"""Lifecycle of one background delegation.

`running` -> (`waiting_for_answer` -> `running`)* -> `completed` | `failed` | `cancelled`.
"""

TERMINAL_STATUSES: frozenset[str] = frozenset({'completed', 'failed', 'cancelled'})
"""Statuses a task never leaves."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TaskState(Generic[AgentDepsT]):
    """Mutable state of one background delegation. Internal -- never handed to the model or the app.

    Created by the delegate tool when a background delegation starts, mutated by the
    worker and the management tools, and dropped when the parent run's `wrap_run`
    finalizer cleans up. `attention` is the owning run's wake-up event (shared by all
    of the run's tasks), set on every status transition so `wait_tasks` can sleep on
    it instead of polling.
    """

    task_id: str
    agent_name: str
    task: str
    parent_ctx: RunContext[AgentDepsT]
    """The parent run's context, captured at delegation time. Its `enqueue` stays a
    live channel for the whole parent run: every rebuilt `RunContext` points at the
    same pending-message queue held on the graph state."""
    attention: asyncio.Event
    on_failure: str | None = None
    status: TaskStatus = 'running'
    result: str | None = None
    error: str | None = None
    usage: RunUsage | None = None
    started_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None
    pending_question: str | None = None
    answer_future: asyncio.Future[str] | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    reminded: bool = False
    run_handle: AgentRun[AgentDepsT, Any] | None = None
    """The child's live `AgentRun` while it is executing -- the steering hook
    (`run_handle.enqueue`). `None` before the run starts and after it ends."""
    asyncio_task: asyncio.Task[None] | None = None

    @property
    def done(self) -> bool:
        """Whether the task reached a terminal status."""
        return self.status in TERMINAL_STATUSES

    def set_status(self, status: TaskStatus) -> None:
        """Transition between live statuses and wake anything sleeping on the run's attention event.

        Terminal transitions go through `finish` instead, which does not wake waiters
        (see there for why); the worker wakes them after the parent notification.
        """
        self.status = status
        self.attention.set()

    def finish(self, status: TaskStatus, *, result: str | None = None, error: str | None = None) -> None:
        """Record a terminal outcome (idempotent -- the first terminal transition wins).

        Idempotence matters on the hard-cancel path: `cancel_task(force=True)` makes the
        worker record `cancelled`, and a slower soft-cancel or timeout path must not
        overwrite the outcome afterwards.

        Deliberately does NOT set the attention event: the worker wakes waiters in its
        `finally`, after the parent notification is enqueued. Waking here would let the
        parent observe the terminal status, decide the run is over, and end before the
        notification exists -- stranding it undelivered.
        """
        if self.done:  # pragma: no cover - defends the force-cancel-vs-completion race
            return
        self.result = result
        self.error = error
        self.status = status
        self.completed_at = _utcnow()

    def resolve_answer(self, answer: str) -> bool:
        """Resolve a pending `ask_parent` future, if any. Returns whether one was resolved."""
        future = self.answer_future
        if future is not None and not future.done():
            future.set_result(answer)
            return True
        return False


@dataclass
class RunTasks(Generic[AgentDepsT]):
    """All background tasks of one parent run, plus the run's shared wake-up event."""

    tasks: dict[str, TaskState[AgentDepsT]] = field(default_factory=dict[str, 'TaskState[AgentDepsT]'])
    attention: asyncio.Event = field(default_factory=asyncio.Event)

    def live(self) -> list[TaskState[AgentDepsT]]:
        """Tasks not yet in a terminal status, in creation order."""
        return [state for state in self.tasks.values() if not state.done]

    async def cancel_all(self) -> None:
        """Hard-cancel every live worker and wait for them to finish their cleanup."""
        live_tasks = [
            state.asyncio_task
            for state in self.live()
            if state.asyncio_task is not None and not state.asyncio_task.done()
        ]
        for task in live_tasks:
            task.cancel()
        if live_tasks:
            await asyncio.gather(*live_tasks, return_exceptions=True)
