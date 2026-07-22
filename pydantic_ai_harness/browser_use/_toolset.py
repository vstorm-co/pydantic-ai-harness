"""The `browse_web` toolset and the factory contract for building browser-use agents."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.browser_use._model import resolve_chat_model
from pydantic_ai_harness.browser_use._settings import BrowserAgentSettings

try:
    from browser_use import Agent as _BrowserUseAgent
    from browser_use.browser import BrowserProfile, BrowserSession
    from browser_use.llm.base import BaseChatModel
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'browser-use is required for BrowserUse. Install it with: pip install "pydantic-ai-harness[browser-use]"'
    ) from _import_error

_TOOL_NAME = 'browse_web'


class BrowserAgentHistory(Protocol):
    """The subset of browser-use's `AgentHistoryList` that the `browse_web` tool reads.

    `final_result` is the text of the agent's final `done` action (or `None` when
    it never finished), `errors` collects per-step error messages, `is_successful`
    is the agent's own verdict on the finished task (`None` while not done), and
    `structured_output` is the final result parsed against the configured output
    schema (`None` when no schema was configured; raises a pydantic
    `ValidationError` when the result does not parse). A real `AgentHistoryList`
    satisfies this protocol as-is.
    """

    def final_result(self) -> None | str:
        """The text of the final result, or `None` when the agent never finished."""
        ...  # pragma: no cover

    def errors(self) -> list[str | None]:
        """One entry per step: the step's error message, or `None` for clean steps."""
        ...  # pragma: no cover

    def is_successful(self) -> bool | None:
        """The agent's own success verdict for a finished task; `None` while not done."""
        ...  # pragma: no cover

    @property
    def structured_output(self) -> BaseModel | None:
        """The final result parsed against the configured output schema, if any."""
        ...  # pragma: no cover


class BrowserAgent(Protocol):
    """A ready-to-run browser agent for one task, as built by a `BrowserAgentFactory`."""

    def run(self, max_steps: int = 500) -> Awaitable[BrowserAgentHistory]:
        """Run the agent's own loop until the task finishes or `max_steps` is reached.

        Declared as returning `Awaitable` (not `async def`) so that
        `browser_use.Agent.run`, whose tracing decorator types it as returning
        a plain `Coroutine`, satisfies the protocol; an `async def`
        implementation satisfies it too.
        """
        ...  # pragma: no cover


@dataclass
class BrowserTask:
    """Everything the `browse_web` tool passes to a `BrowserAgentFactory` for one call.

    A dataclass rather than keyword arguments so that new fields can be added
    without breaking existing factories: unpack what you forward, ignore the
    rest.
    """

    task: str
    """The natural-language goal for the browser agent."""

    llm: BaseChatModel | None
    """The resolved chat model; `None` means browser-use's own default."""

    browser_session: BrowserSession
    """The session to browse in. Owned by the tool: killed after the call in
    `'call'` scope, kept alive and reused in `'agent'` scope."""

    use_vision: bool | Literal['auto']
    """Whether to send page screenshots to the model (`'auto'` follows the model's capabilities)."""

    output_schema: type[BaseModel] | None
    """Schema the agent's final result must conform to, forwarded as browser-use's `output_model_schema`."""

    sensitive_data: dict[str, str | dict[str, str]] | None
    """Secret placeholders for browser-use to substitute without showing the values to the model."""

    extend_system_message: str | None
    """Extra instructions appended to the browser agent's own system prompt."""

    settings: BrowserAgentSettings
    """The remaining browser-use `Agent` options, always a concrete instance.

    Its `*_llm` fields arrive resolved to browser-use chat models, so factories
    can forward them verbatim.
    """


class BrowserAgentFactory(Protocol):
    """Builds the browser agent that `browse_web` runs for one task.

    The default factory constructs a real `browser_use.Agent` from the
    `BrowserTask`, forwarding `BrowserTask.settings` in full. Pass a custom one
    via `BrowserUse.browser_agent` to intercept construction, or to substitute
    a fake in tests. Two rules: the factory must not start or stop the session
    itself (`browse_web` owns the session lifecycle), and it should keep
    browser-use's signal handling off (`enable_signal_handler=False`) -- the
    sub-agent must not install its own SIGINT handling inside a host
    application.
    """

    def __call__(self, request: BrowserTask) -> BrowserAgent:
        """Build a runnable browser agent for one `browse_web` call."""
        ...  # pragma: no cover


def default_browser_agent(request: BrowserTask) -> BrowserAgent:
    """Build a real `browser_use.Agent` (the default `BrowserAgentFactory`).

    The `resolve_chat_model` calls on the settings' `*_llm` fields narrow their
    static type; the toolset already resolved the values, so at runtime they
    pass through unchanged.
    """
    settings = request.settings
    # Explicit type arguments: `Agent`'s context and structured-output type
    # variables are unconstrained by this call, and neither is used here.
    # Signal handling stays off: the sub-agent must not install its own SIGINT
    # pause/resume handling inside a host application.
    return _BrowserUseAgent[None, BaseModel](
        task=request.task,
        llm=request.llm,
        browser_session=request.browser_session,
        use_vision=request.use_vision,
        output_model_schema=request.output_schema,
        sensitive_data=request.sensitive_data,
        extend_system_message=request.extend_system_message,
        enable_signal_handler=False,
        tools=settings.tools,
        override_system_message=settings.override_system_message,
        max_failures=settings.max_failures,
        max_actions_per_step=settings.max_actions_per_step,
        use_thinking=settings.use_thinking,
        flash_mode=settings.flash_mode,
        max_history_items=settings.max_history_items,
        page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
        fallback_llm=resolve_chat_model(settings.fallback_llm),
        use_judge=settings.use_judge,
        judge_llm=resolve_chat_model(settings.judge_llm),
        ground_truth=settings.ground_truth,
        calculate_cost=settings.calculate_cost,
        vision_detail_level=settings.vision_detail_level,
        llm_screenshot_size=settings.llm_screenshot_size,
        llm_timeout=settings.llm_timeout,
        step_timeout=settings.step_timeout,
        directly_open_url=settings.directly_open_url,
        include_recent_events=settings.include_recent_events,
        final_response_after_failure=settings.final_response_after_failure,
        enable_planning=settings.enable_planning,
        planning_replan_on_stall=settings.planning_replan_on_stall,
        planning_exploration_limit=settings.planning_exploration_limit,
        loop_detection_enabled=settings.loop_detection_enabled,
        loop_detection_window=settings.loop_detection_window,
        message_compaction=settings.message_compaction,
        max_clickable_elements_length=settings.max_clickable_elements_length,
        include_tool_call_examples=settings.include_tool_call_examples,
        initial_actions=settings.initial_actions,
        available_file_paths=settings.available_file_paths,
        save_conversation_path=settings.save_conversation_path,
        generate_gif=settings.generate_gif,
        demo_mode=settings.demo_mode,
    )


class BrowserUseToolset(FunctionToolset[AgentDepsT]):
    """Provides the `browse_web` tool: run an autonomous browser-use agent per task."""

    def __init__(
        self,
        *,
        browser_agent: BrowserAgentFactory,
        llm: BaseChatModel | None,
        browser_profile: BrowserProfile | None,
        allowed_domains: list[str] | None,
        headless: bool | None,
        max_steps: int,
        use_vision: bool | Literal['auto'],
        output_schema: type[BaseModel] | None,
        sensitive_data: dict[str, str | dict[str, str]] | None,
        extend_system_message: str | None,
        settings: BrowserAgentSettings,
        session_scope: Literal['call', 'agent'],
        cdp_url: str | None,
    ) -> None:
        super().__init__()
        self._browser_agent = browser_agent
        self._llm = llm
        self._browser_profile = browser_profile
        self._allowed_domains = allowed_domains
        self._headless = headless
        self._max_steps = max_steps
        self._use_vision: bool | Literal['auto'] = use_vision
        self._output_schema = output_schema
        self._sensitive_data = sensitive_data
        self._extend_system_message = extend_system_message
        # Resolve the settings' chat models once, so every factory (custom
        # ones included) receives a `BrowserTask` with ready-to-use models.
        self._settings = replace(
            settings,
            page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
            fallback_llm=resolve_chat_model(settings.fallback_llm),
            judge_llm=resolve_chat_model(settings.judge_llm),
        )
        self._session_scope: Literal['call', 'agent'] = session_scope
        self._cdp_url = cdp_url
        self._shared_session: BrowserSession | None = None
        self._session_lock = asyncio.Lock()
        self.add_function(self.browse_web, name=_TOOL_NAME)

    def _build_session(self) -> BrowserSession:
        """A fresh session, merging the profile with the capability's overrides.

        `BrowserSession` itself merges a provided `browser_profile` with directly
        passed fields, letting the non-`None` direct fields win, so the
        capability's `headless`, `allowed_domains`, and `cdp_url` override the
        profile exactly like they would on a hand-built session. `headless`
        defaults to on only when no profile is given; a profile keeps its own
        setting.

        In `'agent'` scope the session is created with `keep_alive=True`:
        without it, `browser_use.Agent` kills the session at the end of each
        run, which would break reuse across calls. The toolset's own
        `kill()` (in `aclose` and on a failed run) is a force stop and closes
        the browser regardless.
        """
        headless = self._headless
        if headless is None and self._browser_profile is None:
            headless = True
        return BrowserSession(
            cdp_url=self._cdp_url,
            browser_profile=self._browser_profile,
            headless=headless,
            allowed_domains=self._allowed_domains,
            keep_alive=True if self._session_scope == 'agent' else None,
        )

    async def _run_agent(self, task: str, session: BrowserSession) -> BrowserAgentHistory:
        """Build the sub-agent for `task` against `session` and run its loop."""
        agent = self._browser_agent(
            BrowserTask(
                task=task,
                llm=self._llm,
                browser_session=session,
                use_vision=self._use_vision,
                output_schema=self._output_schema,
                sensitive_data=self._sensitive_data,
                extend_system_message=self._extend_system_message,
                settings=self._settings,
            )
        )
        return await agent.run(max_steps=self._max_steps)

    def _render_result(self, history: BrowserAgentHistory) -> str:
        """The tool result for a finished run: text, schema JSON, or a failure report."""
        result = history.final_result()
        if result is None:
            step_errors = [error for error in history.errors() if error]
            detail = '; '.join(step_errors) if step_errors else 'no further details'
            return f'The browser agent stopped without producing a result ({detail}).'
        if self._output_schema is not None:
            try:
                structured = history.structured_output
            except ValidationError as error:
                raise ModelRetry(
                    f'The browser agent finished, but its result did not match the configured output schema: {error}'
                ) from error
            if structured is not None:
                return structured.model_dump_json()
        if history.is_successful() is False:
            return f'The browser agent could not fully complete the task. Its final message: {result}'
        return result

    async def browse_web(self, task: str) -> str:
        """Have an autonomous browser agent carry out a web task and return its result.

        Args:
            task: One self-contained web goal in natural language, e.g.
                "find the price of the Pro plan on example.com and return it".

        Returns:
            The browser agent's final text result, or JSON conforming to the
            configured output schema when one is set.
        """
        if self._session_scope == 'call':
            history = await self._run_in_fresh_session(task)
        else:
            history = await self._run_in_shared_session(task)
        return self._render_result(history)

    async def _run_in_fresh_session(self, task: str) -> BrowserAgentHistory:
        """One disposable session for one call, killed when the call ends, on success or failure."""
        session = self._build_session()
        try:
            return await self._run_agent(task, session)
        finally:
            await session.kill()

    async def _run_in_shared_session(self, task: str) -> BrowserAgentHistory:
        """The `'agent'`-scoped shared session; the lock serializes calls -- one browser, one driver at a time."""
        async with self._session_lock:
            if self._shared_session is None:
                self._shared_session = self._build_session()
            try:
                return await self._run_agent(task, self._shared_session)
            except BaseException:
                # A failed or cancelled run can leave the shared browser in an
                # unknown state; kill it so the next call starts fresh.
                session, self._shared_session = self._shared_session, None
                await session.kill()
                raise

    async def aclose(self) -> None:
        """Kill the shared browser session, if one is alive.

        Only relevant in `'agent'` session scope; a no-op otherwise. Safe to
        call multiple times.
        """
        async with self._session_lock:
            if self._shared_session is not None:
                session, self._shared_session = self._shared_session, None
                await session.kill()
