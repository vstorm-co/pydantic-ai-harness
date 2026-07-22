"""Tests for the BrowserUse capability and BrowserUseToolset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, overload

import pytest

# `browser_use.Agent` is imported from its defining module: the test package
# `tests/browser_use` shadows the top-level `browser_use` name in pyright's
# tests execution environment, while submodule imports resolve correctly.
from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.browser_use import (
    BrowserAgentSettings,
    BrowserTask,
    BrowserUse,
    BrowserUseToolset,
    PydanticAIChatModel,
)

T = TypeVar('T', bound=BaseModel)


@pytest.fixture
def kill_calls(monkeypatch: pytest.MonkeyPatch) -> list[BrowserSession]:
    """Record `BrowserSession.kill` calls instead of running the real teardown."""
    calls: list[BrowserSession] = []

    async def record_kill(self: BrowserSession) -> None:
        calls.append(self)

    monkeypatch.setattr(BrowserSession, 'kill', record_kill)
    return calls


class _FakeChatModel:
    """A `BaseChatModel` double, passed through opaquely and never invoked in these tests.

    Structural conformance rather than inheritance: the protocol declares
    `provider`/`name` as properties and `ainvoke` with overloads, so a subclass
    would have to restate all three to satisfy the override checks.
    """

    _verified_api_keys: bool = True
    model: str = 'fake-model'
    provider: str = 'fake'
    name: str = 'fake-model'
    model_name: str = 'fake-model'

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: object) -> object:
        raise NotImplementedError('the fake chat model is never validated')  # pragma: no cover

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: None = None, **kwargs: object
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T], **kwargs: object
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: object
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        raise NotImplementedError('the fake chat model is never invoked')  # pragma: no cover


class _Facts(BaseModel):
    name: str
    price_usd: int


def _validation_error() -> ValidationError:
    """A real pydantic `ValidationError` for `_Facts`."""
    try:
        _Facts.model_validate({})
    except ValidationError as error:
        return error
    raise AssertionError('unreachable')  # pragma: no cover


@dataclass
class _FakeHistory:
    """A `BrowserAgentHistory` double with canned outcomes."""

    result: str | None = None
    step_errors: list[str | None] = field(default_factory=list[str | None])
    success: bool | None = None
    structured: BaseModel | None = None
    structured_error: ValidationError | None = None

    def final_result(self) -> None | str:
        return self.result

    def errors(self) -> list[str | None]:
        return self.step_errors

    def is_successful(self) -> bool | None:
        return self.success

    @property
    def structured_output(self) -> BaseModel | None:
        if self.structured_error is not None:
            raise self.structured_error
        return self.structured


class _FakeBrowserAgent:
    """A `BrowserAgent` double: records `run` calls, returns a canned history."""

    def __init__(self, history: _FakeHistory, error: Exception | None = None) -> None:
        self.history = history
        self.error = error
        self.run_calls: list[int] = []

    async def run(self, max_steps: int = 500) -> _FakeHistory:
        self.run_calls.append(max_steps)
        if self.error is not None:
            raise self.error
        return self.history


class _FakeFactory:
    """A `BrowserAgentFactory` double: records the `BrowserTask` requests."""

    def __init__(self, agent: _FakeBrowserAgent) -> None:
        self.agent = agent
        self.requests: list[BrowserTask] = []

    def __call__(self, request: BrowserTask) -> _FakeBrowserAgent:
        self.requests.append(request)
        return self.agent


def _success_factory(result: str = 'done') -> _FakeFactory:
    """A factory whose agent finishes successfully with `result`."""
    return _FakeFactory(_FakeBrowserAgent(_FakeHistory(result=result, success=True)))


class TestBrowserUseToolset:
    async def test_returns_final_result(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory('The Pro plan costs $20.')
        toolset = BrowserUse[None](browser_agent=factory).get_toolset()
        assert isinstance(toolset, BrowserUseToolset)

        result = await toolset.browse_web('find the price of the Pro plan')

        assert result == 'The Pro plan costs $20.'
        [request] = factory.requests
        assert request.task == 'find the price of the Pro plan'
        assert request.llm is None
        assert request.use_vision is True
        assert request.output_schema is None
        assert request.sensitive_data is None
        assert request.extend_system_message is None
        assert request.settings == BrowserAgentSettings()
        assert factory.agent.run_calls == [50]

    async def test_configuration_forwarded_to_session_and_agent(self, kill_calls: list[BrowserSession]) -> None:
        llm = _FakeChatModel()
        secrets: dict[str, str | dict[str, str]] = {'x_password': 'hunter2'}
        factory = _success_factory()
        capability = BrowserUse[None](
            llm=llm,
            allowed_domains=['example.com', '*.example.org'],
            headless=False,
            max_steps=7,
            use_vision='auto',
            sensitive_data=secrets,
            extend_system_message='Never submit forms.',
            cdp_url='http://localhost:9222',
            browser_agent=factory,
        )

        await capability.get_toolset().browse_web('task')

        [request] = factory.requests
        assert request.llm is llm
        assert request.use_vision == 'auto'
        assert request.sensitive_data is secrets
        assert request.extend_system_message == 'Never submit forms.'
        session = request.browser_session
        assert session.browser_profile.headless is False
        assert session.browser_profile.allowed_domains == ['example.com', '*.example.org']
        assert session.cdp_url == 'http://localhost:9222'
        assert factory.agent.run_calls == [7]

    async def test_defaults_to_headless_without_profile(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.headless is True
        # In 'call' scope the sub-agent may tear the session down itself; the
        # tool kills it in `finally` regardless.
        assert not session_profile.keep_alive

    async def test_browser_profile_forwarded_and_kept(self, kill_calls: list[BrowserSession]) -> None:
        profile = BrowserProfile(headless=False, allowed_domains=['docs.example.com'], user_agent='harness-test')
        factory = _success_factory()

        await BrowserUse[None](browser_profile=profile, browser_agent=factory).get_toolset().browse_web('task')

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.headless is False
        assert session_profile.allowed_domains == ['docs.example.com']
        assert session_profile.user_agent == 'harness-test'

    async def test_capability_fields_override_browser_profile(self, kill_calls: list[BrowserSession]) -> None:
        profile = BrowserProfile(headless=False, allowed_domains=['docs.example.com'])
        factory = _success_factory()
        capability = BrowserUse[None](
            browser_profile=profile,
            headless=True,
            allowed_domains=['example.com'],
            browser_agent=factory,
        )

        await capability.get_toolset().browse_web('task')

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.headless is True
        assert session_profile.allowed_domains == ['example.com']

    async def test_session_killed_after_success(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        [killed] = kill_calls
        assert killed is factory.requests[0].browser_session

    async def test_session_killed_when_run_raises(self, kill_calls: list[BrowserSession]) -> None:
        factory = _FakeFactory(_FakeBrowserAgent(_FakeHistory(), error=RuntimeError('browser crashed')))

        with pytest.raises(RuntimeError, match='browser crashed'):
            await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        [killed] = kill_calls
        assert killed is factory.requests[0].browser_session

    async def test_no_result_reports_step_errors(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(step_errors=[None, 'timeout on step 2', 'element not found'])
        factory = _FakeFactory(_FakeBrowserAgent(history))

        result = await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        assert result == (
            'The browser agent stopped without producing a result (timeout on step 2; element not found).'
        )

    async def test_no_result_without_errors(self, kill_calls: list[BrowserSession]) -> None:
        factory = _FakeFactory(_FakeBrowserAgent(_FakeHistory()))

        result = await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        assert result == 'The browser agent stopped without producing a result (no further details).'

    async def test_unsuccessful_result_is_flagged(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(result='I could not log in.', success=False)
        factory = _FakeFactory(_FakeBrowserAgent(history))

        result = await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        assert result == ('The browser agent could not fully complete the task. Its final message: I could not log in.')

    async def test_structured_output_returned_as_json(self, kill_calls: list[BrowserSession]) -> None:
        facts = _Facts(name='Pro', price_usd=20)
        history = _FakeHistory(result='{"name": "Pro", "price_usd": 20}', success=True, structured=facts)
        factory = _FakeFactory(_FakeBrowserAgent(history))
        capability = BrowserUse[None](output_schema=_Facts, browser_agent=factory)

        result = await capability.get_toolset().browse_web('task')

        assert json.loads(result) == {'name': 'Pro', 'price_usd': 20}
        assert factory.requests[0].output_schema is _Facts

    async def test_structured_output_mismatch_raises_model_retry(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(result='not json', success=True, structured_error=_validation_error())
        factory = _FakeFactory(_FakeBrowserAgent(history))
        capability = BrowserUse[None](output_schema=_Facts, browser_agent=factory)

        with pytest.raises(ModelRetry, match='did not match the configured output schema'):
            await capability.get_toolset().browse_web('task')

        [killed] = kill_calls
        assert killed is factory.requests[0].browser_session

    async def test_structured_output_missing_falls_back_to_text(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(result='prose result', success=True)
        factory = _FakeFactory(_FakeBrowserAgent(history))
        capability = BrowserUse[None](output_schema=_Facts, browser_agent=factory)

        result = await capability.get_toolset().browse_web('task')

        assert result == 'prose result'

    async def test_default_factory_builds_real_browser_use_agent(
        self, monkeypatch: pytest.MonkeyPatch, kill_calls: list[BrowserSession]
    ) -> None:
        monkeypatch.setenv('ANONYMIZED_TELEMETRY', 'false')
        seen: dict[str, object] = {}

        async def record_run(self: object, max_steps: int = 500, **kwargs: object) -> _FakeHistory:
            seen['agent'] = self
            seen['max_steps'] = max_steps
            return _FakeHistory(result='browsed', success=True)

        monkeypatch.setattr(BrowserUseAgent, 'run', record_run)
        llm = _FakeChatModel()
        secrets: dict[str, str | dict[str, str]] = {'x_password': 'hunter2'}
        toolset = BrowserUse[None](
            llm=llm,
            max_steps=3,
            sensitive_data=secrets,
            extend_system_message='Never buy anything.',
            agent_settings=BrowserAgentSettings(use_judge=False, flash_mode=True, judge_llm='test'),
        ).get_toolset()

        result = await toolset.browse_web('check example.com')

        assert result == 'browsed'
        agent = seen['agent']
        assert isinstance(agent, BrowserUseAgent)
        assert agent.task == 'check example.com'
        assert agent.llm is llm
        assert agent.sensitive_data == secrets
        assert agent.settings.use_judge is False
        assert agent.settings.flash_mode is True
        assert isinstance(agent.judge_llm, PydanticAIChatModel)
        assert seen['max_steps'] == 3
        assert len(kill_calls) == 1

    async def test_pydantic_ai_model_string_is_wrapped(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await BrowserUse[None](llm='test', browser_agent=factory).get_toolset().browse_web('task')

        assert isinstance(factory.requests[0].llm, PydanticAIChatModel)

    async def test_settings_chat_models_arrive_resolved(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()
        capability = BrowserUse[None](
            agent_settings=BrowserAgentSettings(judge_llm='test'),
            browser_agent=factory,
        )

        await capability.get_toolset().browse_web('task')

        assert isinstance(factory.requests[0].settings.judge_llm, PydanticAIChatModel)


class TestSessionScope:
    async def test_agent_scope_reuses_one_session(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()
        capability = BrowserUse[None](session_scope='agent', browser_agent=factory)
        toolset = capability.get_toolset()

        await toolset.browse_web('first task')
        await toolset.browse_web('second task')

        first, second = factory.requests
        assert first.browser_session is second.browser_session
        # keep_alive stops browser_use.Agent from killing the shared session
        # at the end of each of its runs.
        assert first.browser_session.browser_profile.keep_alive is True
        assert kill_calls == []

        await capability.aclose()
        assert kill_calls == [first.browser_session]
        await capability.aclose()
        assert len(kill_calls) == 1

    async def test_agent_scope_error_resets_session(self, kill_calls: list[BrowserSession]) -> None:
        agent = _FakeBrowserAgent(_FakeHistory(result='done', success=True), error=RuntimeError('crash'))
        factory = _FakeFactory(agent)
        toolset = BrowserUse[None](session_scope='agent', browser_agent=factory).get_toolset()

        with pytest.raises(RuntimeError, match='crash'):
            await toolset.browse_web('first task')
        assert kill_calls == [factory.requests[0].browser_session]

        agent.error = None
        await toolset.browse_web('second task')
        assert factory.requests[1].browser_session is not factory.requests[0].browser_session

    async def test_agent_scope_schema_retry_keeps_session(self, kill_calls: list[BrowserSession]) -> None:
        agent = _FakeBrowserAgent(_FakeHistory(result='bad', success=True, structured_error=_validation_error()))
        factory = _FakeFactory(agent)
        capability = BrowserUse[None](output_schema=_Facts, session_scope='agent', browser_agent=factory)
        toolset = capability.get_toolset()

        with pytest.raises(ModelRetry):
            await toolset.browse_web('first task')
        # The run itself finished; only the result was rejected, so the shared
        # browser survives for the follow-up call.
        assert kill_calls == []

        agent.history = _FakeHistory(result='{"name": "Pro", "price_usd": 20}', success=True)
        agent.history.structured = _Facts(name='Pro', price_usd=20)
        await toolset.browse_web('second task')
        assert factory.requests[1].browser_session is factory.requests[0].browser_session
        await capability.aclose()

    async def test_capability_as_async_context_manager(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()
        async with BrowserUse[None](session_scope='agent', browser_agent=factory) as capability:
            await capability.get_toolset().browse_web('task')
            assert kill_calls == []
        assert kill_calls == [factory.requests[0].browser_session]

    async def test_aclose_before_any_call_is_a_no_op(self, kill_calls: list[BrowserSession]) -> None:
        capability = BrowserUse[None](session_scope='agent')
        await capability.aclose()
        assert kill_calls == []

    def test_toolset_is_cached(self) -> None:
        capability = BrowserUse[None]()
        assert capability.get_toolset() is capability.get_toolset()


class TestBrowserUse:
    def test_instructions_reference_the_tool(self) -> None:
        instructions = BrowserUse[None]().get_instructions()
        assert isinstance(instructions, str)
        assert '`browse_web`' in instructions

    def test_custom_guidance_replaces_default(self) -> None:
        assert BrowserUse[None](guidance='Delegate web tasks.').get_instructions() == 'Delegate web tasks.'

    def test_empty_guidance_disables_instructions(self) -> None:
        assert BrowserUse[None](guidance='').get_instructions() is None

    async def test_agent_run_returns_tool_result(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory('The answer is 42.')
        agent = Agent(TestModel(), capabilities=[BrowserUse(browser_agent=factory)])

        result = await agent.run('Find the answer on example.com.')

        parts = [
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'browse_web'
        ]
        assert [part.content for part in parts] == ['The answer is 42.']


class TestAgentSpec:
    def test_spec_schema_includes_browser_use(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([BrowserUse])
        assert 'BrowserUse' in json.dumps(schema)

    def test_from_spec_builds_capability(self) -> None:
        capability = BrowserUse[None].from_spec(
            allowed_domains=['example.com'],
            headless=False,
            max_steps=10,
            use_vision='auto',
            sensitive_data={'x_user': 'kacper'},
            extend_system_message='Stay on the English site.',
            session_scope='agent',
            cdp_url='http://localhost:9222',
            guidance='Delegate.',
        )
        assert capability.allowed_domains == ['example.com']
        assert capability.headless is False
        assert capability.max_steps == 10
        assert capability.use_vision == 'auto'
        assert capability.sensitive_data == {'x_user': 'kacper'}
        assert capability.extend_system_message == 'Stay on the English site.'
        assert capability.session_scope == 'agent'
        assert capability.cdp_url == 'http://localhost:9222'
        assert capability.guidance == 'Delegate.'
        assert capability.llm is None
        assert capability.browser_profile is None
        assert capability.output_schema is None
        assert capability.agent_settings is None
        assert capability.browser_agent is None

    def test_agent_loads_from_spec_file(self, tmp_path: Path) -> None:
        spec = tmp_path / 'agent.yaml'
        spec.write_text('model: test\ncapabilities:\n  - BrowserUse:\n      max_steps: 5\n')
        agent = Agent.from_file(spec, custom_capability_types=[BrowserUse])
        assert isinstance(agent, Agent)
