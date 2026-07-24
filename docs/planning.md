---
title: Planning
description: Give an agent a structured, self-updating task list -- with a cache-safe live reminder, optional persistence, subtasks, dependencies, and events.
---

# Planning

`Planning` gives the model a structured, self-updating task list through a small toolset -- and surfaces the current plan back to the model every turn without ever invalidating the prompt cache. It can stay in memory for a single run or persist to SQLite/Postgres, break steps into subtasks with dependencies, and emit events from granular changes.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/planning/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

Long agentic runs drift: the model loses track of what it set out to do and what's left. The usual fix -- keep a running plan and re-inject it into the system prompt each turn -- invalidates the prompt cache. The system prompt sits at the front of the request, so every plan edit changes the cached prefix and forces the whole conversation to be re-processed at full token price.

## The solution

The model owns the plan through the `planning` toolset. The current plan is surfaced back as an ephemeral reminder appended to the tail of each request, behind a cache breakpoint:

- The reminder is added after the durable history is persisted, so it reaches the model but is never written to `message_history`. No reminders accumulate across turns.
- A `CachePoint` is placed immediately before the reminder, so the cached prefix (tools + system + real conversation) stays byte-identical turn over turn. Only the reminder falls outside the cache.

## Usage

Construct an `Agent` with `Planning()` in its `capabilities`. The tools are registered automatically and static usage guidance is added to the system prompt:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.planning import Planning

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning()])

result = agent.run_sync('Refactor the auth module and add tests.')
print(result.output)
```

## The tools

| Tool | Purpose |
|---|---|
| `write_plan(items)` | Create or replace the full plan (whole-list replacement). |
| `read_plan()` | Read the current plan with step ids and a progress summary. |
| `add_task(content, active_form)` | Append a single `pending` step. |
| `update_task_status(task_id, status)` | Move one step between statuses by id. |
| `update_task_statuses(updates)` | Apply several status changes in one call, validated all-or-nothing. |
| `remove_task(task_id)` | Delete a step by id. |

With `enable_subtasks=True` you also get `add_subtask`, `set_dependency`, and `get_available_tasks`, plus the `blocked` status and a `hierarchical` view in `read_plan`.

## Persistence

By default the plan is a fresh, isolated in-memory plan per run. Pass a `store` to persist it:

```python
from pydantic_ai_harness.planning import Planning, SqlitePlanStore

planning = Planning(store=SqlitePlanStore('plan.db', session='user-123'))
```

Built-in stores are `InMemoryPlanStore`, `SqlitePlanStore`, `PostgresPlanStore` (over a caller-owned asyncpg pool), and `RedisPlanStore` (over a caller-owned `redis.asyncio` client) -- so the harness needs no database driver. Any `PlanStore` implementation works, and `store_resolver` selects one per run. `SqlitePlanStore` requires a file-backed database; use `InMemoryPlanStore` for ephemeral plans rather than `':memory:'`.

## Events

Attach a `PlanEventEmitter` to a store to react to changes:

```python
from pydantic_ai_harness.planning import InMemoryPlanStore, PlanEventEmitter

emitter = PlanEventEmitter()

@emitter.on_completed
async def announce(event):
    print('done:', event.item.content)

store = InMemoryPlanStore(event_emitter=emitter)
```

Events come from granular tools (`add_task`, `update_task_status`, `add_subtask`, ...). `write_plan` is a bulk whole-plan replacement and is **event-silent**, so a UI driven purely by events should also read the plan after a run, or steer the model toward granular tools when it needs live event coverage.

## Caching guarantee

The plan is never injected into the system prompt or instructions. Static usage guidance goes there (cache-stable); only the mutable plan rides the ephemeral tail reminder, which lives solely in the per-request copy and is never persisted. Set `inject=False` to disable it. `CachePoint` is supported on Anthropic and Amazon Bedrock; on providers without prompt caching it is simply ignored.

## Agent spec (YAML/JSON)

`Planning` works with Pydantic AI's [agent spec](/ai/core-concepts/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Planning: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.planning import Planning

agent = Agent.from_file('agent.yaml', custom_capability_types=[Planning])
result = agent.run_sync('...')
print(result.output)
```

## Further reading

- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- [Code Mode](code-mode.md) -- another prompt-cache-aware harness capability

## API reference

::: pydantic_ai_harness.planning.Planning
