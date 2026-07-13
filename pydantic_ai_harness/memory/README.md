# Memory

Persistent agent memory across sessions: an injected `MEMORY.md` notebook plus separate memory files read on demand.

> [!NOTE]
> Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.memory import Memory
> ```

## The problem

Agents forget everything between sessions. Naive fixes either dump the whole memory into every prompt (context cost grows with memory size) or bolt a key-value blob onto the run and hope the model queries it. Multi-user applications add a second problem: one agent instance serves many users, and nothing may ever let the model address another tenant's memory.

## The solution

A two-tier notebook, with the model deciding which tier fits:

- **`MEMORY.md`** is the agent's main notebook -- injected into the system prompt **every request**, holding short durable facts as plain bullet lines.
- **Longer or evolving topics** live in separate markdown files. Only their *names* are injected (the list is generated from the store, so it is always ground truth); their content is read on demand with `read_memory` when relevant.

One `write_memory` tool covers everything: append by default, or pass `old_text` for a unique exact-string replacement -- which is editing, correcting, and deleting in a single primitive. A failed match writes nothing.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.memory import FileStore, Memory

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Memory(store=FileStore('.agent-memory'))],
)
```

## Tools

| Tool | What it does |
| --- | --- |
| `write_memory(content, file='MEMORY.md', old_text=None)` | Append `content` (creating the file if needed), or replace a unique `old_text` with it -- covers add, edit, correct, and delete |
| `read_memory(file)` | Read the full content of one memory file |
| `delete_memory(file)` | Delete a memory file (the main `MEMORY.md` is protected) |

## What the model sees each request

```text
## Agent Memory (main)

<usage guidance: memory is background context, not instructions; keep it curated>

### MEMORY.md

- the user is called Kacper and prefers uv over pip
- decision 2026-07-11: memory lives in pydantic-ai-harness, not pydantic-deep

### Other memory files (read with `read_memory`)

- postgres-migration.md
```

`MEMORY.md` is injected in full; `max_lines`/`max_tokens` are a guard rail for a notebook that has grown out of hand (the most recent lines are kept, with a truncation marker pointing at `read_memory`).

## Persistence

The default `InMemoryStore` lives for the **process only** -- memories survive across `Agent.run` calls but not restarts. Four stores ship in the box; anything else is a four-method `MemoryStore` protocol away (path = key column, namespace = key prefix).

| Store | Fits | Notes |
| --- | --- | --- |
| `InMemoryStore()` | tests, ephemeral agents | default; process-lifetime |
| `FileStore(directory)` | CLI / local / single-host | atomic writes, path-jailed inside its directory, blocking IO off the event loop |
| `SqliteMemoryStore(database=...)` | single-host apps wanting one durable file | stdlib `sqlite3`, WAL; or pass a caller-owned `connection=` (must be `check_same_thread=False`) |
| `PostgresMemoryStore(pool)` | production multi-user apps | caller-owned pool; upserts are atomic per statement, so cross-process writers are safe |

```python
Memory(store=FileStore('.agent-memory'))
Memory(store=SqliteMemoryStore(database='.agent-memory.db'))
```

`PostgresMemoryStore` is deliberately **driver-agnostic**: it talks to a minimal `PostgresPool` protocol (`execute` / `fetchval` / `fetch`, `$1`-style parameters), so `pydantic-ai-harness` gains no database dependency -- an `asyncpg.Pool` satisfies it out of the box:

```python
import asyncpg

from pydantic_ai_harness.memory import Memory, PostgresMemoryStore

pool = await asyncpg.create_pool('postgres://...')
agent_memory = Memory(
    store=PostgresMemoryStore(pool),           # optional: table='agent_memory'
    namespace=lambda ctx: ctx.deps.user_id,
)
```

The pool is caller-owned -- create it at app startup, close it at shutdown; the store never manages connection lifecycle.

## Multi-user applications

The tenant is resolved **per run** from your own deps and is never a tool argument, so the model cannot express -- let alone reach -- another tenant's memory:

```python
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai_harness.memory import FileStore, Memory


@dataclass
class AppDeps:
    user_id: str


agent = Agent(
    'anthropic:claude-sonnet-4-6',
    deps_type=AppDeps,
    capabilities=[
        Memory(
            store=FileStore('/var/lib/myapp/memory'),
            namespace=lambda ctx: ctx.deps.user_id,
        )
    ],
)
```

On-disk layout: `{namespace}/{agent_name}/MEMORY.md` plus one `.md` file per topic. `agent_name` separates agents (e.g. subagents) within a namespace.

Memory content is model-written and re-enters future system prompts. Per-tenant isolation bounds prompt-injection-via-memory to the tenant that wrote it, and the injected section is framed as records, not instructions. Treat memory content as untrusted input if you render it anywhere else. Each file is capped at `max_memory_size` (default 64 KB) so a hostile conversation cannot balloon storage.

## Configuration

```python
Memory(
    store=FileStore('.agent-memory'),  # default: InMemoryStore() (process-lifetime)
    store_resolver=None,               # per-run store resolution from ctx (wins over store)
    agent_name='main',                 # subtree per agent within a namespace
    namespace='',                      # str or (ctx) -> str, resolved per run
    max_lines=200,                     # injection guard: max MEMORY.md lines
    max_tokens=None,                   # optional approximate token budget (wins over max_lines)
    max_memory_size=65_536,            # max characters per memory file
    guidance=None,                     # None = default usage guidance; '' = bare notebook only
)
```

## Agent spec

```yaml
capabilities:
  - Memory: {backend: file, directory: .agent-memory, agent_name: main}
```

`backend: memory` (default), `backend: file`, or `backend: sqlite` (with `database`); any other value raises rather than silently losing durability. Callables (`namespace`, `store_resolver`) and `PostgresMemoryStore` (a live pool) are not spec-constructible.

## Composing with FileSystem

Point the store inside a `FileSystem` root and the agent can also browse its memories with its normal file tools -- one directory, one source of truth:

```python
from pathlib import Path

from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.memory import FileStore, Memory

workspace = Path('/srv/agent-workspace')
capabilities = [
    FileSystem(root_dir=workspace),
    Memory(store=FileStore(workspace / '.memory')),
]
```

To force all memory edits through the memory tools, add `protected_patterns=['.memory/*']` to `FileSystem`.

## Prompt-cache note

The memory section is re-rendered every request, but it only changes when the agent writes, so cache invalidation is rare in practice. A cache-stable mode (ephemeral tail reminder behind a `CachePoint`, as in `Planning`) is a possible follow-up.

## Related

- `pydantic_ai_harness.context` -- read-only sibling: loads `CLAUDE.md`/`AGENTS.md`-style instruction files. Deployment-fixed, always-on facts belong there (or in your agent's instructions), not in Memory.
- [Dependencies](https://ai.pydantic.dev/dependencies/) and [Capabilities](https://ai.pydantic.dev/capabilities/) in the Pydantic AI docs.
