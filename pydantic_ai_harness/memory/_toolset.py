"""Memory toolset: three file-shaped tools over a path-addressed `MemoryStore`."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset

if TYPE_CHECKING:
    from pydantic_ai_harness.memory._capability import Memory
    from pydantic_ai_harness.memory._store import MemoryStore

MAIN_FILENAME = 'MEMORY.md'
"""The main notebook file, injected into the system prompt every request."""

_FILENAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}')
_CHARS_PER_TOKEN = 4
"""Approximate chars-per-token heuristic for the optional injection token budget."""


def normalize_filename(file: str) -> str:
    """Validate a model-supplied memory filename and normalize it to `<name>.md`.

    Raises:
        ModelRetry: If the name is empty, contains path separators or `..`,
            or otherwise cannot name a memory file.
    """
    name = file.strip()
    if name and not name.endswith('.md'):
        name = f'{name}.md'
    if not _FILENAME_RE.fullmatch(name) or '..' in name:
        raise ModelRetry(
            f'{file!r} is not a valid memory filename -- use a short name like "postgres-migration.md" '
            '(letters, digits, dots, dashes; no slashes).'
        )
    return name


async def list_subfiles(store: MemoryStore, scope: str) -> list[str]:
    """Return the scope's memory filenames (excluding the main notebook), sorted."""
    prefix = f'{scope}/'
    names: list[str] = []
    for path in await store.list_paths(prefix):
        name = path.removeprefix(prefix)
        if '/' not in name and name != MAIN_FILENAME and name.endswith('.md'):
            names.append(name)
    return sorted(names)


def _clip_to_budget(lines: list[str], max_lines: int, max_tokens: int | None) -> tuple[list[str], int]:
    """Keep the trailing lines that fit the budget; return them plus the dropped count.

    `max_tokens` (approximate, via `_CHARS_PER_TOKEN`) takes precedence over
    `max_lines`. At least one line is kept when any exist.
    """
    if max_tokens is None:
        kept = min(len(lines), max_lines)
    else:
        budget = max_tokens * _CHARS_PER_TOKEN
        used = 0
        kept = 0
        for line in reversed(lines):
            used += len(line) + 1  # +1 approximates the joining newline
            if used > budget and kept:
                break
            kept += 1
    dropped = len(lines) - kept
    return lines[dropped:], dropped


def render_memory_prompt(
    main_content: str,
    subfiles: list[str],
    *,
    agent_name: str,
    guidance: str,
    max_lines: int,
    max_tokens: int | None,
) -> str:
    """Render the injected memory section: guidance, the (budgeted) notebook, and the file list.

    `MEMORY.md` is injected in full; the line/token budget is a guard rail for
    a notebook that has grown out of hand -- the most recent lines are kept and
    a truncation marker points the model at `read_memory` for the rest. The
    subfile list is generated from the store, so it is always ground truth
    (a stale mention inside `MEMORY.md` cannot fake a file into existence).
    """
    lines, dropped = _clip_to_budget(main_content.rstrip().splitlines(), max_lines, max_tokens)
    if dropped:
        marker = f'... [{dropped} earlier lines -- read_memory("{MAIN_FILENAME}") for the full notebook] ...'
        lines = [marker, *lines]
    sections = [f'## Agent Memory ({agent_name})']
    if guidance:
        sections.append(guidance)
    if lines:
        sections.append(f'### {MAIN_FILENAME}\n\n' + '\n'.join(lines))
    if subfiles:
        listing = '\n'.join(f'- {name}' for name in subfiles)
        sections.append(f'### Other memory files (read with `read_memory`)\n\n{listing}')
    return '\n\n'.join(sections)


def _apply_write(existing: str | None, content: str, old_text: str | None, name: str) -> tuple[str, str]:
    """Compute a file's new content: append when `old_text` is `None`, else replace it uniquely.

    Returns the updated content and the verb for the tool reply.

    Raises:
        ModelRetry: When the replacement target is missing or ambiguous --
            the file is left untouched.
    """
    if old_text is None:
        if existing is not None and existing.strip():
            return f'{existing.rstrip()}\n{content.rstrip()}\n', 'Appended to'
        return f'{content.rstrip()}\n', 'Appended to'
    if existing is None:
        raise ModelRetry(f'There is no memory file named {name!r} to edit -- omit `old_text` to create it.')
    occurrences = existing.count(old_text) if old_text else 0
    if occurrences == 0:
        raise ModelRetry(
            f'`old_text` was not found in {name!r} -- call `read_memory("{name}")` to see its current content.'
        )
    if occurrences > 1:
        raise ModelRetry(
            f'`old_text` appears {occurrences} times in {name!r}; it must match exactly once. '
            'Add surrounding context to make it unique.'
        )
    return existing.replace(old_text, content, 1), 'Updated'


class MemoryToolset(FunctionToolset[AgentDepsT]):
    """`write_memory` / `read_memory` / `delete_memory` over the run's memory scope.

    Every tool operates strictly inside the run's resolved
    `{namespace}/{agent_name}/` scope; writes are serialized per scope by the
    capability's process-wide lock. Model-correctable problems raise
    `ModelRetry`; storage failures propagate and abort the run (fail loudly --
    the injection path in `Memory.get_instructions` is the only place that
    fails soft).
    """

    def __init__(self, capability: Memory[AgentDepsT]) -> None:
        super().__init__()
        self._capability = capability
        self.add_function(self.write_memory, name='write_memory')
        self.add_function(self.read_memory, name='read_memory')
        self.add_function(self.delete_memory, name='delete_memory')

    def _path(self, scope: str, file: str) -> str:
        return f'{scope}/{file}'

    async def write_memory(
        self,
        ctx: RunContext[AgentDepsT],
        content: str,
        file: str = MAIN_FILENAME,
        old_text: str | None = None,
    ) -> str:
        """Write to your persistent memory: append by default, or replace `old_text` with `content`.

        Without `old_text` the content is appended to the file (creating it if
        needed) -- use this to add new facts or notes. With `old_text`, the
        exact text is replaced by `content` -- this covers editing, correcting,
        and deleting (replace with less, or with an empty string). `old_text`
        must match exactly once; if it does not, nothing is changed.

        Keep MEMORY.md curated: short durable facts as plain bullet lines,
        longer or evolving topics in their own file (referenced from
        MEMORY.md). Update or remove entries instead of duplicating them, and
        convert relative dates to absolute when saving.

        Args:
            ctx: The run context (injected by the framework).
            content: Text to append, or the replacement for `old_text`.
            file: Memory filename (default: the main `MEMORY.md`).
            old_text: Exact existing text to replace (must be unique in the file).
        """
        capability = self._capability
        name = normalize_filename(file)
        if old_text is None and not content.strip():
            raise ModelRetry('Nothing to write -- pass the text to append, or `old_text` to replace.')
        store, scope = capability.resolve_scope(ctx)
        path = self._path(scope, name)
        async with capability.scope_lock(scope):
            existing = await store.read(path)
            updated, action = _apply_write(existing, content, old_text, name)
            if len(updated) > capability.max_memory_size:
                raise ModelRetry(
                    f'{name!r} would grow to {len(updated)} characters; the limit is '
                    f'{capability.max_memory_size}. Split the content into separate memory files.'
                )
            await store.write(path, updated)
        return f'{action} {name}.'

    async def read_memory(self, ctx: RunContext[AgentDepsT], file: str) -> str:
        """Read the full content of one memory file.

        Your memory files are listed in the injected memory section; read one
        when it looks relevant to the task at hand, before acting. Memory
        reflects what was true when written -- verify anything volatile.

        Args:
            ctx: The run context (injected by the framework).
            file: Memory filename as shown in the memory section.
        """
        name = normalize_filename(file)
        store, scope = self._capability.resolve_scope(ctx)
        content = await store.read(self._path(scope, name))
        if content is None:
            raise ModelRetry(
                f'There is no memory file named {name!r} -- the existing files are listed in your memory section.'
            )
        return content

    async def delete_memory(self, ctx: RunContext[AgentDepsT], file: str) -> str:
        """Delete one memory file that is no longer worth keeping.

        The main `MEMORY.md` cannot be deleted -- edit it with `write_memory`
        instead. Remember to also remove any reference to the deleted file
        from `MEMORY.md`.

        Args:
            ctx: The run context (injected by the framework).
            file: Memory filename to delete.
        """
        capability = self._capability
        name = normalize_filename(file)
        if name == MAIN_FILENAME:
            raise ModelRetry(
                f'{MAIN_FILENAME} is your main notebook and cannot be deleted -- edit it with `write_memory`.'
            )
        store, scope = capability.resolve_scope(ctx)
        path = self._path(scope, name)
        async with capability.scope_lock(scope):
            existed = await store.read(path) is not None
            await store.delete(path)
        return f'Deleted {name}.' if existed else f'There is no memory file named {name!r}.'
