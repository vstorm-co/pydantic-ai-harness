"""Tests for the Memory capability."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.memory import (
    FileStore,
    InMemoryStore,
    Memory,
    MemoryStore,
    MemoryToolset,
    PostgresMemoryStore,
    PostgresPool,
    SqliteMemoryStore,
)
from pydantic_ai_harness.memory._store import validate_store_path
from pydantic_ai_harness.memory._toolset import (
    list_subfiles,
    normalize_filename,
    render_memory_prompt,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _ctx() -> RunContext[None]:
    return MagicMock()


def _files(capability: Memory[None]) -> dict[str, str]:
    store = capability.store
    assert isinstance(store, InMemoryStore)
    return store.files


async def _render(capability: Memory[None]) -> str | None:
    instructions = capability.get_instructions()
    assert callable(instructions)
    render = cast('Callable[[RunContext[None]], Awaitable[str | None]]', instructions)
    return await render(_ctx())


@dataclass
class ExplodingStore:
    """Store whose methods raise, for exercising the IO-failure paths."""

    fail_read: bool = False
    fail_write: bool = False
    fail_delete: bool = False
    fail_list: bool = False
    inner: InMemoryStore = field(default_factory=InMemoryStore)

    async def read(self, path: str) -> str | None:
        if self.fail_read:
            raise OSError('read boom')
        return await self.inner.read(path)

    async def write(self, path: str, content: str) -> None:
        if self.fail_write:
            raise OSError('write boom')
        await self.inner.write(path, content)

    async def delete(self, path: str) -> None:
        if self.fail_delete:
            raise OSError('delete boom')
        await self.inner.delete(path)

    async def list_paths(self, prefix: str = '') -> list[str]:
        if self.fail_list:
            raise OSError('list boom')
        return await self.inner.list_paths(prefix)


class TestExplodingStorePassthrough:
    async def test_non_failing_methods_delegate(self) -> None:
        store = ExplodingStore()
        await store.write('a/x.md', 'one')
        assert await store.read('a/x.md') == 'one'
        assert await store.list_paths('a/') == ['a/x.md']
        await store.delete('a/x.md')
        assert await store.read('a/x.md') is None


class TestNormalizeFilename:
    def test_appends_md_suffix(self) -> None:
        assert normalize_filename('postgres-migration') == 'postgres-migration.md'
        assert normalize_filename('notes.md') == 'notes.md'

    def test_strips_whitespace(self) -> None:
        assert normalize_filename('  notes  ') == 'notes.md'

    @pytest.mark.parametrize('bad', ['', '  ', 'a/b', '../x', 'nested/file.md', '.hidden', 'x' * 100])
    def test_rejects(self, bad: str) -> None:
        with pytest.raises(ModelRetry, match='not a valid memory filename'):
            normalize_filename(bad)


class TestRenderPrompt:
    def test_guidance_notebook_and_files(self) -> None:
        result = render_memory_prompt(
            '- a fact\n', ['topic.md'], agent_name='main', guidance='G', max_lines=10, max_tokens=None
        )
        assert result == (
            '## Agent Memory (main)\n\nG\n\n### MEMORY.md\n\n- a fact\n\n'
            '### Other memory files (read with `read_memory`)\n\n- topic.md'
        )

    def test_no_guidance_no_files(self) -> None:
        result = render_memory_prompt('- a fact\n', [], agent_name='main', guidance='', max_lines=10, max_tokens=None)
        assert result == '## Agent Memory (main)\n\n### MEMORY.md\n\n- a fact'

    def test_files_only(self) -> None:
        result = render_memory_prompt('', ['topic.md'], agent_name='main', guidance='', max_lines=10, max_tokens=None)
        assert result == '## Agent Memory (main)\n\n### Other memory files (read with `read_memory`)\n\n- topic.md'

    def test_truncation_keeps_tail(self) -> None:
        content = ''.join(f'- fact {i}\n' for i in range(5))
        result = render_memory_prompt(content, [], agent_name='main', guidance='', max_lines=2, max_tokens=None)
        assert '... [3 earlier lines -- read_memory("MEMORY.md") for the full notebook] ...' in result
        assert '- fact 4' in result
        assert '- fact 0' not in result

    def test_token_budget_wins_and_keeps_at_least_one(self) -> None:
        content = ''.join(f'- fact {i} {"x" * 40}\n' for i in range(4))
        result = render_memory_prompt(content, [], agent_name='main', guidance='', max_lines=100, max_tokens=1)
        assert '- fact 3' in result
        assert '3 earlier lines' in result

    def test_token_budget_fits_all(self) -> None:
        result = render_memory_prompt('- a fact\n', [], agent_name='main', guidance='', max_lines=100, max_tokens=1000)
        assert 'earlier lines' not in result

    def test_blank_interior_lines_survive(self) -> None:
        result = render_memory_prompt(
            '# Facts\n\n- one\n\n\n', [], agent_name='main', guidance='', max_lines=10, max_tokens=None
        )
        assert result.endswith('### MEMORY.md\n\n# Facts\n\n- one')


class TestValidateStorePath:
    @pytest.mark.parametrize('bad', ['', '../x', 'a/../b', 'a//b', '/abs', 'a/', 'x' * 201])
    def test_rejects(self, bad: str) -> None:
        with pytest.raises(ValueError):
            validate_store_path(bad)

    def test_accepts(self) -> None:
        validate_store_path('tenant-1/main/MEMORY.md')


class TestInMemoryStore:
    async def test_crud_and_listing(self) -> None:
        store = InMemoryStore()
        assert await store.read('a/x.md') is None
        await store.write('a/x.md', 'one')
        await store.write('b/y.md', 'two')
        assert await store.read('a/x.md') == 'one'
        assert await store.list_paths('a/') == ['a/x.md']
        await store.delete('a/x.md')
        await store.delete('a/x.md')  # idempotent
        assert await store.list_paths() == ['b/y.md']


class TestFileStore:
    async def test_crud_and_listing(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path / 'mem')
        assert await store.read('main/x.md') is None
        assert await store.list_paths() == []
        await store.write('main/x.md', 'one')
        await store.write('main/y.md', 'two')
        assert await store.read('main/x.md') == 'one'
        assert await store.list_paths('main/') == ['main/x.md', 'main/y.md']
        assert await store.list_paths('other/') == []
        await store.delete('main/x.md')
        await store.delete('main/x.md')  # idempotent
        assert await store.list_paths() == ['main/y.md']

    async def test_read_of_directory_is_none(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path)
        await store.write('main/x.md', 'one')
        assert await store.read('main') is None

    async def test_rejects_traversal(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path / 'mem')
        with pytest.raises(ValueError):
            await store.read('../outside.md')

    async def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        root = tmp_path / 'mem'
        root.mkdir()
        outside = tmp_path / 'outside'
        outside.mkdir()
        (outside / 'secret.md').write_text('s', encoding='utf-8')
        os.symlink(outside, root / 'link')
        with pytest.raises(ValueError):
            await FileStore(root).read('link/secret.md')

    async def test_listing_skips_directories(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path)
        await store.write('a/deep/x.md', 'one')
        assert await store.list_paths() == ['a/deep/x.md']


class TestSqliteMemoryStore:
    async def test_crud_and_listing(self, tmp_path: Path) -> None:
        store = SqliteMemoryStore(database=tmp_path / 'memory.db')
        assert await store.read('main/x.md') is None
        assert await store.list_paths() == []
        await store.write('main/x.md', 'one')
        await store.write('main/x.md', 'one-updated')  # upsert
        await store.write('other/y.md', 'two')
        assert await store.read('main/x.md') == 'one-updated'
        assert await store.list_paths('main/') == ['main/x.md']
        await store.delete('main/x.md')
        await store.delete('main/x.md')  # idempotent
        assert await store.list_paths() == ['other/y.md']

    async def test_caller_owned_connection(self) -> None:
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        try:
            store = SqliteMemoryStore(connection=connection)
            await store.write('main/x.md', 'one')
            assert await store.read('main/x.md') == 'one'
        finally:
            connection.close()

    def test_requires_exactly_one_of_database_or_connection(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            SqliteMemoryStore()
        connection = sqlite3.connect(':memory:')
        try:
            with pytest.raises(ValueError, match='exactly one'):
                SqliteMemoryStore(database=':memory:', connection=connection)
        finally:
            connection.close()

    async def test_like_wildcards_in_prefix_are_literal(self, tmp_path: Path) -> None:
        # Scope segments may legally contain `_` (a LIKE wildcard) -- it must not
        # match arbitrary characters, or one tenant could list another's paths.
        store = SqliteMemoryStore(database=tmp_path / 'memory.db')
        await store.write('user_1/main/MEMORY.md', 'a')
        await store.write('userX1/main/MEMORY.md', 'b')
        assert await store.list_paths('user_1/') == ['user_1/main/MEMORY.md']

    def test_satisfies_protocol(self, tmp_path: Path) -> None:
        assert isinstance(SqliteMemoryStore(database=tmp_path / 'memory.db'), MemoryStore)

    @pytest.mark.parametrize('bad', [':memory:', ''])
    def test_rejects_in_memory_database(self, bad: str) -> None:
        # Per-call connections would each see a fresh empty in-memory database.
        with pytest.raises(ValueError, match='InMemoryStore'):
            SqliteMemoryStore(database=bad)

    async def test_prefix_listing_is_case_sensitive(self, tmp_path: Path) -> None:
        # SQLite LIKE is case-insensitive for ASCII by default -- a LIKE-based
        # listing would leak `alice/` paths to the `Alice/` tenant.
        store = SqliteMemoryStore(database=tmp_path / 'memory.db')
        await store.write('Alice/main/MEMORY.md', 'a')
        await store.write('alice/main/MEMORY.md', 'b')
        assert await store.list_paths('Alice/') == ['Alice/main/MEMORY.md']
        assert await store.list_paths('alice/') == ['alice/main/MEMORY.md']


@dataclass
class FakePool:
    """Asyncpg-shaped fake: implements `PostgresPool` over a dict."""

    rows: dict[str, str] = field(default_factory=dict[str, str])
    statements: list[str] = field(default_factory=list[str])

    async def execute(self, query: str, *args: object) -> object:
        self.statements.append(query)
        if query.startswith('CREATE TABLE'):
            return 'CREATE TABLE'
        if query.startswith('INSERT'):
            path, content = args
            assert isinstance(path, str) and isinstance(content, str)
            self.rows[path] = content
            return 'INSERT 0 1'
        assert query.startswith('DELETE')
        path = args[0]
        assert isinstance(path, str)
        self.rows.pop(path, None)
        return 'DELETE 1'

    async def fetchval(self, query: str, *args: object) -> object:
        self.statements.append(query)
        path = args[0]
        assert isinstance(path, str)
        return self.rows.get(path)

    async def fetch(self, query: str, *args: object) -> list[tuple[str]]:
        self.statements.append(query)
        prefix = args[0]
        assert isinstance(prefix, str)
        return [(path,) for path in sorted(self.rows) if path.startswith(prefix)]


class TestPostgresMemoryStore:
    async def test_crud_and_listing(self) -> None:
        store = PostgresMemoryStore(FakePool())
        assert await store.read('main/x.md') is None
        await store.write('main/x.md', 'one')
        await store.write('main/x.md', 'one-updated')  # upsert
        await store.write('other/y.md', 'two')
        assert await store.read('main/x.md') == 'one-updated'
        assert await store.list_paths('main/') == ['main/x.md']
        await store.delete('main/x.md')
        await store.delete('main/x.md')  # idempotent
        assert await store.list_paths() == ['other/y.md']

    async def test_schema_created_once_and_statements_parametrized(self) -> None:
        pool = FakePool()
        store = PostgresMemoryStore(pool, table='memories')
        await store.write('a', '1')
        await store.read('a')
        await store.list_paths()
        creates = [statement for statement in pool.statements if statement.startswith('CREATE TABLE')]
        assert len(creates) == 1
        assert 'memories' in creates[0]
        assert all('$1' in statement for statement in pool.statements if not statement.startswith('CREATE'))

    def test_rejects_invalid_table_name(self) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            PostgresMemoryStore(FakePool(), table='memories; DROP TABLE users')

    def test_satisfies_protocols(self) -> None:
        assert isinstance(FakePool(), PostgresPool)
        assert isinstance(PostgresMemoryStore(FakePool()), MemoryStore)


class TestResolveScope:
    def test_default_scope(self) -> None:
        store, scope = Memory[None]().resolve_scope(_ctx())
        assert isinstance(store, InMemoryStore)
        assert scope == 'main'

    def test_static_namespace(self) -> None:
        _, scope = Memory[None](namespace='tenant-1').resolve_scope(_ctx())
        assert scope == 'tenant-1/main'

    def test_callable_namespace_and_multi_segment(self) -> None:
        capability = Memory[None](namespace=lambda ctx: 'user-1/conv-2')
        _, scope = capability.resolve_scope(_ctx())
        assert scope == 'user-1/conv-2/main'

    def test_callable_namespace_error_propagates(self) -> None:
        def broken(ctx: RunContext[None]) -> str:
            raise RuntimeError('resolver bug')

        with pytest.raises(RuntimeError):
            Memory[None](namespace=broken).resolve_scope(_ctx())

    def test_invalid_segment_rejected(self) -> None:
        with pytest.raises(ValueError):
            Memory[None](namespace='..').resolve_scope(_ctx())
        with pytest.raises(ValueError):
            Memory[None](agent_name='a b').resolve_scope(_ctx())

    @pytest.mark.parametrize('degenerate', ['/victim', 'victim/', 'victim//other', '/'])
    def test_degenerate_namespaces_rejected_not_normalized(self, degenerate: str) -> None:
        # Silently dropping empty segments would collapse `victim`, `/victim`, and
        # `victim//` into one scope and merge tenants the app believes are distinct.
        with pytest.raises(ValueError, match='invalid memory path'):
            Memory[None](namespace=degenerate).resolve_scope(_ctx())

    def test_store_resolver_wins(self) -> None:
        special = InMemoryStore()
        capability = Memory[None](store_resolver=lambda ctx: special)
        store, _ = capability.resolve_scope(_ctx())
        assert store is special


class TestWriteMemory:
    async def test_append_creates_main(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        result = await toolset.write_memory(_ctx(), '- user is Kacper')
        assert result == 'Appended to MEMORY.md.'
        assert _files(capability)['main/MEMORY.md'] == '- user is Kacper\n'

    async def test_append_extends_existing(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        await toolset.write_memory(_ctx(), '- first')
        await toolset.write_memory(_ctx(), '- second')
        assert _files(capability)['main/MEMORY.md'] == '- first\n- second\n'

    async def test_append_creates_subfile_without_md_suffix(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        result = await toolset.write_memory(_ctx(), 'Topic body.', file='postgres-migration')
        assert result == 'Appended to postgres-migration.md.'
        assert _files(capability)['main/postgres-migration.md'] == 'Topic body.\n'

    async def test_replace_unique(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        await toolset.write_memory(_ctx(), '- likes pip')
        result = await toolset.write_memory(_ctx(), '- likes uv', old_text='- likes pip')
        assert result == 'Updated MEMORY.md.'
        assert _files(capability)['main/MEMORY.md'] == '- likes uv\n'

    async def test_replace_with_empty_deletes_text(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        await toolset.write_memory(_ctx(), '- keep\n- drop')
        await toolset.write_memory(_ctx(), '', old_text='\n- drop')
        assert _files(capability)['main/MEMORY.md'] == '- keep\n'

    async def test_replace_not_found_retries_and_writes_nothing(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        await toolset.write_memory(_ctx(), '- a fact')
        before = dict(_files(capability))
        with pytest.raises(ModelRetry, match='was not found'):
            await toolset.write_memory(_ctx(), 'x', old_text='missing')
        assert _files(capability) == before

    async def test_replace_ambiguous_retries(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        await toolset.write_memory(_ctx(), 'x y x')
        with pytest.raises(ModelRetry, match='appears 2 times'):
            await toolset.write_memory(_ctx(), 'z', old_text='x')

    async def test_replace_in_missing_file_retries(self) -> None:
        toolset = MemoryToolset(Memory[None]())
        with pytest.raises(ModelRetry, match='omit `old_text` to create it'):
            await toolset.write_memory(_ctx(), 'x', file='nope', old_text='y')

    async def test_empty_append_retries(self) -> None:
        with pytest.raises(ModelRetry, match='Nothing to write'):
            await MemoryToolset(Memory[None]()).write_memory(_ctx(), '   ')

    async def test_invalid_filename_retries(self) -> None:
        with pytest.raises(ModelRetry, match='not a valid memory filename'):
            await MemoryToolset(Memory[None]()).write_memory(_ctx(), 'x', file='../etc')

    async def test_size_cap_steers_to_split(self) -> None:
        toolset = MemoryToolset(Memory[None](max_memory_size=10))
        with pytest.raises(ModelRetry, match='Split the content'):
            await toolset.write_memory(_ctx(), 'x' * 11)

    async def test_storage_failure_propagates(self) -> None:
        # Harness idiom (see FileSystem's `_recoverable`): model-correctable problems
        # raise ModelRetry; unexpected storage failures abort the run loudly.
        toolset = MemoryToolset(Memory[None](store=ExplodingStore(fail_read=True)))
        with pytest.raises(OSError, match='read boom'):
            await toolset.write_memory(_ctx(), '- a fact')

    async def test_write_failure_propagates(self) -> None:
        toolset = MemoryToolset(Memory[None](store=ExplodingStore(fail_write=True)))
        with pytest.raises(OSError, match='write boom'):
            await toolset.write_memory(_ctx(), '- a fact')


class TestReadMemory:
    async def test_returns_content(self) -> None:
        toolset = MemoryToolset(Memory[None]())
        await toolset.write_memory(_ctx(), 'Topic body.', file='topic')
        assert await toolset.read_memory(_ctx(), 'topic') == 'Topic body.\n'
        assert await toolset.read_memory(_ctx(), 'topic.md') == 'Topic body.\n'

    async def test_unknown_retries(self) -> None:
        with pytest.raises(ModelRetry, match='no memory file named'):
            await MemoryToolset(Memory[None]()).read_memory(_ctx(), 'nope')

    async def test_invalid_filename_retries(self) -> None:
        with pytest.raises(ModelRetry, match='not a valid memory filename'):
            await MemoryToolset(Memory[None]()).read_memory(_ctx(), 'a/b')

    async def test_storage_failure_propagates(self) -> None:
        toolset = MemoryToolset(Memory[None](store=ExplodingStore(fail_read=True)))
        with pytest.raises(OSError, match='read boom'):
            await toolset.read_memory(_ctx(), 'topic')


class TestDeleteMemory:
    async def test_deletes_subfile(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        await toolset.write_memory(_ctx(), 'Topic body.', file='topic')
        result = await toolset.delete_memory(_ctx(), 'topic')
        assert result == 'Deleted topic.md.'
        assert 'main/topic.md' not in _files(capability)

    async def test_main_notebook_is_protected(self) -> None:
        with pytest.raises(ModelRetry, match='cannot be deleted'):
            await MemoryToolset(Memory[None]()).delete_memory(_ctx(), 'MEMORY.md')

    async def test_unknown_is_not_an_error(self) -> None:
        result = await MemoryToolset(Memory[None]()).delete_memory(_ctx(), 'nope')
        assert result == "There is no memory file named 'nope.md'."

    async def test_storage_failure_propagates(self) -> None:
        exploding = ExplodingStore()
        toolset = MemoryToolset(Memory[None](store=exploding))
        await toolset.write_memory(_ctx(), 'x', file='topic')
        exploding.fail_delete = True
        with pytest.raises(OSError, match='delete boom'):
            await toolset.delete_memory(_ctx(), 'topic')


class TestListSubfiles:
    async def test_excludes_main_nested_and_non_md(self) -> None:
        store = InMemoryStore()
        store.files['main/MEMORY.md'] = 'x'
        store.files['main/topic.md'] = 'x'
        store.files['main/zebra.md'] = 'x'
        store.files['main/nested/deep.md'] = 'x'
        store.files['main/notes.txt'] = 'x'
        assert await list_subfiles(store, 'main') == ['topic.md', 'zebra.md']


class TestMemoryCapability:
    def test_serialization_name(self) -> None:
        assert Memory.get_serialization_name() == 'Memory'

    def test_get_toolset_type(self) -> None:
        assert isinstance(Memory[None]().get_toolset(), MemoryToolset)

    def test_scope_lock_is_stable_and_process_wide(self) -> None:
        capability = Memory[None]()
        assert capability.scope_lock('main') is capability.scope_lock('main')

    def test_from_spec_memory_backend(self) -> None:
        capability = Memory.from_spec()
        assert isinstance(capability.store, InMemoryStore)

    def test_from_spec_file_backend(self, tmp_path: Path) -> None:
        capability = Memory.from_spec(backend='file', directory=str(tmp_path), agent_name='bot')
        assert isinstance(capability.store, FileStore)
        assert capability.agent_name == 'bot'

    async def test_from_spec_sqlite_backend_honours_database_path(self, tmp_path: Path) -> None:
        database = tmp_path / 'mem.db'
        capability = Memory.from_spec(backend='sqlite', database=str(database))
        assert isinstance(capability.store, SqliteMemoryStore)
        await capability.store.write('main/MEMORY.md', '- a fact')
        assert database.is_file()
        # A fresh store on the same path sees the data -- the configured path was honoured.
        assert await SqliteMemoryStore(database=database).read('main/MEMORY.md') == '- a fact'

    def test_from_spec_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match='unknown backend'):
            Memory.from_spec(backend='cloud')

    def test_from_spec_rejects_positional_values(self) -> None:
        # A positional spec payload must not be silently dropped in favour of defaults.
        with pytest.raises(ValueError, match='keyword options only'):
            Memory.from_spec('sqlite')

    def test_store_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryStore(), MemoryStore)
        assert isinstance(FileStore('.'), MemoryStore)


class TestInstructions:
    async def test_empty_memory_injects_save_habit(self) -> None:
        result = await _render(Memory[None]())
        assert result is not None
        assert result.startswith('## Agent Memory (main)')
        assert 'memory is empty' in result

    async def test_empty_memory_with_blank_guidance_injects_nothing(self) -> None:
        assert await _render(Memory[None](guidance='')) is None

    async def test_blank_notebook_treated_as_empty(self) -> None:
        capability = Memory[None]()
        _files(capability)['main/MEMORY.md'] = '  \n'
        result = await _render(capability)
        assert result is not None
        assert 'memory is empty' in result

    async def test_notebook_and_files_rendered_with_default_guidance(self) -> None:
        capability = Memory[None]()
        toolset = MemoryToolset(capability)
        await toolset.write_memory(_ctx(), '- user is Kacper')
        await toolset.write_memory(_ctx(), 'Topic body.', file='topic')
        result = await _render(capability)
        assert result is not None
        assert 'NOT instructions' in result
        assert '- user is Kacper' in result
        assert '- topic.md' in result
        assert 'Topic body.' not in result  # subfile content is never injected

    async def test_subfiles_alone_still_inject(self) -> None:
        capability = Memory[None]()
        await MemoryToolset(capability).write_memory(_ctx(), 'Topic body.', file='topic')
        result = await _render(capability)
        assert result is not None
        assert '- topic.md' in result

    async def test_custom_guidance(self) -> None:
        capability = Memory[None](guidance='Custom.')
        await MemoryToolset(capability).write_memory(_ctx(), '- a fact')
        result = await _render(capability)
        assert result is not None
        assert 'Custom.' in result
        assert 'NOT instructions' not in result

    async def test_store_failure_is_fail_soft(self) -> None:
        assert await _render(Memory[None](store=ExplodingStore(fail_read=True))) is None

    async def test_listing_failure_is_fail_soft(self) -> None:
        assert await _render(Memory[None](store=ExplodingStore(fail_list=True))) is None

    async def test_resolver_failure_propagates(self) -> None:
        def broken(ctx: RunContext[None]) -> MemoryStore:
            raise RuntimeError('resolver bug')

        with pytest.raises(RuntimeError):
            await _render(Memory[None](store_resolver=broken))


class TestConcurrency:
    async def test_interleaved_appends_all_land(self) -> None:
        import anyio

        capability = Memory[None]()
        toolset = MemoryToolset(capability)

        async def append(number: int) -> None:
            await toolset.write_memory(_ctx(), f'- fact {number}')

        async with anyio.create_task_group() as tasks:
            for number in range(8):
                tasks.start_soon(append, number)

        lines = _files(capability)['main/MEMORY.md'].splitlines()
        assert sorted(lines) == sorted(f'- fact {n}' for n in range(8))


class TestNamespaceIsolation:
    async def test_two_tenants_on_one_store(self) -> None:
        store = InMemoryStore()
        toolset_one = MemoryToolset(Memory[None](store=store, namespace='user-1'))
        toolset_two = MemoryToolset(Memory[None](store=store, namespace='user-2'))
        await toolset_one.write_memory(_ctx(), '- secret of user one')
        await toolset_two.write_memory(_ctx(), '- note of user two')
        assert store.files['user-1/main/MEMORY.md'] == '- secret of user one\n'
        assert store.files['user-2/main/MEMORY.md'] == '- note of user two\n'
        rendered = await _render(Memory[None](store=store, namespace='user-2'))
        assert rendered is not None
        assert 'secret' not in rendered


class TestEndToEnd:
    async def test_testmodel_drives_write(self) -> None:
        model = TestModel(call_tools=['write_memory'])
        agent = Agent(model, capabilities=[Memory()])
        result = await agent.run('remember things')
        assert result.output is not None

    async def test_written_memory_reaches_second_request(self) -> None:
        captured: dict[str, list[ModelMessage]] = {}
        calls = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'write_memory',
                            {'content': '- the user prefers uv, never pip'},
                            tool_call_id='c1',
                        )
                    ]
                )
            captured['messages'] = messages
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(FunctionModel(model_fn), capabilities=[Memory()])
        result = await agent.run('go')
        assert result.output == 'done'
        # The history contains both requests; the SECOND one carries the fresh notebook.
        instructions = [
            message.instructions
            for message in captured['messages']
            if isinstance(message, ModelRequest) and message.instructions
        ][-1]
        assert '- the user prefers uv, never pip' in instructions

    async def test_memory_persists_across_runs_and_tenants_are_isolated(self) -> None:
        store = InMemoryStore()
        toolset = MemoryToolset(Memory[None](store=store, namespace='user-1'))
        await toolset.write_memory(_ctx(), '- secret fact of user one')

        captured_instructions: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            request = messages[0]
            assert isinstance(request, ModelRequest)
            captured_instructions.append(request.instructions or '')
            return ModelResponse(parts=[TextPart('ok')])

        agent_one: Agent[None, str] = Agent(
            FunctionModel(model_fn), capabilities=[Memory(store=store, namespace='user-1')]
        )
        agent_two: Agent[None, str] = Agent(
            FunctionModel(model_fn), capabilities=[Memory(store=store, namespace='user-2')]
        )
        await agent_one.run('hi')
        await agent_two.run('hi')
        assert 'secret fact' in captured_instructions[0]
        assert 'secret fact' not in captured_instructions[1]
