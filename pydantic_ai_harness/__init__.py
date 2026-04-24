"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .guardrails import (
        GuardrailError,
        GuardrailFunc,
        InputBlocked,
        InputGuard,
        OutputBlocked,
        OutputGuard,
    )

__all__ = [
    'CodeMode',
    'GuardrailError',
    'GuardrailFunc',
    'InputBlocked',
    'InputGuard',
    'OutputBlocked',
    'OutputGuard',
]


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in {'GuardrailError', 'GuardrailFunc', 'InputBlocked', 'InputGuard', 'OutputBlocked', 'OutputGuard'}:
        from . import guardrails

        return getattr(guardrails, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
