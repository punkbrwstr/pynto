from typing import Any
from .vocabulary import vocab
from .base import Word, Column, toggle_debug
from .periods import Range, Periodicity, datelike
from .database import get_client

db = get_client()


class _Definer:
    def __setitem__(self, name: str, word: Word) -> None:
        vocab[name] = ('Ad-hoc', name, lambda n, v, w=word: Word(n, v) + w)


define = _Definer()


def now():
    return Periodicity.B.current()[-1]


def __getattr__(name: str) -> Any:
    if not name.startswith('__'):
        word = vocab.resolve(name)
        if word:
            return word
    if name not in globals():
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return globals()[name]


__all__ = [
    # vocabulary exports
    'Word',
    'Column',
    'resolve',
    'toggle_debug',
    'vocab',
    # periods exports
    'Range',
    'Periodicity',
    'datelike',
    # database exports / instances
    'get_client',
    'db',
    # convenience helpers
    'define',
    'now',
]
