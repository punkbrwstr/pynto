from .vocabulary import Word, Column, resolve, toggle_debug, vocab
from .periods import Range, Periodicity, datelike
from .database import get_client

db = get_client()


class _Definer:
    def __setitem__(self, name: str, word: Word) -> None:
        vocab[name] = ('Ad-hoc', name, lambda name: Word(name) + word)


define = _Definer()


def now():
    return Periodicity.B.current()[-1]


def __getattr__(name: str):
    if not name.startswith('__'):
        return resolve(name)
    else:
        if name not in globals():
            raise AttributeError
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
