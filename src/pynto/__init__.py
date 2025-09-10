import re
import sys
from .vocabulary import Word, Column, resolve, set_debug, vocab
from .periods import Range, Periodicity, datelike
from .database import get_client

db = get_client()

class _Definer:
    def __setitem__(self, name: str, word: Word) -> None:
        vocab[name] = ('Ad-hoc', name, lambda name: Word(name) + word)

define = _Definer()

def __getattr__(name: str):
    return resolve(name)
