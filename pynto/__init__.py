import re
import sys
from .main import Word, Column, resolve
from .periods import Range, Periodicity
from .database import get_client

db = get_client()

#__ = Word('')
#__.open_quote = True


def __getattr__(name):
    return resolve(name)
