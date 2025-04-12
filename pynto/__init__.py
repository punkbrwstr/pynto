import re
import sys
from .main2 import Word, Column, resolve
from .periods import Range, Periodicity
#from .vocabulary import _define as define
from .database import get_client

db = get_client()
#define('db', lambda: Saved())

#__ = Word('')
#__.open_quote = True


def __getattr__(name):
    return resolve(name)
