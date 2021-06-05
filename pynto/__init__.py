from .periodicities import *
from .ranges import Range
from .vocabulary import *
from .vocabulary import _define
from .database import get_client

db = get_client()

def define(name: str, word) -> None:
    globals()[name] = word
    _define(name, word)
