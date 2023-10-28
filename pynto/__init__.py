import re
import sys
#from .vocabulary import *
from .main import Word, _resolve
from .periodicities import *
from .ranges import Range
from .vocabulary import _define as define
from .database import get_client, Saved

db = get_client()
define('db', lambda: Saved())

__ = Word('')
__.open_quote = True


def __getattr__(name):
    return _resolve(name)
