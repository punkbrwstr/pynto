import re
import sys
from .vocabulary import Word, Column, resolve, set_debug
from .periods import Range, Periodicity, datelike
from .database import get_client

db = get_client()

def __getattr__(name: str):
    return resolve(name)
