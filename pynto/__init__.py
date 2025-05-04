import re
import sys
from .vocabulary import Word, Column, resolve
from .periods import Range, Periodicity
from .database import get_client

db = get_client()

def __getattr__(name: str):
    return resolve(name)
