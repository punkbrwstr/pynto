import re
import sys
from .vocabulary import *
from .periodicities import *
from .ranges import Range
from .vocabulary import _define as define
from .database import get_client
from . import main
from . import vocabulary

db = get_client()

__ = main.Word('')
__.open_quote = True

       


def __getattr__(name):

    if re.match('f\d[_\d]*',name) is not None:
        return vocabulary.c(float(name[1:].replace('_','.')))
    elif re.match('f_\d[_\d]*',name) is not None:
        return vocabulary.c(-float(name[2:].replace('_','.')))
    elif re.match('i\d+',name) is not None:
        return vocabulary.c(int(name[1:].replace('_','.')))
    elif re.match('i_\d+',name) is not None:
        return vocabulary.c(-int(name[2:].replace('_','.')))
    else:
        return getattr(vocabulary, name)
