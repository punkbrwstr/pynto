from __future__ import annotations

import copy
import datetime
import logging
import numbers
import re
import sys
import traceback
from collections import deque
from dataclasses import dataclass, field, KW_ONLY, replace
from functools import partial, reduce
from operator import add
from typing import Any, Callable
from enum import Enum

import bottleneck as bn
import numpy as np
import numpy.ma as ma
import pandas as pd

from . import database as db
from .periods import Range, Periodicity, datelike

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
_DEBUG = False

def toggle_debug():
    global _DEBUG
    _DEBUG = not _DEBUG
    if _DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

class ColumnIDGenerator:
    def __init__(self):
        self.current = 0

    def get_next(self):
        self.current += 1
        return self.current

    def reset(self):
        self.current = 0

_IDs = ColumnIDGenerator()

class ResampleMethod(Enum):
    LAST = 1
    SUM = 2
    AVG = 3
    LAST_NOFILL = 4

@dataclass(eq=False)
class Column:
    header: str = ''
    range_: Range | None = None
    output_columns: int = 1
    input_stack: list[Column] = field(default_factory=list)
    cache: dict[Range,None | np.ndarray] = field(default_factory=dict)
    id_: int = field(default_factory=_IDs.get_next)
    resample_method: ResampleMethod = ResampleMethod.LAST

    def calculate(self) -> None:
        pass

    @property
    def values(self) -> np.ndarray:
        return self.cache[self.range_]

    @values.setter
    def values(self, values: np.ndarray) -> np.ndarray:
        self.cache[self.range_] = values

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        if range_ not in self.cache:
            if len(self.cache) != 0:
                inputs = self.input_stack[:]
                self.input_stack = [copy.copy(i) for i in inputs]
            self.set_input_range()
            self.cache[range_] = None
        else:
            self.input_stack = []

    def set_input_range(self) -> None:
        for col in self.input_stack:
            col.set_range(self.range_)

    @property
    def calculated(self) -> bool:
        return self.cache.get(self.range_) is not None

    def get_bounds(self) -> tuple[datetime.date,datetime.date,Periodicity] | None:
        return None

    def drop(self):
        pass

    def __copy__(self):
        logger.debug('copying '+ debug_col_repr(self, 4))
        return replace(self, id_=_IDs.get_next())

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other: Column):
        return self.id_ == other.id_

    def __del__(self):
        pass #logger.debug('destroying '+ debug_col_repr(self, 4))

@dataclass(kw_only=True,eq=False)
class SiblingColumn(Column):
    siblings: set[SiblingColumn]
    ordinal: int | None = None
    input_column: Column | None = None
    allow_sibling_drops: bool = True
    copies: list[SiblingColumn] | None = None

    def set_range(self, range_: Range) -> None:
        if self.ordinal is None:
            output_columns = len(self.siblings)
            for i, sib in enumerate(self.siblings):
                sib.set_ordinal(i)
                sib.output_columns = output_columns
                if sib.input_column:
                    self.input_stack.append(sib.input_column)
                    sib.input_column = None
            self.siblings.clear()
        super().set_range(range_)

    def __copy__(self):
        copy_ = super().__copy__()
        copy_.copies = None
        if self.copies is None:
            self.copies = [copy_]
        else:
            self.copies.append(copy_)
        return copy_

    def set_ordinal(self, ordinal: int) -> None:
        self.ordinal = ordinal
        if self.copies:
            for c in self.copies:
                c.set_ordinal(ordinal)
            self.copies.clear()

    @property
    def values(self) -> np.ndarray:
        try:
            return self.cache[self.range_][:, [self.ordinal]]
        except:
            print(f'look up here {self.ordinal}')
            print(f'     {self.header}')
            print(f'     {self}')
            raise

    def drop(self):
        if self.allow_sibling_drops:
            pass
            #self.siblings.remove(self)
        super().drop()

class Word:
    def __init__(self, name: str, slice_: slice = slice(None),
                    copy_selected: bool = False, discard_excluded: bool = False,
                 inverse_selection: bool = False, raise_on_empty: bool = False):
        self.name = name
        self.slice_ = slice_
        self.copy_selected = copy_selected
        self.discard_excluded = discard_excluded
        self.inverse_selection = inverse_selection
        self.raise_on_empty = raise_on_empty
        self.filters: list[str] | None = None
        self.next_: Word | None = None
        self.prev: Word | None = None
        self.open_quotes: int = 0
        self.called: bool = False

    def operate(self, stack: list[Column]) -> None:
        if self.raise_on_empty and len(stack) == 0:
            raise IndexError(f'Empty stack for {self.name}')
        pass

    def __call__(self, kwargs = None) -> Word:
        self.called = True
        if kwargs:
            for key, value in kwargs.items():
                if key not in ['__class__','self']:
                    setattr(self, key, value)
        return self

    def __add__(self, addend: Word, copy_addend=True) -> Word:
        this = self.copy_expression()
        if copy_addend:
            addend_tail = addend.copy_expression()
        else:
            addend_tail = addend
        addend_head = addend_tail._head()
        if this is not None:
            this.next_ = addend_head
            addend_head.prev = this
            if this.open_quotes != 0:
                current = addend_head
                while current.next_ is not None:
                    current.open_quotes += this.open_quotes
                    current = current.next_
                current.open_quotes += this.open_quotes
        return addend_tail

    def concat(self, other: Word) -> Word:
        return self.__add__(other)

    def __getattr__(self, name: str):
        word = resolve(name, False)
        if word:
            return self.__add__(word, copy_addend=False)
        else:
            return self.__getattribute__(name)

    def __getitem__(self, key: int | tuple[int, bool] \
                                | slice | tuple[slice,bool] \
                                | str | tuple[str, bool] \
                                | list[str] | tuple[list[str], bool] ) -> Word:
        if isinstance(key, tuple):
            assert isinstance(key[1], bool), 'Second argument must be boolean'
            self.copy_selected = key[1]
            if len(key) == 3:
                assert isinstance(key[2], bool), 'Third argument must be boolean'
                self.discard_excluded = key[2]
            key = key[0]
        if isinstance(key, int):
            if key == -1:
                self.slice_ = slice(key,None)
            else:
                self.slice_ = slice(key, key + 1)
        elif isinstance(key, slice):
            self.slice_ = key
        elif isinstance(key, str):
            self.filters = [key]
        elif isinstance(key, list):
            self.filters = key
        else:
            raise IndexError('Invalid column indexer')
        return self

    @property
    def p(self):
        assert self.open_quotes > 0, 'No quote to close.'
        current = self
        open_quotes = self.open_quotes
        while current.prev is not None and current.prev.open_quotes == open_quotes:
            current.open_quotes = 0
            current = current.prev
        current.open_quotes -= 1
        assert isinstance(current, Quotation), 'something wrong'
        current.next_.prev = None
        current = current(current.next_._tail())
        current.next_ = None
        return current

    def __dir__(self):
        return sorted(set(vocab.keys()))


    def build_stack(self, stack: list[Column] | None = None) -> list[Column]:
        assert self.open_quotes == 0, 'Unclosed quotation.  Cannot evaluate'
        if stack is None:
            stack = []
            prefix = 0
        else:
            prefix = 8      
        current = self._head()
        logger.debug(f'{" " * prefix}Build stack')
        while current is not None:
            if not current.called: # need to set defaults (if any)
                current()
            logger.debug(f'{" " * prefix}   current word {current} slice {current.slice_}')
            selected = list(range(len(stack))[current.slice_])
            if current.filters:
                filtered = []
                for filter_ in current.filters:
                    pattern = re.compile(filter_)
                    for i in range(len(stack)):
                        if bool(pattern.match(stack[i].header)):
                            filtered.append(i)
                selected = filtered
            if current.inverse_selection:
                selected = list(set(range(len(stack))) - set(selected))
            logger.debug(f'{" " * prefix}      selected {selected}')
            current_stack = []
            to_delete = set()
            for i in selected:
                if current.copy_selected:
                    c = copy.copy(stack[i])
                    current_stack.append(c)
                else:
                    to_delete.add(i)
                    current_stack.append(stack[i])

            if current.discard_excluded:
                excluded = set(range(len(stack))) - set(selected)
                for i in excluded:
                    logger.debug(f'{" " * prefix}      dropping {debug_col_repr(stack[i],4)}')
                    stack[i].drop() # no longer needed as siblings   
                to_delete.update(excluded)
            for i in sorted(to_delete, reverse=True):
                del(stack[i])
            current.operate(current_stack)
            stack.extend(current_stack)
            current = current.next_
            #debug_stack(stack, offset=prefix + 4)
        return stack

    @property
    def stack(self) -> list[Column]:
        return self.build_stack()

    @property
    def columns(self) -> list[str]:
        return [col.header for col in self.stack]

    @property
    def first(self) -> Evaluator:
        return Evaluator(self)[0]

    @property
    def last(self) -> Evaluator:
        return Evaluator(self)[-1]

    @property
    def rows(self) -> Evaluator:
        return Evaluator(self)

    @property
    def values(self) -> Evaluator:
        return Evaluator(self, True)

    def __copy__(self) -> Word:
        cls = self.__class__
        copied = cls.__new__(cls)
        copied.__dict__.update(self.__dict__)
        return copied

    def copy_expression(self) -> Word:
        first = copy.copy(self)
        current = first
        while current.prev is not None:
            prev = copy.copy(current.prev)
            current.prev = prev
            prev.next_ = current
            current = prev
        return first

    def _head(self) -> Word:
        current = self
        while current.prev is not None:
            current = current.prev
        return current

    def _tail(self) -> Word:
        current = self
        while current.next_ is not None:
            current = current.next_
        return current

    @property
    def local(self) -> Word:
        return Quotation('q')(self).call[-1:0]

    def __str__(self) -> str:
        ignore = 'prev next_ closed called generator slice_ filters copy_selected' \
                    + ' discard_excluded inverse_selection operation open_quotes' \
                    + ' name allow_sibling_drops ascending raise_on_empty'
        str_args = []
        for k,v in self.__dict__.items():
            if k != self and k not in ignore.split():
                if isinstance(v, str):
                    str_args.append(f"{k}='{v[:2000]}'")
                    if len(v) > 2000:
                        str_args[-1] += '...'
                else:
                    str_args.append(f"{k}={str(v)}")
        s = self.name
        if str_args:
            s += f'({", ".join(str_args)})'
        if self.filters:
            s += f'[{self.filters}]' 
        else:
            s += f'[{self.slice_.start or ""}:{self.slice_.stop or ""}:{self.slice_.step or ""}]'
        return s

    def __repr__(self) -> str:
        s = ''
        current = self
        while True:
            s = current.__str__() + s
            if current.prev is None:
                break
            else:
                s = '\n    .' + s
                current = current.prev
        s = 'pt.' + s
        return s

def debug_col_repr(c: Column, n: int) -> str:
    if not _DEBUG:
        return ''
    return  f'{' '*n}#{c.id_}-{c.__class__.__name__}' \
        + f' "{c.header}"' \
        + f' {c.range_}' \
        + f' (cache #{str(id(c.cache))[-4:]})' \
        + (f' ordinal {c.ordinal})' if hasattr(c,'ordinal') else '') \
        + (f' sibs [{",".join([str(c2.id_) for c2 in c.siblings])}]' if hasattr(c,'siblings') else '') \
        + (f' cops [{",".join([str(c2.id_) for c2 in c.copies])}]' 
                if hasattr(c,'copies') and c.copies else '')

def debug_stack(stack,title=None, offset=4):   
    if not _DEBUG:
        return
    if title:
        logger.debug((' ' * offset) + title)
        offset += 4
    for c in stack:
        logger.debug(debug_col_repr(c,offset))
        if c.input_stack:
            debug_stack(c.input_stack,'inputs',offset+4)
        if hasattr(c,'siblings'):
            debug_stack(c.siblings,'sibs',offset+4)

def _resample(to_range: Range, to_values: np.ndarray,
                from_range: Range, from_values: np.ndarray,
                method: ResampleMethod) -> None:
    idx, idx_input = from_range.resample_indicies(to_range, False)
    match method:
        case ResampleMethod.LAST:
            x = from_values.ravel()
            where = np.where(~np.isnan(x),np.arange(len(x)),0)
            np.maximum.accumulate(where,out=where)
            from_values[:] = x[where][:,None]
            to_values[idx] = from_values[idx_input]
        case ResampleMethod.LAST_NOFILL:
            to_values[idx] = from_values[idx_input]
        case ResampleMethod.SUM:
            sums = np.nancumsum(from_values)
            to_values[idx[0]] = from_values[idx_input[0]] 
            to_values[idx[1:]] = (sums[idx_input[1:]] - sums[idx_input[:-1]])[:,None]
        case ResampleMethod.AVG:
            sums = np.nancumsum(from_values)
            counts = np.nancumsum(from_values / from_values)
            to_values[idx[0]] = from_values[idx_input[0]] 
            to_values[idx[1:]] = (sums[idx_input[1:]] - sums[idx_input[:-1]])[:,None]
            to_values[idx[1:]] /= (counts[idx_input[1:]] - counts[idx_input[:-1]])[:,None]

class Evaluator:
    def __init__(self, word: Word, values_only: bool = False):
        self.word = word
        self.values_only = values_only

    def __getitem__(self, key: slice | int | str | Range) -> pd.DataFrame:
        _IDs.reset()
        stack = self.word.build_stack()
        min_, max_, per = None, None, None
        working = stack[:]
        flat = []
        while working:
            col = working.pop()
            bounds = col.get_bounds()
            if bounds:
                min_ = bounds[0] if min_ is None else min(min_, bounds[0])
                max_ = bounds[1] if max_ is None else max(max_, bounds[1])
                per = bounds[2]
            flat.append(col)
            working.extend(col.input_stack)
        flat.reverse()
        if isinstance(key, Range):
            range_ = key
        elif isinstance(key, datelike) or isinstance(key, int):
            p = per if per else Periodicity.B
            range_ = p[key].expand(0)
        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            p = Periodicity[step] if step \
                else per if per else Periodicity.B
            if isinstance(start, int) and max_ is not None:
                if start < 0:
                    start = p[max_] + start
                else:
                    start = p[min_] + start
            if isinstance(stop, int) and max_ is not None:
                if stop < 0:
                    stop = p[max_] + stop
                else:
                    stop = p[min_] + stop
            range_ = p[start or min_:stop or max_]
        else: 
            raise TypeError(f'Unsupported indexer')
        for col in stack:
            logger.debug('pre set range '+ debug_col_repr(col, 0))
            try:
                col.set_range(range_)
            except Exception as e:
                logger.error(f'Error setting range for {col}')
                raise e
            logger.debug('post set range'+ debug_col_repr(col, 0))
        logger.debug('Stack')
        debug_stack(stack)
        working = stack[:]
        flat = deque()
        while working:
            col = working.pop()
            flat.append(col)
            working.extend(col.input_stack)
        flat.reverse()
        logger.debug('Flat')
        debug_stack(flat)
        saveds = [col for col in flat if isinstance(col, SavedColumn)]
        if saveds:
            p = db.get_client().connection.pipeline()
            offsets, needed_saveds, resample = [], [], []
            for col in saveds:
                if col.range_ and not col.calculated:
                    col.cache[col.range_] = np.full((len(col.range_),1), np.nan, order='F')
                    if col.range_.periodicity != col.md.periodicity:
                        r = col.range_.change_periodicity(col.md.periodicity)
                        resample.append(r)
                    else:
                        r = col.range_
                        resample.append(None)
                    offsets.append(db.get_client()._req(col.md, r.start, r.stop, p))
                    needed_saveds.append(col)
            for col, offset, bytes_,r in zip(needed_saveds, offsets, p.execute(), resample):
                if len(bytes_) > 0:
                    data = np.frombuffer(bytes_, col.md.type_.dtype)
                    if r is not None:
                        from_values = np.full((len(r),1), np.nan, order='F')
                        from_values[offset:offset + len(data),0] = data
                        _resample(col.range_, col.values,
                                    r, from_values, col.resample_method)
                    else:
                        col.values[offset:offset + len(data),0] = data
        logger.debug('Processing flat')
        while flat:
            col = flat.popleft()
            if col.calculated or not col.range_:
                logger.debug(f'    skip' + debug_col_repr(col,1)) 
                continue
            logger.debug(f'    calc' + debug_col_repr(col,1)) 
            col.cache[col.range_] = np.empty((len(col.range_), col.output_columns), order='F')
            col.calculate()
            col.input_stack = []
        if not stack:
            return None
        values = np.concatenate([col.values for col in stack]) \
                    .reshape((len(range_),len(stack)),order='F')
        if self.values_only:
            return values
        return pd.DataFrame(values, columns=[col.header for col in stack], index=range_.to_index())

# Nullary/generator words
class NullaryWord(Word):
    def __init__(self, name: str, generator: Callable[[Range, np.ndarray],None]):
        self.generator = generator
        super().__init__(name, slice(-1,0))

    def operate(self, stack: list[Column]) -> None:
        stack.append(NullaryColumn(header=self.name, generator=self.generator))

@dataclass(kw_only=True, eq=False)
class NullaryColumn(Column):
    generator: Callable[[Range, np.ndarray],None]

    def calculate(self) -> None:
        self.generator(self.range_, self.values)

class RandomNormal(NullaryWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = np.random.randn(len(range_))[:,None]

    def __init__(self, name: str):
        super().__init__(name,  self.generate)

class Timestamp(NullaryWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = range_.to_index().view('int').astype(np.float64)[:,None]

    def __init__(self, name: str):
        super().__init__(name, self.generate)

class PeriodOrdinal(NullaryWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = np.arange(range_.start,range_.stop).astype(np.float64)[:,None]

    def __init__(self, name: str):
        super().__init__(name, self.generate)

class Daycount(NullaryWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = np.array(range_.day_counts(), dtype=np.float64)[:,None]

    def __init__(self, name: str):
        super().__init__(name, self.generate)

class Constant(NullaryWord):
    @staticmethod
    def generate(constant: float, range_: Range, values: np.ndarray):
        values[:] = constant

    def __init__(self, name: str):
        super().__init__(name, self.generate)

    def __call__(self, *values: list[float]) -> Word:
        self.constants = values
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for constant in self.constants:
            stack.append(NullaryColumn(
                            header=self.name,
                            generator=partial(self.generate, constant)))

class ConstantRange(Constant):
    def __call__(self, n: int) -> Word:
        return super().__call__(*range(n))

@dataclass(kw_only=True, eq=False)
class PandasColumn(SiblingColumn):
    pandas: pd.DataFrame
    round_: bool
    pandas_range: Range| None = None

    def __post_init__(self):
        self.pandas_range = Range.from_index(self.pandas.index)

    def get_bounds(self) -> tuple[datetime.date,datetime.date,Periodicity] | None:
        return (self.pandas_range[0][-1],
                self.pandas_range.expand()[-1][-1],
                self.pandas_range.periodicity)

    def calculate(self) -> None:
        self.cache[self.range_][:] = np.nan
        idx, idx_input = self.pandas_range \
                            .resample_indicies(self.range_, self.round_)
        self.cache[self.range_][idx,:] = self.pandas.values[idx_input,:]

class FromPandas(Word):
    def __call__(self, pandas: pd.DataFrame | pd.Series, round_: bool = False) -> Word:
        if isinstance(pandas, pd.Series):
            self.pandas = self.pandas.toframe()
        else:
            self.pandas = pandas
        if self.pandas.index.freq is None:
            self.pandas.index.freq = pd.infer_freq(self.pandas.index)
        self.round_ = round_
        return super().__call__()

    def operate(self, stack):
        if self.pandas.index.freq is None:
            self.pandas.index.freq = pd.infer_freq(self.pandas.index)
        siblings = set()
        input_stack = []
        cache = {}
        for i, (header, col) in enumerate(self.pandas.items()):
            sib = PandasColumn(
                        header,
                        round_=self.round_,
                        pandas=self.pandas,
                        input_stack=input_stack,
                        cache=cache,
                        siblings=siblings
                        )
            siblings.add(sib)
            stack.append(sib)

@dataclass(kw_only=True, eq=False)
class SavedColumn(Column):
    md: db.Metadata

    def get_bounds(self) -> tuple[datetime.date,datetime.date,Periodicity] | None:
        return (self.md[0][-1], self.md.expand()[-1][-1], self.md.periodicity)

class Saved(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,0))

    def __call__(self, key: str) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        for md in db.get_client().get_metadata(self.key):
            stack.append(SavedColumn(md.col_header + md.row_header, md=md))


# Quotation / combinator words
class Quotation(Word):
    def __init__(self, name: str, slice_=slice(-1,0)):
        super().__init__(name, slice_)

    def __call__(self, quoted: Word | None = None) -> Word:
        if quoted is None:
            return self
        else:
            this = self.copy_expression()
            this.quoted = quoted
            return this

    def __add__(self, other: Word, copy_addend: bool = True) -> Word:
        if not hasattr(self, 'quoted') or self.quoted is None:
            self.open_quotes += 1
        return super().__add__(other, copy_addend)

class Combinator(Word):
    def __init__(self, name: str, slice_: slice = slice(None), num_quotations: int = 1):
        self.num_quotations = num_quotations
        super().__init__(name, slice_)

    def get_quotations(self) -> list[Word]:
        quotations = []
        current = self.prev
        needed = self.num_quotations 
        while needed != 0 and current and isinstance(current, Quotation):
            quotations.append(current.quoted)
            needed -=1
            current = current.prev
        assert needed <= 0, 'Missing quotation for combinator'
        return quotations

class Call(Combinator):
    def operate(self, stack: list[Column]) -> None:
        self.get_quotations()[0].build_stack(stack)

class IfExists(Combinator):
    def __call__(self, count: int = 1) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if len(stack) >= self.count:
            self.get_quotations()[0].build_stack(stack)

class IfExistsElse(Combinator):
    def __init__(self, name: str):
        super().__init__(name, num_quotations = 2)

    def __call__(self, count: int = 1) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if len(stack) >= self.count:
            self.get_quotations()[0].build_stack(stack)
        else:
            self.get_quotations()[1].build_stack(stack)

class IfHeaders(Combinator):
    def __call__(self, predicate: Callable[[list[str]],bool]) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if self.predicate([col.header for col in stack]):
            self.get_quotations()[0].build_stack(stack)

class IfHeadersElse(Combinator):
    def __init__(self, name: str):
        super().__init__(name, num_quotations = 2)

    def __call__(self, predicate: Callable[[list[str]],bool]) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if self.predicate([col.header for col in stack]):
            self.get_quotations()[0].build_stack(stack)
        else:
            self.get_quotations()[1].build_stack(stack)

class Map(Combinator):
    def __call__(self, every: int = 1):
        self.every = every
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        assert len(stack) % self.every == 0, \
           f'Stack length {len(stack)} not evenly divisible by every {self.every}'
        copied = stack[:]
        stack.clear()
        quoted = self.get_quotations()[0]
        for t in zip(*[iter(copied)]*self.every):
            this_stack = list(t)
            quoted.build_stack(this_stack)
            stack += this_stack

class Repeat(Combinator):
    def __call__(self, times: int = 2) -> Word:
        self.times = times
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for _ in range(self.times):
            self.get_quotations()[0].build_stack(stack)

class HMap(Combinator):
    def operate(self, stack: list[Column]) -> None:
        quoted = self.get_quotations()[0]
        new_stack = []
        for header in set([c.header for c in stack]):
            to_del, filtered_stack = [], []
            for i,col in enumerate(stack):
                if header == col.header:
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            quoted.build_stack(filtered_stack)
            new_stack += filtered_stack
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        del(stack[:])
        stack += new_stack

class Cleave(Combinator):
    def __init__(self, name):
        super().__init__(name, num_quotations = -1)

    def __call__(self, num_quotations: int = -1):
        self.num_quotations = num_quotations
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        output = []
        for quote in reversed(self.get_quotations()):
            this_stack = [copy.copy(col) for col in stack]
            quote.build_stack(this_stack)
            output += this_stack
        stack.clear()
        stack.extend(output)

class BoundWord(Word):
    def __init__(self, bound: list[Column]):
        self.bound = bound
        self.operate_count = 0
        super().__init__('bound')

    def operate(self, stack: list[Column]) -> None:
        if self.operate_count == 0:
            stack.extend(self.bound)
        else:
            bound_copy = [copy.copy(c) for c in self.bound]
            stack.extend(bound_copy)
        self.operate_count += 1

class Partial(Quotation, Combinator):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))
        
    def __add__(self, other: Word, copy_addend: bool = True) -> Word:
        return Word.__add__(self, other, copy_addend)

    def operate(self, stack: list[Column]) -> None:
        self.quoted = BoundWord(stack[:]) + self.get_quotations()[0]
        stack.clear()

class Compose(Quotation, Combinator):
    def __init__(self, name: str):
        Combinator.__init__(self, name, num_quotations=2)

    def __call__(self, num_quotations: int = 2):
        self.num_quotations = num_quotations
        return super().__call__()

    def __add__(self, other: Word, copy_addend: bool = True) -> Word:
        return Word.__add__(self, other, copy_addend)

    def operate(self, stack: list[Column]) -> None:
        for i, quoted in enumerate(reversed(self.get_quotations())):
            if i == 0:
                self.quoted = quoted
            else:
                self.quoted += quoted

class Resample(Word):
    def __init__(self, name: str, method: ResampleMethod):
        self.method = method
        super().__init__(name)

    def operate(self, stack: list[Column]) -> None:
        for col in stack:
            col.resample_method = self.method

@dataclass(kw_only=True, eq=False)
class PeriodicityColumn(Column):
    periodicity: Periodicity

    def set_input_range(self) -> None:
        self.input_stack[0].set_range(self.range_.change_periodicity(self.periodicity))

    def calculate(self) -> None:
        self.values[:] = np.nan
        _resample(self.range_,self.values, 
                    self.input_stack[0].range_, self.input_stack[0].values,
                    self.input_stack[0].resample_method)

class SetPeriodicity(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))

    def __call__(self, periodicity: str | Periodicity) -> Word:
        if isinstance(periodicity, str):
            self.periodicity = Periodicity[periodicity]
        else:
            self.periodicity = periodicity
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for col in inputs:
            stack.append(PeriodicityColumn(col.header,
                               periodicity=self.periodicity, input_stack=[col]))

@dataclass(kw_only=True, eq=False)
class StartColumn(Column):
    start: datelike

    def set_input_range(self) -> None:
        input_start = self.range_.periodicity[self.start].ordinal
        input_range = Range(input_start, self.range_.stop, self.range_.periodicity)
        self.input_stack[0].set_range(input_range)

    def calculate(self) -> None:
        offset = self.range_.start - self.input_stack[0].range_.start
        if offset >= 0:
            self.values[:] = self.input_stack[0].values[offset:]
        else:
            self.values[offset:] = self.input_stack[0].values
            

class SetStart(Word):
    def __call__(self, start: datelike, \
                    round_: bool = False) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for col in inputs:
            stack.append(StartColumn(col.header, start=self.start, input_stack=[col]))

@dataclass(kw_only=True, eq=False)
class FillColumn(Column):
    value: float

    def calculate(self) -> None:
        self.values[:] = self.input_stack[0].values
        self.values[np.isnan(self.values)] = self.value

class Fill(Word):
    def __call__(self, value: float) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for col in inputs:
            stack.append(FillColumn(col.header, value=self.value, input_stack=[col]))

@dataclass(kw_only=True, eq=False)
class FFillColumn(Column):
    lookback: int
    leave_end: bool

    def set_input_range(self) -> None:
        input_range = self.range_.expand(int(-self.lookback))
        for col in self.input_stack:
            col.set_range(input_range)

    def calculate(self) -> None:
        x = self.input_stack[0].values.ravel()
        idx = np.where(~np.isnan(x),np.arange(len(x)),0)
        np.maximum.accumulate(idx,out=idx)
        self.values[:] = x[idx][self.lookback:][:,None]
        if self.leave_end:
            if len(x) > idx.max() + 1:
                at_end = len(x) - idx.max() - 1
                self.values[-at_end:] = np.nan

class FFill(Word):
    def __call__(self, lookback: int = 10, leave_end: bool = True) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for col in inputs:
            stack.append(FFillColumn(col.header, lookback=self.lookback,
                                     leave_end=self.leave_end, input_stack=[col]))

@dataclass(kw_only=True, eq=False)
class JoinColumn(Column):
    date: datelike

    def set_input_range(self) -> None:
        r = self.range_
        cutover_index = r.periodicity[self.date].ordinal
        if r.stop < cutover_index:
            self.input_stack[0].set_range(r)
        elif r.start >= cutover_index:
            self.input_stack[-1].set_range(r)
        else:
            self.input_stack[0].set_range(Range(r.start, cutover_index, r.periodicity))
            self.input_stack[1].set_range(Range(cutover_index, r.stop, r.periodicity))

    def calculate(self) -> None:
        i = 0
        for input_ in self.input_stack:
            if input_.range_ is not None:
                v = input_.values
                self.values[i:i+len(v)] = v
                i += len(v)

class Join(Word):
    def __init__(self, name: str):
        super().__init__(name, slice_=slice(-2,None))

    def __call__(self, date: datelike) -> Word:
        self.date = date
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        stack.append(JoinColumn(stack[-2].header, date=self.date,
                                input_stack=[stack.pop(-2),stack.pop()]))

@dataclass(kw_only=True, eq=False)
class ReductionColumn(Column):
    operation: Callable[[np.ndarray], np.ndarray]
    ignore_nans: bool

    def calculate(self) -> None:
        inputs = np.concatenate([col.values for col in self.input_stack]) \
                    .reshape((len(self.range_),len(self.input_stack)),order='F')
        values = self.operation(inputs)[:, None]
        if self.ignore_nans:
            values[np.all(np.isnan(inputs),axis=1)] = np.nan
        else:
            values[np.any(np.isnan(inputs),axis=1)] = np.nan
        self.values[:] = values

class Reduction(Word):
    def __init__(self, name: str, operation: Callable[[np.ndarray, np.ndarray], None]):
        self.operation = operation
        super().__init__(name, slice(-2, None))

    def __call__(self, ignore_nans: bool = False) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        stack.append(ReductionColumn(inputs[0].header,
                    operation=self.operation,
                    ignore_nans=self.ignore_nans,
                    input_stack=inputs))

@dataclass(kw_only=True, eq=False)
class RollingColumn(SiblingColumn):
    operation: Callable[[np.ndarray, int], np.ndarray]
    window: int
    reduction: bool = False
    lookback: int | None = None

    def set_input_range(self) -> None:
        self.lookback = max(5,int(self.window * 1.25))
        input_range = self.range_.expand(-self.lookback)
        for col in self.input_stack:
            col.set_range(input_range)

    def calculate(self) -> None:
        data = np.concatenate([col.values for col in self.input_stack]) \
                .reshape((len(self.input_stack[0].range_),len(self.input_stack)),order='F')
        mask = ~np.any(np.isnan(data), axis=1)
        idx = np.where(mask)[0]
        out = self.cache[self.range_]
        if np.any(idx >= self.lookback):
            start = np.where(idx >= self.lookback)[0][0]
            out[(idx - self.lookback)[start:]] = self.operation(data[mask], self.window)[start:]
            if not np.all(mask):
                idx = np.where(~mask)[0]
                if np.any(idx >= self.lookback):
                    start = np.where(idx >= self.lookback)[0][0]
                    out[(idx - self.lookback)[start:]] = np.nan
        else:
            out[:] = np.nan
                


class Rolling(Word):
    def __init__(self, name: str, operation: np.ufunc, slice_=slice(-1,None)):
        self.operation = operation
        super().__init__(name, slice_=slice_)

    def __call__(self, window: int = 2) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        cache = {}
        inputs = [*stack]
        stack.clear()
        siblings = set()
        input_stack = []
        cache = {}
        for i, col in enumerate(inputs):
            sib = RollingColumn(
                        col.header,
                        operation=self.operation,
                        window=self.window,                           
                        input_stack=input_stack,
                        input_column=col,
                        cache=cache,
                        siblings=siblings
                        )
            siblings.add(sib)
            stack.append(sib)

class RollingReduction(Rolling):    
    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        stack.append(RollingColumn(inputs[0].header,
                    window=self.window,
                    operation=self.operation,
                    reduction=True,
                    input_stack=inputs))

@dataclass(kw_only=True, eq=False)
class AccumulatorColumn(Column):
    func: Callable[[np.ndarray, int], np.ndarray]
    ascending: bool = True

    def calculate(self) -> None:
        data = self.input_stack[0].values

        mask = ~np.isnan(data)
        mask = ~np.any(np.isnan(data), axis=1)
        idx = np.where(mask)[0]
        if not np.all(~mask):
            if not self.ascending:
                self.values[idx] = self.func(data[mask][::-1])[::-1]
            else:
                self.values[idx] = self.func(data[mask])
        if not np.all(mask):
            self.values[~mask] = np.nan

class Accumulator(Word):
    def __init__(self, name: str, func: Callable[[np.ndarray,np.ndarray],None],
                 ascending: bool = True) -> Word:
        self.func = func
        self.ascending = ascending
        super().__init__(name, slice_=slice(-1,None))

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        for col in inputs:
            stack.append(AccumulatorColumn(col.header, func=self.func, 
                                           ascending=self.ascending, input_stack=[col]))

class OneForOneFunction(Word):
    def __init__(self, name: str,
                    operation: Callable[[np.ndarray,np.ndarray],None],
                    slice_ = slice(-1, None), ascending: bool = True,
                    allow_sibling_drops: bool = True):
        self.ascending = ascending
        self.operation = operation
        self.allow_sibling_drops = allow_sibling_drops
        super().__init__(name, slice_)

    def __call__(self) -> Word:
    #def __call__(self, ascending: bool = True) -> Word:
        self.ascending = True
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        siblings = set()
        input_stack = []
        cache = {}
        for i, col in enumerate(inputs):
            sib = OneForOneFunctionColumn(
                        col.header,
                        siblings=siblings,
                        input_stack=input_stack,
                        operation=self.operation,
                        cache=cache,
                        input_column=col,
                        ascending=self.ascending,
                        allow_sibling_drops=self.allow_sibling_drops
                        )
            siblings.add(sib)
            stack.append(sib)

@dataclass(kw_only=True, eq=False)
class OneForOneFunctionColumn(SiblingColumn):
    operation: Callable[[np.ndarray,np.ndarray],None]
    ascending: bool = True

    def calculate(self) -> None:
        cols_values = []
        for i, col in enumerate(self.input_stack):
            cols_values.append(col.values)
        data =  np.concatenate(cols_values).reshape((len(self.range_),i+1),order='F')
        if not self.ascending:
            data = data[::-1]
        self.operation(data, out=self.cache[self.range_])

def rank(inputs: np.ndarray, out: np.ndarray) -> None:
    out[:] = inputs.argsort(axis=1).argsort(axis=1)


class Roll(Word):
    def operate(self, stack: list[Column]) -> None:
        stack.insert(0,stack.pop())

class Drop(Word):
    def operate(self, stack: list[Column]) -> None:
        stack.clear()

class Reverse(Word):
    def operate(self, stack: list[Column]) -> None:
        stack.reverse()

class Interleave(Word):
    def __call__(self, parts: int = 2) -> Word:
        self.parts = parts
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        if len(stack) == 0: return
        assert len(stack) % self.parts == 0, f'Stack not divisible by {self.parts}'
        count = len(stack) // self.parts
        last = 0
        lists = []
        for i in range(len(stack)+1):
            if i % count == 0 and i != 0:
                lists.append(stack[i-count:i])
                last = i
        del(stack[:last])
        stack += [val for tup in zip(*lists) for val in tup]

class HSort(Word):
    def operate(self, stack: list[Column]) -> None:
        stack.sort(key=lambda c: c.header)

class HeaderSet(Word):
    def __call__(self, *headers: str) -> Word:
        self.headers = headers[0].split(',') if len(headers) == 1 else headers
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        start = len(stack) - len(self.headers)
        assert start >= 0, \
            f"Insufficient columns for hset('{",".join(self.headers)}')"
        for i in range(start,len(stack)):
            stack[i].header = self.headers[i - start]

class HeaderSetAll(Word):
    def __call__(self, *headers: str) -> Word:
        self.headers = headers[0].split(',') if len(headers) == 1 else headers
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for i in np.arange(len(stack)) - len(stack):
            stack[i].header = self.headers[i % len(self.headers)]

class HeaderFormat(Word):
    def __call__(self, format_spec: str) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        for col in stack:
            col.header = self.format_spec.format(col.header)

class HeaderReplace(Word):
    def __call__(self, old: str, new: str = '') -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        for col in stack:
            col.header = col.header.replace(self.old, self.new)

class HeaderApply(Word):
    def __call__(self, header_func: Callable[[str],str]) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        for col in stack:
            col.header = self.header_func(col.header)

def alphabet_generator():
    n = 0
    while True:
        result = ''
        num = n
        while True:
            result = chr(97 + (num % 26)) + result
            num = num // 26 - 1
            if num < 0:
                break
        yield result
        n += 1

class HeaderAlphabetize(Word):
    def operate(self, stack: list[Column]) -> None:
        gen = alphabet_generator()
        for col in stack:
            col.header = next(gen)

def zero_first_op(x: np.ndarray, out: np.ndarray) -> None:
    out[1:] = x[1:]
    out[0] = 0.0

def zero_to_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.equal(x,0),np.nan,x)

def is_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.isnan(x), 1, 0)

def inc_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = x + 1

def dec_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = x - 1

def expanding_mean(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        return np.add.accumulate(x) / (np.arange(len(x)) + 1)
    else:
        return np.add.accumulate(x) / (np.arange(len(x)) + 1)[:,None]

def expanding_var(x: np.ndarray) -> np.ndarray:
    cumsumOfSquares = np.add.accumulate(x * x)
    cumsum = np.add.accumulate(x)
    count = np.add.accumulate(x / x)
    count[0] += 1
    return (cumsumOfSquares - cumsum * cumsum / count) / (count - 1)

def expanding_std(x: np.ndarray) -> np.ndarray:
    return np.sqrt(expanding_var(x))

def rolling_lag(x: np.ndarray, window: int) -> np.ndarray:
    window -= 1
    if len(x.shape) == 1:
        return np.concat([np.full(window, np.nan), x[:-window]])
    else:
        return np.concat([np.full((window,x.shape[1]), np.nan), x[:-window]])

def expanding_lag(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape,x[0])

def rolling_diff(x: np.ndarray, window: int) -> np.ndarray:
    window -= 1
    if len(x.shape) == 1:
        return np.concat([np.full(window, np.nan), x[window:] - x[:-window]])
    else:
        return np.concat([np.full((window,x.shape[1]), np.nan), x[window:] - x[:-window]])

def expanding_diff(x: np.ndarray) -> np.ndarray:
    return x - x[0]

def rolling_ret(x: np.ndarray, window: int) -> np.ndarray:
    window -= 1
    if len(x.shape) == 1:
        return np.concat([np.full(window, np.nan), x[window:] / x[:-window] - 1])
    else:
        return np.concat([np.full((window,x.shape[1]), np.nan), x[window:] / x[:-window] - 1])

def expanding_ret(x: np.ndarray) -> np.ndarray:
    return x / x[0] - 1

def rolling_cov(x: np.ndarray, window: int) -> np.ndarray:
    means = bn.move_mean(x, window, axis=0)
    meanXY = bn.move_mean(np.multiply.reduce(x,axis=1), window)
    return meanXY - np.multiply.reduce(means,axis=1)

def rolling_cor(x: np.ndarray, window: int) -> np.ndarray:
    vars_ = bn.move_var(x, window, axis=0)
    return rolling_cov(x, window) /  np.multiply.reduce(vars_,axis=1)

def rolling_ewma(data: np.ndarray, window: int) -> np.ndarray:
    alpha = 2 / (window + 1.0)
    scale = np.power(1 - alpha, np.arange(data.shape[0] + 1))
    adj_scale = (alpha * scale[-2]) / scale[:-1]
    if len(data.shape) == 2:
        adj_scale = adj_scale[:, None]
        scale = scale[:, None]
    offset = data[0] * scale[1:]
    return np.add.accumulate(data * adj_scale) / scale[-2::-1] + offset

def rolling_ewv(data: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_ewma(data, window)
    delta = data - mean

    alpha = 2 / (window + 1.0)
    ws = (1 - alpha) ** np.arange(window)
    w_sum = ws.sum()
    bias = (w_sum ** 2) / ((w_sum ** 2) - (ws ** 2).sum())
    return rolling_ewma(delta ** 2, window) 

def rolling_ews(data: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(rolling_ewv(data, window))

def rolling_zsc(data: np.ndarray, window: int) -> np.ndarray:
    return (data - bn.move_mean(data, window=window, axis=0, min_count=2)) / \
                bn.move_std(data, window=window, axis=0, min_count=2)

vocab: dict[str, tuple[str,str,Callable[[str],Word]]] = {}

cat = 'Column Creation'
vocab['c'] = (cat, 'Pushes constant columns for each of _values_', Constant)
vocab['r'] = (cat, 'Pushes constant columns for each whole number from 0 to _n_ - 1', ConstantRange)
vocab['nan'] = (cat, 'Pushes a constant nan-valued column', lambda name: Constant(name)(np.nan))
vocab['randn'] = (cat, 'Pushes a column with values from a random normal distribution', RandomNormal)
vocab['ts'] = (cat, 'Pushes a column with the timestamp of the end of the period', Timestamp)
vocab['po'] = (cat, 'Pushes a column with the period ordinal', PeriodOrdinal)
vocab['dc'] = (cat, 'Pushes a column with the number of days in the period', Daycount)
vocab['saved'] = (cat, 'Pushes columns saved to internal DB as _key_', Saved)
vocab['pandas'] = (cat,'Pushes columns from Pandas DataFrame or Series _pandas_', FromPandas)

cat = 'Stack Manipulation'
vocab['id'] = (cat, 'Identity/no-op', lambda name: Word(name))
vocab['pull'] = (cat, 'Brings selected columns to the top', lambda name: Word(name, raise_on_empty=True))
vocab['dup'] = (cat, 'Duplicates columns',
                lambda name: Word(name, slice_=slice(-1,None), copy_selected=True, raise_on_empty=True))
vocab['filter'] = (cat, 'Removes non-selected columns',
                lambda name: Word(name, discard_excluded=True))
vocab['drop'] = (cat, 'Removes selected columns',
             lambda name: Word(name, inverse_selection=True,
                            discard_excluded=True, copy_selected=False, slice_=slice(-1,None)))
vocab['nip'] = (cat, 'Removes non-selected columns, defaulting selection to top',
                lambda name: Word(name, discard_excluded=True, slice_=slice(-1,None)))
vocab['roll'] = (cat, 'Permutes selected columns', Roll)
vocab['swap'] = (cat, 'Swaps top and bottom selected columns', lambda name: Roll(name, slice(-2,None)))
vocab['rev'] = (cat, 'Reverses the order of selected columns', Reverse)
vocab['interleave'] = (cat, 'Divides columns in _parts_ groups and interleaves the groups', Interleave)
vocab['hsort'] = (cat, 'Sorts columns by header', HSort)

vocab['q'] = ('Quotation', 'Wraps the following words until *p* as a quotation, or ' \
              + 'wraps _quoted_ expression as a quotation', Quotation)

cat = 'Header manipulation'
vocab['hset'] = (cat, 'Set headers to _*headers_ ', HeaderSet)
vocab['hsetall'] = (cat, 'Set headers to _*headers_ repeating, if necessary', HeaderSetAll)
vocab['hformat'] = (cat, 'Apply _format_spec_ to headers', HeaderFormat)
vocab['hreplace'] = (cat, 'Replace _old_ with _new_ in headers', HeaderReplace)
vocab['happly'] = (cat, 'Apply _header_func_ to headers_', HeaderApply)
vocab['halpha'] = (cat, 'Set headers to alphabetical values', HeaderAlphabetize)

cat = 'Combinators'
vocab['call'] = (cat, 'Applies quotation',Call)
vocab['map'] = (cat, 'Applies quotation in groups of _every_',Map)
vocab['repeat'] = (cat, 'Applies quotation _times_ times',Repeat)
vocab['hmap'] = (cat, 'Applies quotation to stacks created grouping columns by header',HMap)
vocab['cleave'] = (cat, 'Applies all preceding quotations',Cleave)
vocab['ifexists'] = (cat, 'Applies quotation if stack has at least _count_ columns',IfExists)
vocab['ifexistselse'] = (cat, 'Applies top quotation if stack has at least _count_ columns' \
                         + ', otherwise applies second quotation',IfExistsElse)
vocab['ifheaders'] = (cat, 'Applies top quotation if list of column headers fulfills _predicate_',IfHeaders)
vocab['ifheaderselse'] = (cat, 'Applies quotation if list of column headers fulfills _predicate_' \
                         + ', otherwise applies second quotation', IfHeadersElse)
vocab['partial'] = (cat, 'Pushes stack columns to the front of quotation',Partial)
vocab['compose'] = (cat, 'Combines quotations', Compose)

cat = 'Data cleanup'
vocab['fill'] = (cat, 'Fills nans with _value_ ',Fill)
vocab['ffill'] = (cat, 'Fills nans with previous values, looking back _lookback_ before range ' \
                  + 'and leaving trailing nans unless not _leave_end_', FFill)
vocab['join'] = (cat, 'Joins two columns at _date_', Join)
vocab['zero_first'] = (cat, 'Changes first value to zero',
                    lambda name: OneForOneFunction(name, zero_first_op))
vocab['zero_to_na'] = (cat, 'Changes zeros to nans',
                    lambda name: OneForOneFunction(name, zero_to_na_op))
vocab['resample_sum'] = (cat, 'Sets periodicity resampling method to sum',
                    lambda name: Resample(name, ResampleMethod.SUM))
vocab['resample_last'] = (cat, 'Sets periodicity resampling method to last',
                    lambda name: Resample(name, ResampleMethod.LAST))
vocab['resample_avg'] = (cat, 'Sets periodicity resampling method to avg',
                    lambda name: Resample(name, ResampleMethod.AVG))
vocab['per'] = (cat, 'Changes column periodicity to _periodicity_, then resamples', SetPeriodicity)
vocab['start'] = (cat, 'Changes period start to _start_, then resamples', SetStart)



_funcs = [
    ('add',  'Addition', partial(bn.nansum, axis=1),
                partial(bn.move_sum, axis=0), np.add.accumulate),
    ('sub',  'Subtraction', partial(np.subtract.reduce, axis=1),
                None, np.subtract.accumulate),
    ('mul',  'Multiplication', partial(np.multiply.reduce, axis=1),
                None, np.multiply.accumulate),
    ('div',  'Division', partial(np.divide.reduce, axis=1),
                None, None),
    ('pow',  'Power', partial(np.power.reduce, axis=1),
                None, None),
    ('mod',  'Modulo', partial(np.mod.reduce, axis=1),
                None, None),
    ('avg',  'Arithmetic average', partial(bn.nanmean, axis=1),
                partial(bn.move_mean, axis=0, min_count=2), expanding_mean),
    ('std',  'Standard deviation', partial(bn.nanstd, axis=1),
                partial(bn.move_std, axis=0), expanding_std),
    ('var',  'Variance', partial(bn.nanvar, axis=1),
                partial(bn.move_var, axis=0), expanding_var),
    ('min',  'Minimum', partial(bn.nanmax, axis=1),
                partial(bn.move_min, axis=0), np.minimum.accumulate),
    ('max',  'Maximum', partial(bn.nanmin, axis=1),
                partial(bn.move_max, axis=0), np.maximum.accumulate),
    ('med',  'Median', partial(bn.nanmedian, axis=1),
                partial(bn.move_median, axis=1), None),
    ('lag',  'Lag', None, rolling_lag, expanding_lag),
    ('dif',  'Lagged difference', None, rolling_diff, expanding_diff),
    ('ret',  'Lagged return', None, rolling_ret, expanding_ret),
    ('ewm',  'Exponentially-weighted moving average', None, rolling_ewma, None),
    ('ewv',  'Exponentially-weighted variance', None, rolling_ewv, None),
    ('ews',  'Exponentially-weighted standard deviation', None, rolling_ews, None),
    ('zsc',  'Z-score', None, rolling_zsc, None),

    ]
for code, desc, red, roll, scan in _funcs:
    if red:
        vocab[code] = ('Row-wise Reduction', desc,
                    lambda name, func=red: Reduction(name, func))
        vocab[f'n{code}'] = ('Row-wise Reduction Ignoring NaNs', desc,
                    lambda name, func=red: Reduction(name, func)(True))
    if roll:
        vocab[f'r{code}'] = ('Rolling Window', desc,
                     lambda name, func=roll: Rolling(name, func))
    if scan:
        vocab[f'c{code}'] = ('Cumulative', desc,
                     lambda name, func=scan: Accumulator(name, func))
        vocab[f'rc{code}'] = ('Reverse Cumulative', desc,
                     lambda name, func=scan: Accumulator(name, func, ascending=False))
vocab['rcov'] = ('Rolling Window', 'Covariance', 
                 lambda name: RollingReduction(name, rolling_cov, slice_=slice(-2,None)))
vocab['rcor'] = ('Rolling Window', 'Correlation', 
                 lambda name: RollingReduction(name, rolling_cor, slice_=slice(-2,None)))
vocab['rewm'] = ('Rolling Window', 'Exponentially-weighted average', 
                 lambda name: Rolling(name, rolling_ewma))


cat = 'One-for-one functions'
vocab['neg'] = (cat, 'Additive inverse',lambda name: OneForOneFunction(name,np.negative))
vocab['inv'] = (cat, 'Multiplicative inverse',lambda name: OneForOneFunction(name,np.reciprocal))
vocab['abs'] = (cat, 'Absolute value',lambda name: OneForOneFunction(name,np.abs))
vocab['sqrt'] = (cat, 'Square root',lambda name: OneForOneFunction(name,np.sqrt))
vocab['log'] = (cat, 'Natural log',lambda name: OneForOneFunction(name,np.log))
vocab['exp'] = (cat, 'Exponential',lambda name: OneForOneFunction(name,np.exp))
vocab['lnot'] = (cat, 'Logical not',lambda name: OneForOneFunction(name, np.logical_not))
vocab['expm1'] = (cat, 'Exponential minus one',lambda name: OneForOneFunction(name,np.expm1))
vocab['log1p'] = (cat, 'Natural log of increment',lambda name: OneForOneFunction(name,np.log1p))
vocab['sign'] = (cat, 'Sign',lambda name: OneForOneFunction(name, np.sign))
vocab['rank'] = (cat, 'Row-wise rank',
                 lambda name: OneForOneFunction(name, rank,
                                    slice_=slice(None), allow_sibling_drops=False))
vocab['inc'] = (cat, 'Increment',lambda name: OneForOneFunction(name, inc_op))
vocab['dec'] = (cat, 'Decrement',lambda name: OneForOneFunction(name, dec_op))

def resolve(name: str, throw_exception: bool = True) -> Word | None:
    if re.match(r'c\d[_\d]*',name) is not None:
        return Constant('c')(float(name[1:].replace('_','.')))
    elif re.match(r'c_\d[_\d]*',name) is not None:
        return Constant('c')(-float(name[2:].replace('_','.')))
    elif re.match(r'r\d+',name) is not None:
        return Constant('c')(*range(int(name[1:])))
    elif name in vocab:
        return vocab[name][-1](name)
    else:
        if throw_exception:
            raise NameError(f"name '{name}' is not defined")
        else:
            return 

