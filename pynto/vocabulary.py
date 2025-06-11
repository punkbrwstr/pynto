from __future__ import annotations
import re
import sys
import copy
import warnings
import numbers
import string
import logging
import itertools
import datetime
import numpy as np
import bottleneck as bn
import numpy.ma as ma
import pandas as pd
import traceback
from collections import deque
from dataclasses import dataclass, field, KW_ONLY
from functools import partial,reduce
from operator import add
from typing import Callable, Any
from . import database as db
from .periods import Range, Periodicity, datelike

logger = logging.getLogger(__name__)

def set_debug():
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ColumnIDGenerator:
    def __init__(self):
        self.current = 0

    def get_next(self):
        self.current += 1
        return self.current

    def reset(self):
        self.current = 0

_IDs = ColumnIDGenerator()

@dataclass(eq=False)
class Column:
    header: str = ''
    range_: Range | None = None
    output_columns: int = 1
    input_stack: list[Column] = field(default_factory=list)
    cache: dict[Range,None | np.ndarray] = field(default_factory=dict)
    id_: int = field(default_factory=_IDs.get_next)

    def calculate(self) -> None:
        pass

    @property
    def values(self) -> np.ndarray:
        return self.cache[self.range_]

    @values.setter
    def values(self, values: np.ndarray) -> np.ndarray:
        self.cache[self.range_] = values

    @property
    def input_matrix(self) -> np.ndarray:
        shape = (len(self.input_stack[0].range_),len(self.input_stack))
        return np.concatenate([col.values for col in self.input_stack]) \
                        .reshape(shape,order='F')

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
        copied = copy.copy(super())
        copied.id_ = _IDs.get_next()
        logger.debug('  copy '+ debug_col_repr(copied, 4))
        return copied

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other: Column):
        return self.id_ == other.id_

    def __del__(self):
        logger.debug('destroying '+ debug_col_repr(self, 4))

@dataclass(kw_only=True,eq=False)
class SiblingColumn(Column):
    siblings: set[SiblingColumn]
    ordinal: int | None = None
    input_column: Column | None = None
    allow_sibling_drops: bool = True

    def set_range(self, range_: Range) -> None:
        self.setup_siblings()
        super().set_range(range_)

    def setup_siblings(self) -> None:
        if self.ordinal is None:
            output_columns = len(self.siblings)
            for i, sib in enumerate(self.siblings):
                sib.ordinal = i
                sib.output_columns = output_columns
                if sib.input_column:
                    self.input_stack.append(sib.input_column)
                    sib.input_column = None
            self.siblings.clear()

    @property
    def values(self) -> np.ndarray:
        return self.cache[self.range_][:, [self.ordinal]]

    def drop(self):
        if self.allow_sibling_drops:
            self.siblings.remove(self)
        super().drop()

class Word:
    def __init__(self, name: str, slice_: slice = slice(None),
                    copy_selected: bool = False, discard_excluded: bool = False,
                 inverse_selection: bool = False):
        self.name = name
        self.slice_ = slice_
        self.copy_selected = copy_selected
        self.discard_excluded = discard_excluded
        self.inverse_selection = inverse_selection
        self.filters: list[str] | None = None
        self.next_: Word | None = None
        self.prev: Word | None = None
        self.open_quotes: int = 0
        self.called: bool = False

    def operate(self, stack: list[Column]) -> None:
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
        if isinstance(addend_head, Combinator):
            current = this
            while addend_head.num_quotations != 0 \
                    and current is not None and isinstance(current, Quotation):
                addend_head.quotations.append(current)
                addend_head.num_quotations -= 1
                current = current.prev
            assert addend_head.num_quotations <= 0, \
                'Missing quotation for combinator'
            this = current
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


    def build_stack(self, stack: list[Column]) -> None:
        assert self.open_quotes == 0, 'Unclosed quotation.  Cannot evaluate'
        assert not isinstance(self, Quotation), 'Cannot evaluate quotation'
        current = self._head()
        logger.debug('Build stack')
        while current is not None:
            if not current.called: # need to set defaults (if any)
                current()
            logger.debug(f'   current word {current} slice {current.slice_}')
            selected = list(range(len(stack))[current.slice_])
            if current.filters:
                filtered = []
                for filter_ in current.filters:
                    pattern = re.compile(filter_)
                    for i in range(len(stack)):
                        if bool(pattern.match(stack[i].header)):
                            filtered.append(i)
                selected = filtered
            logger.debug(f'      selected {selected}')
            if current.inverse_selection:
                selected = list(set(range(len(stack))) - set(selected))
            logger.debug(f'      selected {selected}')
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
                    logger.debug(f'      dropping {debug_col_repr(stack[i],4)}')
                    stack[i].drop() # no longer needed as siblings   
                to_delete.update(excluded)
            for i in sorted(to_delete, reverse=True):
                del(stack[i])
            current.operate(current_stack)
            stack.extend(current_stack)
            current = current.next_

    @property
    def stack(self) -> list[Column]:
        s = []
        self.build_stack(s)
        return s

    @property
    def columns(self) -> list[str]:
        return [col.header for col in self.stack]

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
        return Quotation('q')(self).call(0)

    def __str__(self) -> str:
        s = self.name
        str_args = []
        for k,v in self.__dict__.items():
            if k != self and k not in ['prev','next_','closed', 'called']:
                if isinstance(v, str):
                    str_args.append(f"{k}='{v[:2000]}'")
                    if len(v) > 2000:
                        str_args[-1] += '...'
                else:
                    str_args.append(f"{k}={str(v)}")
        s += '(' + ', '.join(str_args) + ')'
        return s

    def __repr__(self) -> str:
        s = ''
        current = self
        while True:
            s = current.__str__() + s
            if current.prev is None:
                break
            else:
                s = '.' + s
                current = current.prev
        s = 'pt.' + s
        return s

debug_col_repr = lambda c, n: f'{' '*n}{c.id_}-{c.__class__.__name__}' \
        + f' "{c.header}"' \
        + f' range {c.range_}' \
        + f' cache #{str(id(c.cache))[-4:]})'
def debug_stack(stack):   
    for c in stack:
        logger.debug(debug_col_repr(c,4))
        if c.input_stack:
            logger.debug((' ' * 8) + 'inputs')
        for c2 in c.input_stack:
            logger.debug(debug_col_repr(c2,10))
        if hasattr(c,'siblings'):
            logger.debug((' ' * 8) + 'sibs')
            for c2 in c.siblings:
                logger.debug(debug_col_repr(c2,10))
                logger.debug((' ' * 10) + 'input')
                if c2.input_column:
                    logger.debug(debug_col_repr(c2.input_column,12))

class Evaluator:
    def __init__(self, word: Word, values_only: bool = False):
        self.word = word
        self.values_only = values_only

    def __getitem__(self, key: slice | int | str | Range) -> pd.DataFrame:
        _IDs.reset()
        stack = []
        self.word.build_stack(stack)
        min_, max_, per = None, None, None
        working = stack[:]
        flat = []
        logger.debug('Stack before')
        debug_stack(stack)
        while working:
            col = working.pop()
            bounds = col.get_bounds()
            if bounds:
                min_ = bounds[0] if not min_ else max(min_, bounds[0])
                max_ = bounds[1] if not max_ else min(max_, bounds[1])
                per = bounds[2]
            flat.append(col)
            working.extend(col.input_stack)
        flat.reverse()
        p = Periodicity[key.step] if hasattr(key, 'step') and key.step \
            else per if per else Periodicity.B
        if isinstance(key, Range):
            range_ = key
        elif isinstance(key, str):
            range_ = p[key]
        else:
            if isinstance(key, int):
                key = slice(key, key + 1)
            range_ = p[key.start or min_:key.stop or max_]
        for col in stack:
            logger.debug('setting range'+ debug_col_repr(col, 4))
            col.set_range(range_)
        logger.debug('Stack after')
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
            offsets, needed_saveds = [], []
            for col in saveds:
                if not col.calculated:
                    col.cache[col.range_] = np.full(len(col.range_), np.nan, order='F')
                    offsets.append(db.get_client()._req(
                        col.md, col.range_.start, col.range_.stop, p))
                    needed_saveds.append(col)
            for col, offset, bytes_ in zip(needed_saveds, offsets, p.execute()):
                if len(bytes_) > 0:
                    data = np.frombuffer(bytes_, col.md.type_.dtype)
                    col.values[offset:offset + len(data)] = data
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
        for i, col in enumerate(stack):
            assert col.range_ == range_, \
                f'Mismatching range needs resample for col {i - len(stack)}: {col.header}'
        values = np.concatenate([col.values for col in stack]).reshape((len(range_),len(stack)),order='F')
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
        values[:] = range_.to_index().view('int').astype(np.float64)

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
        return (self.pandas_range[0].last_date,
                self.pandas_range.expand()[-1].last_date,
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

    def set_range(self, range_: Range) -> None:
        self.range_ = range_.change_periodicity(self.md.periodicity)

    def get_bounds(self) -> tuple[datetime.date,datetime.date,Periodicity] | None:
        return (self.md[0].last_date, self.md.expand()[-1].last_date, self.md.periodicity)

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

    def operate(self, stack: list[Column]) -> None:
        raise TypeError('Quotation must be pared with a combinator for evalutation')

class Combinator(Word):
    def __init__(self, name: str, slice_: slice = slice(None), num_quotations: int = 1):
        self.num_quotations = num_quotations
        self.quotations = []
        super().__init__(name, slice_)

    def operate(self, stack: list[Column]) -> None:
        pass

class Call(Combinator):
    def operate(self, stack: list[Column]) -> None:
        self.quotations[0].quoted.build_stack(stack)

class IfExists(Combinator):
    def __call__(self, count: int = 1) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if len(stack) >= self.count:
            self.quotations[0].quoted.build_stack(stack)

class IfExistsElse(Combinator):
    def __init__(self, name: str):
        super().__init__(name, num_quotations = 2)

    def __call__(self, count: int = 1) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if len(stack) >= self.count:
            self.quotations[0].quoted.build_stack(stack)
        else:
            self.quotations[1].quoted.build_stack(stack)

class IfHeaders(Combinator):
    def __call__(self, predicate: Callable[[list[str]],bool]) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if self.predicate([col.header for col in stack]):
            self.quotations[0].quoted.build_stack(stack)

class IfHeadersElse(Combinator):
    def __init__(self, name: str):
        super().__init__(name, num_quotations = 2)

    def __call__(self, predicate: Callable[[list[str]],bool]) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if self.predicate([col.header for col in stack]):
            self.quotations[0].quoted.build_stack(stack)
        else:
            self.quotations[1].quoted.build_stack(stack)

class Map(Combinator):
    def __call__(self, every: int = 1):
        self.every = every
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        assert len(stack) % self.every == 0, \
           f'Stack length {len(stack)} not evenly divisible by every {self.every}'
        copied = stack[:]
        stack.clear()
        for t in zip(*[iter(copied)]*self.every):
            this_stack = list(t)
            self.quotations[0].quoted.build_stack(this_stack)
            stack += this_stack

class Repeat(Combinator):
    def __call__(self, times: int = 2) -> Word:
        self.times = times
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for _ in range(self.times):
            self.quotations[0].quoted.build_stack(stack)

class HMap(Combinator):
    def operate(self, stack: list[Column]) -> None:
        new_stack = []
        for header in set([c.header for c in stack]):
            to_del, filtered_stack = [], []
            for i,col in enumerate(stack):
                if header == col.header:
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            self.quotations[0].quoted.build_stack(filtered_stack)
            new_stack += filtered_stack
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        del(stack[:])
        stack += new_stack

class Cleave(Combinator):
    def __init__(self, name):
        super().__init__(name, num_quotations = -1)

    def operate(self, stack: list[Column]) -> None:
        output = []
        for quote in reversed(self.quotations):
            this_stack = [copy.copy(col) for col in stack]
            quote.quoted.build_stack(this_stack)
            output += this_stack
        stack.clear()
        stack.extend(output)

class BoundWord(Word):
    def __init__(self):
        super().__init__('bound')

    def __init__(self, bound: list[Column]):
        self.bound = bound
        super().__init__()

    def operate(self, stack: list[Column]) -> None:
        stack.extend(self.bound)

class Partial(Quotation, Combinator):
    def operate(self, stack: list[Column]) -> None:
        self.quoted = BoundWord(stack) + self.quoted

class Compose(Quotation, Combinator):
    def __init__(self, name: str):
        Combinator.__init__(self, name, num_quotations=2)

    def __call__(self, num_quotations: int = 2):
        self.num_quotations = num_quotations
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        self.quoted = reduce(add, [quote.quoted for quote in self.quotations])

@dataclass(kw_only=True, eq=False)
class ResampleColumn(Column):
    round_: bool

    def calculate(self) -> None:
        self.values[:] = np.nan
        idx, idx_input = self.input_stack[0].range_ \
                            .resample_indicies(self.range_, self.round_)
        self.values[idx] = self.input_stack[0].values[idx_input]

class Resample(Word):
    def __call__(self, round_: bool = False):
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for col in inputs:
            stack.append(ResampleColumn(col.header, 
                        round_=self.round_, inputs=[col]))

@dataclass(kw_only=True, eq=False)
class PeriodicityColumn(ResampleColumn):
    periodicity: Periodicity

    def set_input_range(self) -> None:
        input_range = self.range_.change_periodicity(self.periodicity)
        for col in self.input_stack:
            col.set_range(input_range)

class SetPeriodicity(Word):
    def __call__(self, periodicity: str | Periodicity, \
                    round_: bool = False) -> Word:
        if isinstance(periodicity, str):
            self.periodicity = Periodicity[periodicity]
        else:
            self.periodicity = periodicity
        self.round_ = round_
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for col in inputs:
            stack.append(PeriodicityColumn(col.header, round_=self.round_,
                               periodicity=self.periodicity, input_stack=[col]))

@dataclass(kw_only=True, eq=False)
class StartColumn(ResampleColumn):
    start: datelike

    def set_input_range(self) -> None:
        input_range = Range(self.range_.periodicity[self.start].start,
                            self.range_.stop, self.range_.periodicity)
        for col in self.input_stack:
            col.set_range(input_range)

class SetStart(Word):
    def __call__(self, start: datelike, \
                    round_: bool = False) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for col in inputs:
            stack.append(StartColumn(col.header, round_=self.round_,
                               start=self.start, input_stack=[col]))

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
        range_ = self.range_
        cutover_index = range_.periodicity[self.date].start
        if range_.stop < cutover_index:
            self.input_stack.pop()
            self.input_stack[0].set_range(range_)
        elif range_.start >= cutover_index:
            self.input_stack.pop(-2)
            self.input_stack[0].set_range(range_)
        else:
            r_first = copy.copy(range_)
            r_first.stop = cutover_index
            self.input_stack[0].set_range(r_first)
            r_second = copy.copy(range_)
            r_second.start = cutover_index
            self.input_stack[1].set_range(r_second)

    def calculate(self) -> None:
        i = 0
        for input_ in self.input_stack:
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

    def calculate(self) -> None:
        self.values[:] = self.operation(self.input_matrix)[:, None]

class Reduction(Word):
    def __init__(self, name: str, operation: Callable[[np.ndarray, np.ndarray], None]):
        self.operation = operation
        super().__init__(name, slice(-2, None))

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        stack.append(ReductionColumn(inputs[0].header,
                    operation=self.operation,
                    input_stack=inputs))

@dataclass(kw_only=True, eq=False)
class RollingColumn(SiblingColumn):
    operation: Callable[[np.ndarray, int], np.ndarray]
    window: int
    reduction: bool = False

    def set_input_range(self) -> None:
        input_range = self.range_.expand(int(-self.window * 1.25))
        for col in self.input_stack:
            col.set_range(input_range)

    def calculate(self) -> None:
        lookback = int(self.window * 1.25)
        data = self.input_matrix
        mask = ~np.any(np.isnan(data), axis=1)
        idx = np.where(mask)[0]
        start = np.where(idx >= lookback)[0][0]
        out = self.cache[self.range_]
        out[(idx - lookback)[start:]] = self.operation(data[mask], self.window)[start:]
        if not np.all(mask):
            idx = np.where(~mask)[0]
            if np.any(idx >= lookback):
                start = np.where(idx >= lookback)[0][0]
                out[(idx - lookback)[start:]] = np.nan


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

    def __call__(self, ascending: bool = True) -> Word:
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
        data = self.input_matrix
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
    def __call__(self, old: str, new: str) -> Word:
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
    letters = string.ascii_lowercase
    length = 1
    while True:
        for combination in itertools.product(letters, repeat=length):
            yield ''.join(combination)
            length += 1

class HeaderAlphabetize(Word):
    def operate(self, stack: list[Column]) -> None:
        gen = alphabet_generator()
        for col in stack:
            col.header = next(gen)

def zero_first_op(x: np.ndarray, out: np.ndarray) -> None:
    out[1:] = x
    out[0] = 0.0

def zero_to_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.equal(x,0),np.nan,x)

def is_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.isnan(x), 1, 0)

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


vocab: dict[str, tuple[str,str,Callable[[str],Word]]] = {}

cat = 'Column Creation'
vocab['c'] = (cat, 'Pushes constant columns for each of _values_', Constant)
vocab['r'] = (cat, 'Pushes constant columns for each whole number from 0 to _n_ - 1', ConstantRange)
vocab['nan'] = (cat, 'Pushes a constant nan-valued column', lambda name: Constant(name)(np.nan))
vocab['randn'] = (cat, 'Pushes a column with values from a random normal distribution', RandomNormal)
vocab['ts'] = (cat, 'Pushes a column with the timestamp of the end of the period', Timestamp)
vocab['dc'] = (cat, 'Pushes a column with the number of days in the period', Daycount)
vocab['saved'] = (cat, 'Pushes columns saved to internal DB as _key_', Saved)
vocab['pandas'] = (cat,'Pushes columns from Pandas DataFrame or Series _pandas_', FromPandas)

cat = 'Stack Manipulation'
vocab['pull'] = (cat, 'Brings selected columns to the top', lambda name: Word(name))
vocab['dup'] = (cat, 'Duplicates columns',
                lambda name: Word(name, slice_=slice(-1,None), copy_selected=True))
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
vocab['resample'] = (cat, 'Adapts periodicity to match range with optional rounding _round_',
                    Resample)
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
                partial(bn.move_mean, axis=0), expanding_mean),
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
    ('lag',  'Lag', None, rolling_lag, None),
    ('dif',  'Lagged difference', None, rolling_diff, expanding_diff),
    ('ret',  'Lagged return', None, rolling_ret, expanding_ret),
    ('ewm',  'Exponentially-weighted moving average', None, rolling_ewma, None),
    ('ewv',  'Exponentially-weighted variance', None, rolling_ewv, None),
    ('ews',  'Exponentially-weighted standard deviation', None, rolling_ews, None),

    ]
for code, desc, red, roll, scan in _funcs:
    if red:
        vocab[code] = ('Row-wise Reduction', desc,
                    lambda name, func=red: Reduction(name, func))
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


cat = 'Other functions'
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

