from __future__ import annotations
import re
import abc
import copy
import warnings
import numbers
import uuid
import datetime
import numpy as np
import numpy.ma as ma
import pandas as pd
import traceback
from . import database as db
from collections import deque
from dataclasses import dataclass, field
from functools import partial,reduce
from operator import add
from typing import Callable, Any
from . import tools
from .periods import Range,Periodicity



@dataclass(repr=False)
class Column:
    header: str
    range_: Range | None = None
    references: list[int] = field(default_factory=lambda: [1])
    values_cache: dict[Range,np.ndarray] = field(default_factory=dict)
    children: list[Column] = field(default_factory=list)
    siblings: list[Column] = field(default_factory=list)
    sibling_ordinal: int = 0
    calculated: bool = False

    def calculate(self) -> None:
        pass

    @property
    def values(self) -> np.ndarray:
        v = self.values_cache.get(self.range_)
        if v is None or len(self.siblings) == 0:
            return v
        else:
            return v[:,self.sibling_ordinal]

    @values.setter
    def values(self, values: np.ndarray) -> None:
        self.values_cache[self.range_] = values

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        for col in self.children:
            col.set_range(range_)

    def set_calculated(self) -> None:
        self.calculated = True
        for col in self.siblings:
            col.calculated = True


    def get_bounds(self) -> tuple[datetime.date,datetime.date,Periodicity] | None:
        return None

    def check_child_ranges(self) -> None:
        pass
       # for col in self.children:
       #     assert col.range_ == self.range_, 'Mismatching child range'

        
class Word(abc.ABC):
    def __init__(self, name: str, slice_: slice = slice(None)):
        self.name = name
        self.slice_ = slice_
        self.next_: Word | None = None
        self.prev: Word | None = None
        self.open_quotes: int = 0
        self.keep: bool = False
        self.filters: list[str] | None = None
        self.called: bool = False

    #@abc.abstractmethod
    def operate(self, stack: list[Column]) -> None:
        pass

    def __call__(self) -> Word:
        self.called = True
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
            assert len(key) == 2, 'Indexer must have 1 or 2 arguments'
            assert isinstance(key[1], bool), 'Keep argument must be boolean'
            self.keep = key[1]
            key = key[0]
        if isinstance(key, int):
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
        while current is not None:
            if not current.called: # need to set defaults (if any)
                current()
            if current.filters is None:
                current_stack = stack[current.slice_]
                if not current.keep:
                    del(stack[current.slice_])
            else:
                current_stack = []
                for header in current.headers:
                    to_del = []
                    matcher = lambda c: header == c.header if current.exact_match \
                                else re.match(header,col.header) is not None
                    for i,col in enumerate(stack):
                        if matcher(col):
                            current_stack.append(stack[i])
                            to_del.append(i)
                    to_del.sort(reverse=True)
                    for i in to_del:
                        del(stack[i])
            current.operate(current_stack)
            stack.extend(current_stack)
            current = current.next_
        for col in stack:
            col.references[0] += 1 # top level columns can't be overwritten

    @property
    def stack(self) -> list[Column]:
        s = []
        self.build_stack(s)
        return s

    @property
    def columns(self) -> list[str]:
        return [col.header for col in self.stack]

    @property
    def rows(self) -> Rows:
        return Rows(self)
    
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
        return Quotation()(self).call(0)

    def __str__(self) -> str:
        s = self.name
        str_args = []
        for k,v in self.__dict__.items():
            if k != self and k not in ['prev','next_','open_quotes','closed', 'called']:
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

class Rows:
    def __init__(self, word: Word):
        self.word = word

    def __getitem__(self, key: Range | slice) -> pd.DataFrame:
        stack = []
        self.word.build_stack(stack)
        min_, max_, per = None, None, None
        working = stack[:]
        flat = []
        while working:
            col = working.pop()
            bounds = col.get_bounds()
            if bounds:
                min_ = bounds[0].last_date if not min_ \
                    else max(min_, bounds[0].last_date)
                max_ = bounds[1].last_date if not max_ \
                    else min(max_, bounds[1].last_date)
                per = bounds[2]
            flat.append(col)
            working.extend(col.children)
        flat.reverse()
        if isinstance(key, Range):
            range_ = key 
        else:
            p = Periodicity[key.step] if hasattr(key, 'step') and key.step \
                else per if per else Periodicity.B
            range_ = p[key.start or min_:key.stop or max_]
        for col in stack:
            col.set_range(range_)
        saveds: list[SavedColumn] = []
        for col in flat:
            if isinstance(col, SavedColumn):
                saveds.append(col)
        if saveds:
            p = db.get_client().connection.pipeline()
            offsets = [db.get_client()._req(col.md, col.range_.start,
                        col.range_.stop, p) for col in saveds]
            for col, offset, bytes_ in zip(saveds, offsets, p.execute()):
                col.values = np.full(len(col.range_), np.nan, order='F')
                if len(bytes_) > 0:
                    data = np.frombuffer(bytes_, col.md.type_.dtype)
                    col.values[offset:offset + len(data)] = data
        for col in flat:
            for child in col.children:
                child.references[0] -= 1
                if col.values is None and child.references[0] == 0 \
                    and col.range_ == child.range_ \
                    and len(col.siblings) == 0:
                    col.values = child.values 
            if col.values is None:
                rows, cols = len(col.range_), len(col.siblings)
                shape = (rows, cols) if cols else rows
                col.values = np.empty(shape, order='F')
                #col.check_child_ranges()
            if not col.calculated:
                col.calculate()
                col.set_calculated()
            col.children.clear()
        for i, col in enumerate(stack):
            assert col.range_ == range_, \
                f'Mismatching range needs resample for col {i - len(stack)}: {col.header}'
        return pd.DataFrame(np.column_stack([col.values for col in stack]),
                            columns=[col.header for col in stack], index=range_.to_index())


# Nullary/generator words
class NullaryWord(Word):
    def __init__(self, name: str, generator: Callable[[Range, np.ndarray],None]):
        self.generator = generator
        super().__init__(name, slice(-1,0))

    def operate(self, stack: list[Column]) -> None:
        stack.append(NullaryColumn(self.name, generator=self.generator))

@dataclass(kw_only=True)
class NullaryColumn(Column):
    generator: Callable[[Range, np.ndarray],None]

    def calculate(self) -> None:
        self.generator(self.range_, self.values)

class RandomNormal(NullaryWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = np.random.randn(len(range_))

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
        values[:] = np.array(range_.day_counts(), dtype=np.float64)

    def __init__(self, name: str):
        super().__init__(name, self.generate)

class Constant(NullaryWord):
    @staticmethod
    def generate(constant: float, range_: Range, values: np.ndarray):
        values[:] = constant

    def __init__(self, name: str):
        super().__init__(name, self.generate)

    def __call__(self, *constants: list[float]) -> Word:
        self.constants = constants
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for constant in self.constants:
            stack.append(NullaryColumn(self.name,
                generator=partial(self.generate, constant)))

@dataclass(kw_only=True)
class SavedColumn(Column):
    md: db.Metadata

    def set_range(self, range_: Range) -> None:
        self.range_ = range_.change_periodicity(self.md.periodicity)

    def get_bounds(self) -> tuple[datetime.date,datetime.date] | None:
        return (self.md[0], self.md[-1], self.md.periodicity)

class Saved(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,0))

    def __call__(self, key: str) -> Word:
        self.key = key
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for md in db.get_client().get_metadata(self.key):
            stack.append(SavedColumn(md.col_header + md.row_header, (0, 1), md=md))


# Quotation / combinator words
class Quotation(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,0))
    
    def __call__(self, quoted: Word | None = None) -> Word:
        this = self.copy_expression()
        this.quoted = quoted
        return this

    def __getattr__(self, name):
        try:
            word = resolve(name)
            if  not hasattr(self, 'quoted'):
                self.open_quotes += 1 
            return self.__add__(word, False)
        except Exception as e:
            #raise e
            return self.__getattribute__(name)

    def __add__(self, other: Word, copy_addend: bool = True) -> Word:
        if not hasattr(self, 'quoted') and copy_addend:
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
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))

    def operate(self, stack: list[Column]) -> None:
        self.quotations[0].quoted.build_stack(stack)

class IfExists(Combinator):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))

    def __call__(self, count: int = 2, else_: bool = False, copy: bool = False) -> Word:
        self.count = count
        self.else_ = else_
        self.copy = copy
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        assert len(stack) == 0 or not isinstance(stack[-1],Quotation), \
                'Missing quotation for ifexists'
        quoted = stack.pop().quoted
        if self.else_:
            assert len(stack) == 0 or not isinstance(stack[-1],Quotation) ,  \
                'Missing second quotation for ifexists with else_=True'
            quoted_else = stack.pop().quoted
        if len(stack) >= self.count:
            quoted.build_stack(stack)
        elif self.else_:
            quoted_else.build_stack(stack)

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
            self.quotations[0].build_stack(filtered_stack)
            new_stack += filtered_stack
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        del(stack[:])
        stack += new_stack

class Cleave(Combinator):
    def __init__(self, name: str,  num_quotations: int = -1):
        return super().__call__(name, num_quotations=num_quotations)

    def operate(self, stack: list[Column]) -> None:
        for quote in self.quotations:
            this_stack = stack[:]
            quote.build_stack(this_stack)
            stack += this_stack

class BoundWord(Word):
    def __init__(self):
        super().__init__('bound')

    def __init__(self, bound: list[Column]):
        self.bound = bound
        super().__init__()

    def operate(self, stack: list[Column]) -> None:
        stack.extend(self.bound)

class Partial(Quotation, Combinator):
    def __init__(self):
        super().__init__('partial')

    def operate(self, stack: list[Column]) -> None:
        self.quoted = BoundWord(stack) + self.quoted

class Compose(Quotation, Combinator):
    def __init__(self, name: str):
        super(Combinator, self).__init__(name, num_quotations=2)

    def __call__(self, num_quotations: int = 2):
        self.num_quotations = num_quotations
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        self.quoted = reduce(add, [quote.quoted for quote in self.quotations])

@dataclass(kw_only=True)
class ResampleColumn(Column):
    round_: bool

    def check_child_ranges(self) -> None:
        pass

    def calculate(self) -> None:
        idx, idx_child = self.children[0].range_ \
                            .resample_indicies(self.range_, self.round_)
        self.values[idx] = self.children[0].values[idx_child]

class Resample(Word):
    def __call__(self, round_: bool = False):
        self.round_ = round_
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        stack.append(ResampleColumn(stack[-1].header, 
                        round_=self.round_, children=[stack.pop()]))

@dataclass(kw_only=True)
class PeriodicityColumn(Column):
    periodicity: Periodicity

    def set_range(self, range_: Range) -> None:
        self.range_ = range_.change_periodicity(self.periodicity)

class SetPeriodicity(Word):
    def __call__(self, periodicity: str | Periodicity) -> Word:
        if isinstance(periodicity, str):
            self.periodicity = Periodicity[periodicity]
        else:
            self.periodicity = periodicity
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        stack.append(ResampleColumn('per', periodicity=self.periodicity,
                        children=[stack.pop()]))

class Operator(Word):
    def __init__(self, name: str, operation: np.ufunc, slice_: slice):
        self.operation = operation
        super().__init__(name, slice_)

class BinaryOperator(Operator):
    def __init__(self, name: str,
                    operation: np.ufunc | Callable[[np.ndarray,np.ndarray],np.ndarray]):
        super().__init__(name, operation, slice(-2,None))

    def __call__(self, rolling: int | None = None,
                        accumulate: bool = False) -> Word:
        self.rolling, self.accumulate = rolling, accumulate
        if self.rolling:
            self.slice_ = slice(-1,None)
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        children = [*stack]
        stack.clear()
        if not self.rolling: 
            stack.append(BinaryOperatorColumn(children[0].header,
                        operation=self.operation,
                        children=children))
        else:
            for col in children:
                stack.append(BinaryOperatorColumn(col.header,
                        operation=self.operation, children=[col],
                        window=self.rolling))

@dataclass(kw_only=True)
class BinaryOperatorColumn(Column):
    operation: np.ufunc | Callable[[np.ndarray,np.ndarray,np.ndarray],None]
    window: int | None = None

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        if self.window:
            child_range = self.range_.expand(int(-self.window * 1.25))
        else:
            child_range = self.range_
        for col in self.children:
            col.set_range(child_range)


    def calculate(self) -> None:
        operands = np.column_stack([child.values for child in self.children])
        if not self.window: 
            self.operation.reduce(operands, out=self.values, axis=1)
        else:
            lookback = int(self.window * 1.25)
            expanded = self.children[0].values
            mask = ~np.isnan(expanded)
            no_nans = expanded[mask]
            indexes = np.add.accumulate(np.where(mask))[0]
            if no_nans.shape[-1] - self.window < 0:
                #warnings.warn('Insufficient non-nan values in lookback.')
                no_nans = np.hstack([np.full(self.window - len(no_nans),np.nan),no_nans])
            shape = no_nans.shape[:-1] + (no_nans.shape[-1] - self.window + 1, self.window)
            strides = no_nans.strides + (no_nans.strides[-1],)
            windows = np.lib.stride_tricks.as_strided(no_nans, shape=shape, strides=strides)
            td = np.full((expanded.shape[0],self.window), np.nan)
            td[indexes[-windows.shape[0]:],:] = windows
            self.operation.reduce(td[lookback:,::-1], out=self.values, axis=1)

class UnaryOperator(Operator):
    def __init__(self, name: str,
                    operation: np.ufunc | Callable[[np.ndarray,np.ndarray],None]):
        super().__init__(name, operation, slice(-1,None))

    def operate(self, stack: list[Column]) -> None:
        values_cache = {}
        siblings = []
        children = [*stack]
        stack.clear()
        for i, col in enumerate(children):
            siblings.append(UnaryOperatorColumn(col.header,
                        operation=self.operation,
                        children=children,
                        values_cache=values_cache,
                        siblings=siblings,
                        sibling_ordinal=i))
        stack.extend(siblings)

@dataclass(kw_only=True)
class UnaryOperatorColumn(Column):
    operation: np.ufunc | Callable[[np.ndarray,np.ndarray],None]

    def calculate(self) -> None:
        if len(self.children) > 1:
            operands = np.column_stack([child.values for child in self.children])
        else:
            operands = self.children[0].values
        self.operation(operands, out=self.values_cache[self.range_])

def rank(inputs: np.ndarray, out: np.ndarray) -> None:
    out[:] = inputs.argsort(axis=1).argsort(axis=1)


@dataclass(kw_only=True)
class EWMAColumn(Column):
    window: float
    fill_nans: bool

    def calculate(self) -> None:
        data = self.children[0].values
        idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
        nans = np.where(~np.isnan(data),1,np.nan)
        data = data[~np.isnan(data)]
        if len(data) == 0:
            self.values[:] = np.nan
        out = tools.ewma(data, 2 / (self.window + 1.0)) * nans
        self.values[:] = out[idx]

class EWMA(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))

    def __call__(self, window: float, fill_nans: bool = True) -> Word:
        self.window = window
        self.fill_nans = fill_nans
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        stack.append(EWMAColumn(self.name, window=self.window, \
                        fill_nans=self.fill_nans, children=[stack.pop()]))

class Duplicate(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))

    def operate(self, stack: list[Column]) -> None:
        for column in stack[:]:
            column.references[0] += 1
            stack.append(copy.copy(column))

class Roll(Word):
    def operate(self, stack: list[Column]) -> None:
        stack.insert(0,stack.pop())

class Drop(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))

    def operate(self, stack: list[Column]) -> None:
        stack.clear()

class Rev(Word):
    def operate(self, stack: list[Column]) -> None:
        stack.reverse()

class Interleave(Word):
    def __call__(self, count=None, split_into: int = 2) -> Word:
        self.count, self.split_into = count, split_into
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        if len(stack) == 0: return
        assert self.count or len(stack) % self.split_into == 0, \
                'Stack not divisible by ' + self.split_into 
        count = self.count or len(stack) // self.split_into
        last = 0
        lists = []
        for i in range(len(stack)+1):
            if i % count == 0 and i != 0:
                lists.append(stack[i-count:i])
                last = i
        del(stack[:last])
        stack += [val for tup in zip(*lists) for val in tup]

class Pop(Word):
    def __call__(self, count: int = 1) -> Word:
        self.count = count
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        del(stack[-abs(int(self.count)):])

class Top(Word):
    def __call__(self, count: int = 1) -> Word:
        self.count = count
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        del(stack[:-abs(int(self.count))])

class Pull(Word):
    def __call__(self, start: int, end: int | None = None, clear: bool = False):
        self.start, self.end, self.clear = start, end, clear
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        end = -self.start - 1 if self.end is None else -self.end
        start = len(stack) if self.start == 0 else -self.start
        pulled = stack[end:start]
        if self.clear:
            del(stack[:])
        else:
            del(stack[end:start])
        stack += pulled

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
            header = self.headers[i - start]
            stack[i].header = header

class HeaderPull(Word):
    def __call__(self, *headers: str, clear: bool = False, \
                    exact_match: bool = False) -> Word:
        self.headers = headers
        self.clear = clear
        self.exact_match = exact_match
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        filtered_stack = []
        for header in self.headers:
            to_del = []
            matcher = lambda c: header == c.header if self.exact_match \
                        else re.match(header,col.header) is not None
            for i,col in enumerate(stack):
                if matcher(col):
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        if self.clear:
            del(stack[:])
        stack += filtered_stack

class HeaderFilter(HeaderPull):
    def __call__(self, *headers: str, exact_match: bool = False) -> Word:
        return super().__call__(*headers, clear=True, exact_match=exact_match)


def zero_first_op(x: np.ndarray, out: np.ndarray) -> None:
    out[1:] = x
    out[0] = 0.0

def zero_to_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.equal(x,0),np.nan,x)

def is_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.isnan(x), 1, 0)

def logical_not_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.logical_not(x), 1, 0)

def logical_and_op(x: np.ndarray, y: np.ndarry, out: np.ndarray) -> None:
    out[:] = np.where(np.logical_and(x,y), 1, 0)

def logical_or_op(x: np.ndarray, y: np.ndarry, out: np.ndarray) -> None:
    out[:] = np.where(np.logical_or(x,y), 1, 0)

def logical_xor_op(x: np.ndarray, y: np.ndarry, out: np.ndarray) -> None:
    out[:] = np.where(np.logical_xor(x,y), 1, 0)


vocab: dict[str, type] = {}
def register_word(name: str, word_constructor: Callable[[str],Word]):
    vocab[name] = word_constructor(name)

register_word('rows',Rows)
register_word('f',Constant)
register_word('nan',lambda name: Constant(name)(np.nan))
register_word('saved',Saved)
register_word('randn',RandomNormal)
register_word('ts',Timestamp)
register_word('dc',Daycount)
register_word('q',Quotation)
register_word('call',Call)
register_word('ifexists',IfExists)
register_word('partial',Partial)
register_word('compose', Compose)
register_word('each',Map)
register_word('map',Map)
register_word('repeat',Repeat)
register_word('heach',HMap)
register_word('hmap',HMap)
register_word('cleave',Cleave)
register_word('resample',Resample)
register_word('per',SetPeriodicity)
register_word('add',lambda name: BinaryOperator(name,np.add))
register_word('sub',lambda name: BinaryOperator(name,np.subtract))
register_word('mul',lambda name: BinaryOperator(name,np.multiply))
register_word('pow',lambda name: BinaryOperator(name,np.power))
register_word('div',lambda name: BinaryOperator(name,np.divide))
register_word('mod',lambda name: BinaryOperator(name,np.mod))
register_word('expm1',lambda name: BinaryOperator(name,np.expm1))
register_word('log1p',lambda name: BinaryOperator(name,np.log1p))

register_word('eq',lambda name: BinaryOperator(name,np.equal))
register_word('ne',lambda name: BinaryOperator(name,np.not_equal))
register_word('ge',lambda name: BinaryOperator(name,np.greater_equal))
register_word('gt',lambda name: BinaryOperator(name,np.greater))
register_word('le',lambda name: BinaryOperator(name,np.less_equal))
register_word('lt',lambda name: BinaryOperator(name,np.less))
register_word('logical_and',lambda name: BinaryOperator(name, logical_and_op))
register_word('logical_or',lambda name: BinaryOperator(name, logical_or_op))
register_word('logical_xor',lambda name: BinaryOperator(name, logical_xor_op))
register_word('neg',lambda name: UnaryOperator(name,np.negative))
register_word('inv',lambda name: UnaryOperator(name,np.reciprocal))
register_word('abs',lambda name: UnaryOperator(name,np.abs))
register_word('sqrt',lambda name: UnaryOperator(name,np.sqrt))
register_word('exp',lambda name: UnaryOperator(name,np.exp))
register_word('log',lambda name: UnaryOperator(name,np.log))
register_word('zero_first',lambda name: UnaryOperator(name, zero_first_op))
register_word('zero_to_na',lambda name: UnaryOperator(name, zero_to_na_op))
register_word('logical_not',lambda name: UnaryOperator(name, logical_not_op))
register_word('rank',lambda name: UnaryOperator(name, rank, -1))
register_word('ewma', EWMA)
register_word('dup', Duplicate)
register_word('roll', Roll)
register_word('swap', lamdbda name: Roll(name, slice(-2,None))
register_word('drop', Drop)
register_word('clear', lamdbda name: Drop(name, slice(None))
register_word('rev', Rev)
register_word('clear', Clear)
register_word('interleave', Interleave)
register_word('pop', Pop)
register_word('top', Top)
register_word('pull', Pull)
register_word('hsort', HSort)
register_word('hset', HeaderSet)

def resolve(name: str, throw_exception: bool = True) -> Word | None:
    if re.match(r'f\d[_\d]*',name) is not None:
        return Constant()(float(name[1:].replace('_','.')))
    elif re.match(r'f_\d[_\d]*',name) is not None:
        return Constant()(-float(name[2:].replace('_','.')))
    elif name in vocab:
        return vocab[name]()
    else:
        if throw_exception:
            raise NameError(f"name '{name}' is not defined")
        else:
            return 
