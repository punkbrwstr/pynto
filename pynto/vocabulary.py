from __future__ import annotations
import re
import copy
import warnings
import numbers
import string
import itertools
import datetime
import numpy as np
import bottleneck as bn
import numpy.ma as ma
import pandas as pd
import traceback
from collections import deque
from dataclasses import dataclass, field
from functools import partial,reduce
from operator import add
from typing import Callable, Any
from . import tools
from . import database as db
from .periods import Range, Periodicity, datelike


@dataclass(repr=False)
class Column:
    header: str
    range_: Range | None = None
    references: list[int] = field(default_factory=lambda: [1])
    values_cache: dict[Range,np.ndarray] = field(default_factory=dict)
    children: list[Column] = field(default_factory=list)
    siblings: list[Column] = field(default_factory=list)
    sibling_ordinal: int = 0
    #calculated: set[Range] = field(default_factory=set)

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

    @property
    def calculated(self) -> bool:
        self.range_ in self.values_cache

    #def set_calculated(self) -> None:
        #self.calculated.add(self.range_)
        #for col in self.siblings:
            #col.calculated = True

    def get_bounds(self) -> tuple[datetime.date,datetime.date,Periodicity] | None:
        return None

    def check_child_ranges(self) -> None:
        pass

    def __copy__(self):
        print(f'copy {str(id(self))[-4:]}[{self.references[0]}]')
        self.references[0] += 1
        copied = copy.copy(super())
        print(f'copy {str(id(self))[-4:]}[{self.references[0]}] to {str(id(copied))[-4:]}')
        copied.siblings = []
        copied.siblings.extend(self.siblings)
        copied.siblings.append(self)
        self.siblings.append(copied)
        copied.children = []
        for child in self.children:
            print(f'copy child {str(id(child))[-4:]}')
            copied.children.append(copy.copy(child))
        return copied


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
            selected = list(range(len(stack))[current.slice_])
            if current.filters:
                filtered = []
                for filter_ in current.filters:
                    pattern = re.compile(filter_)
                    for i in selected:
                        if bool(pattern.match(stack[i].header)):
                            filtered.append(i)
                selected = filtered
            if current.inverse_selection:
                selected = list(set(range(len(stack))) - set(selected))
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
                to_delete.update(set(range(len(stack))) - set(selected))
            for i in sorted(to_delete, reverse=True):
                stack[i].references[0] -= 1
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

    @property
    def values(self) -> Rows:
        return Rows(self, True)

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

def pc(c, n):
    print(f'{' '*n}{str(id(c))[-4:]}-{str(id(c.values_cache))[-4:]}-{c.references}')
def ps(stack):   
    for c in stack:
        pc(c, 4)
        print((' ' * 8) + 'childs')
        for c2 in c.children:
            pc(c2, 8)
        print((' ' * 8) + 'sibs')
        for c2 in c.siblings:
            pc(c2, 8)

class Rows:
    def __init__(self, word: Word, values_only: bool = False):
        self.word = word
        self.values_only = values_only

    def __getitem__(self, key: slice | int | str | Range) -> pd.DataFrame:
        stack = []
        self.word.build_stack(stack)
        min_, max_, per = None, None, None
        working = stack[:]
        flat = []
        print('stack')
        ps(stack)
        while working:
            col = working.pop()
            bounds = col.get_bounds()
            if bounds:
                min_ = bounds[0] if not min_ \
                    else max(min_, bounds[0])
                max_ = bounds[1] if not max_ \
                    else min(max_, bounds[1])
                per = bounds[2]
            flat.append(col)
            working.extend(col.children)
        flat.reverse()
        print('flat')
        ps(flat)
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
                    print(f'reassign {str(id(child.values_cache))[-4:]} to {str(id(col.values_cache))[-4:]}') 
                    col.values = child.values
            if not col.range_: # unused col
                continue
            if col.values is None:
                rows, cols = len(col.range_), len(col.siblings)
                shape = (rows, cols) if cols else rows
                col.values = np.empty(shape, order='F')
                print(f'assign {str(id(col.values_cache))[-4:]}') 
                print(col.values_cache)
            if not col.calculated:
                col.calculate()
                #col.set_calculated()
            col.children.clear()
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

    def __call__(self, *values: list[float]) -> Word:
        self.constants = values
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for constant in self.constants:
            stack.append(NullaryColumn(self.name,
                generator=partial(self.generate, constant)))

class ConstantRange(Constant):
    def __call__(self, n: int) -> Word:
        return super().__call__(*range(n))

@dataclass(kw_only=True)
class PandasColumn(Column):
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
        self.values_cache[self.range_][:] = np.nan
        idx, idx_child = self.pandas_range \
                            .resample_indicies(self.range_, self.round_)
        self.values_cache[self.range_][idx,:] = self.pandas.values[idx_child,:]

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
        siblings = []
        values_cache = {}
        for i, (header, col) in enumerate(self.pandas.items()):
            siblings.append(PandasColumn(header,
                        round_=self.round_,
                        pandas=self.pandas,
                        values_cache=values_cache,
                        siblings=siblings,
                        sibling_ordinal=i))
        stack.extend(siblings)

@dataclass(kw_only=True)
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
            stack.append(SavedColumn(md.col_header + md.row_header, (0, 1), md=md))


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
            this_stack = stack[:]
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

@dataclass(kw_only=True)
class ResampleColumn(Column):
    round_: bool

    def check_child_ranges(self) -> None:
        pass

    def calculate(self) -> None:
        self.values[:] = np.nan
        idx, idx_child = self.children[0].range_ \
                            .resample_indicies(self.range_, self.round_)
        self.values[idx] = self.children[0].values[idx_child]

class Resample(Word):
    def __call__(self, round_: bool = False):
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        children = stack[:]
        stack.clear()
        for col in children:
            stack.append(ResampleColumn(col.header, 
                        round_=self.round_, children=[col]))

@dataclass(kw_only=True)
class PeriodicityColumn(ResampleColumn):
    periodicity: Periodicity

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        for col in self.children:
            col.set_range(range_.change_periodicity(self.periodicity))

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
        children = stack[:]
        stack.clear()
        for col in children:
            stack.append(PeriodicityColumn(col.header, round_=self.round_,
                               periodicity=self.periodicity, children=[col]))

@dataclass(kw_only=True)
class FillColumn(Column):
    value: float

    def calculate(self) -> None:
        self.values[:] = self.children[0].values
        self.values[np.isnan(self.values)] = self.value

class Fill(Word):
    def __call__(self, value: float) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        children = stack[:]
        stack.clear()
        for col in children:
            stack.append(FillColumn(col.header, value=self.value, children=[col]))

@dataclass(kw_only=True)
class FFillColumn(Column):
    lookback: int
    leave_end: bool

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        child_range = self.range_.expand(int(-self.lookback))
        for col in self.children:
            col.set_range(child_range)

    def calculate(self) -> None:
        x = self.children[0].values
        idx = np.where(~np.isnan(x),np.arange(len(x)),0)
        np.maximum.accumulate(idx,out=idx)
        self.values[:] = x[idx][self.lookback:]
        if self.leave_end:
            if len(x) > idx.max() + 1:
                at_end = len(x) - idx.max() - 1
                self.values[-at_end:] = np.nan

class FFill(Word):
    def __call__(self, lookback: int = 10, leave_end: bool = True) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        children = stack[:]
        stack.clear()
        for col in children:
            stack.append(FFillColumn(col.header, lookback=self.lookback,
                                     leave_end=self.leave_end, children=[col]))

@dataclass(kw_only=True)
class JoinColumn(Column):
    date: datelike

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        cutover_index = range_.periodicity[self.date].start
        if range_.stop < cutover_index:
            self.children.pop()
            self.children[0].set_range(range_)
        elif range_.start >= cutover_index:
            self.children.pop(-2)
            self.children[0].set_range(range_)
        else:
            r_first = copy.copy(range_)
            r_first.stop = cutover_index
            self.children[0].set_range(r_first)
            r_second = copy.copy(range_)
            r_second.start = cutover_index
            self.children[1].set_range(r_second)

    def calculate(self) -> None:
        i = 0
        for child in self.children:
            v = child.values
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
                                children=[stack.pop(-2),stack.pop()]))

@dataclass(kw_only=True)
class ReductionColumn(Column):
    operation: Callable[[np.ndarray], np.ndarray]

    def calculate(self) -> None:
        operands = np.concatenate([col.values for col in self.children]) \
                        .reshape((len(self.range_),len(self.children)),order='F')
        # axis=1
        self.values = self.operation(operands)

class Reduction(Word):
    def __init__(self, name: str, operation: Callable[[np.ndarray, np.ndarray], None]):
        self.operation = operation
        super().__init__(name, slice(-2, None))

    def operate(self, stack: list[Column]) -> None:
        children = [*stack]
        stack.clear()
        stack.append(ReductionColumn(children[0].header,
                    operation=self.operation,
                    children=children))

@dataclass(kw_only=True)
class RollingColumn(Column):
    operation: Callable[[np.ndarray, int], np.ndarray]
    window: int

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        child_range = self.range_.expand(int(-self.window * 1.25))
        for col in self.children:
            col.set_range(child_range)

    def calculate(self) -> None:
        data = self.children[0].values
        print(data)
        print(self.values)
        lookback = int(self.window * 1.25)
        mask = ~np.isnan(data)
        idx = np.where(mask)[0]
        start = np.where(idx >= lookback)[0][0]
        self.values[(idx - lookback)[start:]] = self.operation(data[mask], self.window)[start:]
        idx_nan = np.where(~mask)[0]
        nan_start = np.where(idx_nan >= lookback)[0][0]
        self.values[(idx_nan - lookback)[nan_start:]] = np.nan

class Rolling(Word):
    def __init__(self, name: str, operation: np.ufunc):
        self.operation = operation
        super().__init__(name, slice_=slice(-1,None))

    def __call__(self, window: int = 2) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        children = [*stack]
        stack.clear()
        for col in children:
            stack.append(RollingColumn(col.header, operation=self.operation,
                        children=[col], window=self.window))

'''
@dataclass(kw_only=True)
class RollingUFuncColumn(Column):
    operation: np.ufunc
    window: int

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        child_range = self.range_.expand(int(-self.window * 1.25))
        for col in self.children:
            col.set_range(child_range)

    def calculate(self) -> None:
        lookback = int(self.window * 1.25)
        expanded = self.children[0].values
        mask = ~np.isnan(expanded)
        no_nans = expanded[mask]
        indexes = np.add.accumulate(np.where(mask))[0]
        if no_nans.shape[-1] - self.window < 0:
            no_nans = np.hstack([np.full(self.window - len(no_nans),np.nan),no_nans])
        shape = no_nans.shape[:-1] + (no_nans.shape[-1] - self.window + 1, self.window)
        strides = no_nans.strides + (no_nans.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(no_nans, shape=shape, strides=strides)
        td = np.full((expanded.shape[0],self.window), np.nan)
        td[indexes[-windows.shape[0]:],:] = windows
        self.operation(td[lookback:,::-1], out=self.values, axis=1)
        '''

class NonReduceFunction(Word):
    def __init__(self, name: str,
                    operation: Callable[[np.ndarray,np.ndarray],None],
                    slice_ = slice(-1, None), ascending: bool = True):
        self.ascending = ascending
        self.operation = operation
        super().__init__(name, slice_)

    def __call__(self, ascending: bool = True) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        values_cache = {}
        siblings = []
        children = [*stack]
        stack.clear()
        for i, col in enumerate(children):
            siblings.append(NonReduceFunctionColumn(col.header,
                        operation=self.operation,
                        children=children,
                        values_cache=values_cache,
                        siblings=siblings,
                        sibling_ordinal=i,
                        ascending=self.ascending))
        stack.extend(siblings)

@dataclass(kw_only=True)
class NonReduceFunctionColumn(Column):
    operation: Callable[[np.ndarray,np.ndarray],None]
    ascending: bool = True

    def calculate(self) -> None:
        if len(self.children) > 1:
            operands = np.concatenate([col.values for col in self.children]) \
                        .reshape((len(self.range_),len(self.children)),order='F')
        else:
            operands = self.children[0].values[:,np.newaxis]
        if not self.ascending:
            operands = operands[::-1]
        self.operation(operands, out=self.values_cache[self.range_])


def rank(inputs: np.ndarray, out: np.ndarray) -> None:
    out[:] = inputs.argsort(axis=1).argsort(axis=1)


def ewma_calc(window: float, data: np.ndarray, out: np.ndarray) -> None:
    idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
    nans = np.where(~np.isnan(data),1,np.nan)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        out[:] = np.nan
    ewma_out = tools.ewma(data, 2 / (window + 1.0)) * nans
    out[:] = self_out[idx]

class EWMA(Word):
    def __init__(self, name: str):
        super().__init__(name, slice(-1,None))

    def __call__(self, window: float) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        stack.append(NonReduceFunctionColumn(self.name,
                        operation=partial(ewma_calc, window=self.window),
                        fill_nans=self.fill_nans, children=[stack.pop()]))

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

def expanding_mean(x: np.ndarray, out: np.ndarray) -> None:
    nan_mask: npt.NDArray[np.bool] = np.isnan(x)
    cumsum: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x))
    count: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x / x))
    out[:] = np.where(nan_mask, np.nan, cumsum) / count

def expanding_var(x: np.ndarray, out: np.ndarray) -> None:
    nan_mask: npt.NDArray[np.bool] = np.isnan(x)
    cumsum: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x))
    cumsumOfSquares: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x * x))
    count: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x / x))
    out[:] = (cumsumOfSquares - cumsum * cumsum / count) / (count - 1)

def expanding_std(x: np.ndarray, out: np.ndarray) -> None:
    nan_mask = np.isnan(x)
    cumsum = np.add.accumulate(np.where(nan_mask, 0, x))
    cumsumOfSquares = np.add.accumulate(np.where(nan_mask, 0, x * x))
    count = np.add.accumulate(np.where(nan_mask, 0, x / x))
    out[:] = np.sqrt((cumsumOfSquares - cumsum * cumsum / count) / (count - 1))

def rolling_covariance(window_size, data, out):
    # Calculate the rolling mean for each column, ignoring NaN values
    mean1 = np.convolve(np.nan_to_num(data[:, 0]), np.ones(window_size), 'valid') / window_size
    mean2 = np.convolve(np.nan_to_num(data[:, 1]), np.ones(window_size), 'valid') / window_size

    # Calculate the covariance for each window, ignoring NaN values
    cov = np.convolve(np.nan_to_num((data[:, 0] - mean1) * (data[:, 1] - mean2)),
                      np.ones(window_size), 'valid') / window_size

    # Adjust for NaN values in the original data
    valid_counts = np.convolve(~np.isnan(data[:, 0]) & ~np.isnan(data[:, 1]),
                               np.ones(window_size), 'valid')
    cov /= valid_counts
    return cov

'''
class RollingCovariance(Word):
    def __init__(self, name):
        super().__init__(name, slice(-2,None))

    def __call__(self, window: int = 10) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        col0, col1 = stack.pop(), stack.pop()
        stack.append(RollingFunctionColumn(col0.header + '-' + col1.header,
                    operation=rolling_covariance, children=[col0,col1],
                    window=self.window))
                    '''


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
vocab['filter'] = (cat, 'Removes non-select columns', lambda name: Word(name, discard_excluded=True))
vocab['drop'] = (cat, 'Removes selected columns',
             lambda name: Word(name, inverse_selection=True, discard_excluded=True, copy_selected=False))
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
vocab['join'] = (cat, 'Joins two columns at _date_',Join)
vocab['zero_first'] = (cat, 'Changes first value to zero',lambda name: NonReduceFunction(name, zero_first_op))
vocab['zero_to_na'] = (cat, 'Changes zeros to nans',lambda name: NonReduceFunction(name, zero_to_na_op))
vocab['resample'] = (cat, 'Adapts periodicity to match range with optional rounding _round_',Resample)
vocab['per'] = (cat, 'Changes periodicity to _periodicity_', SetPeriodicity)

_funcs = [('add',  'Addition', partial(bn.nansum, axis=1), bn.move_sum),
           ]
for code, desc, red, roll in _funcs:
    vocab[code] = ('Row-wise Reduction', desc,
                    lambda name, func=red: Reduction(name, func))
    vocab[f'w{code}'] = ('Rolling Window', desc,
                     lambda name, func=roll: Rolling(name, func))

'''
    ('sub',  'Subtraction', np.subtract), ('mul',  'Multiplication', np.multiply),
    ('div',  'Division', np.divide), ('pow',  'Power', np.power),
    ('mod',  'Modulo', np.mod), ('eq',  'Equals', np.equal),
    ('ne',  'Not equals', np.not_equal), ('gt',  'Greater than', np.greater),
    ('lt',  'Less than', np.less), ('ge',  'Greater than or equal to', np.greater_equal),
    ('le',  'Less than or equal to', np.less_equal), ('land',  'Logical and', np.logical_and),
    ('lor', 'Logical or', np.logical_or), ('lxor',  'Logical xor', np.logical_xor)]

    vocab[f's{code}'] = ('Scan/Accumulation', desc,
                     lambda name, func=func.accumulate: ScanOperator(name, func))
_funcs = [('mean',  'Arithmetic average', np.nanmean, expanding_mean),
            ('std',  'Standard deviation', np.nanmean, expanding_mean),
            ('var',  'Variance', np.nanmean, expanding_mean)]
for code, desc, func, scan_func in _funcs:
    vocab[code] = ('Row-wise Reduction', desc,
                    lambda name, func=func: ReductionOperator(name, func))
    vocab[f'w{code}'] = ('Rolling Window', desc,
                     lambda name, func=func: RollingUFuncOperator(name, func))
    vocab[f's{code}'] = ('Scan/Accumulation', desc,
                     lambda name, func=scan_func: ScanOperator(name, func))

vocab[f'wcov'] = ('Rolling Window', desc, RollingCovariance)

'''
cat = 'Other functions'
vocab['neg'] = (cat, 'Additive inverse',lambda name: NonReduceFunction(name,np.negative))
vocab['inv'] = (cat, 'Multiplicative inverse',lambda name: NonReduceFunction(name,np.reciprocal))
vocab['abs'] = (cat, 'Absolute value',lambda name: NonReduceFunction(name,np.abs))
vocab['sqrt'] = (cat, 'Square root',lambda name: NonReduceFunction(name,np.sqrt))
vocab['log'] = (cat, 'Natural log',lambda name: NonReduceFunction(name,np.log))
vocab['exp'] = (cat, 'Exponential',lambda name: NonReduceFunction(name,np.exp))
vocab['lnot'] = (cat, 'Logical not',lambda name: NonReduceFunction(name, np.logical_not))
vocab['expm1'] = (cat, 'Exponential minus one',lambda name: NonReduceFunction(name,np.expm1))
vocab['log1p'] = (cat, 'Natural log of increment',lambda name: NonReduceFunction(name,np.log1p))
vocab['sign'] = (cat, 'Sign',lambda name: NonReduceFunction(name, np.sign))
vocab['rank'] = (cat, 'Row-wise rank',lambda name: NonReduceFunction(name, rank, slice_=slice(None)))
vocab['ewma'] = (cat, 'Exponentially weighted moving average with half-life of _window_', EWMA)



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

