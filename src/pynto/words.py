from __future__ import annotations

import copy
import datetime
import sys
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import partial
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .vocabulary import Vocabulary
from . import database as db
from .base import (
    Word,
    Column,
    ResampleMethod,
    resample,
    GroupShared,
    SavedColumn,
    Quotation,
)
from .periods import Range, Periodicity, datelike

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

type Generator = Callable[[Range, np.ndarray], None]


# Generator words
class GeneratorWord(Word):
    def __init__(self, name: str, vocab: Vocabulary, generator: Generator):
        self.generator = generator
        super().__init__(name, vocab, slice(-1, 0))

    def operate(self, stack: list[Column]) -> None:
        stack.append(
            GeneratorColumn(header=self.name, generator=self.generator, name=self.name)
        )


@dataclass(kw_only=True, eq=False)
class GeneratorColumn(Column):
    generator: Generator

    def operate(self) -> None:
        self.generator(self.range_, self.values)


class RandomNormal(GeneratorWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = np.random.randn(len(range_))[:, None]

    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, self.generate)


class Timestamp(GeneratorWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = range_.to_index().view('int').astype(np.float64)[:, None]

    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, self.generate)


class PeriodOrdinal(GeneratorWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = np.arange(range_.start, range_.stop).astype(np.float64)[:, None]

    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, self.generate)


class Daycount(GeneratorWord):
    @staticmethod
    def generate(range_: Range, values: np.ndarray):
        values[:] = np.array(range_.day_counts(), dtype=np.float64)[:, None]

    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, self.generate)


class Constant(GeneratorWord):
    @staticmethod
    def generate(constant: float, range_: Range, values: np.ndarray):
        values[:] = constant

    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, self.generate)

    def __call__(self, *values: list[float]) -> Word:
        self.constants = values
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        for constant in self.constants:
            header = f'c{str(constant).replace("-", "_").replace(".", "_")}'
            stack.append(
                GeneratorColumn(
                    header=header,
                    generator=partial(self.generate, constant),
                    name=self.name,
                )
            )


class ConstantRange(Constant):
    def __call__(self, n: int) -> Word:
        return super().__call__(*range(n))


@dataclass(kw_only=True, eq=False)
class PandasColumn(Column):
    pandas: pd.DataFrame
    round_: bool
    pandas_range: Range | None = None

    def __post_init__(self):
        super().__post_init__()
        self.pandas_range = Range.from_index(self.pandas.index)

    def get_bounds(self) -> tuple[datetime.date, datetime.date, Periodicity] | None:
        return (
            self.pandas_range[0][-1],
            self.pandas_range.expand()[-1][-1],
            self.pandas_range.periodicity,
        )

    def operate(self) -> None:
        self.values[:] = np.nan
        idx, idx_input = self.pandas_range.resample_indicies(self.range_, self.round_)
        self.group_values[idx, :] = self.pandas.values[idx_input, :]


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
        group = GroupShared()
        for header in self.pandas.columns:
            stack.append(
                PandasColumn(
                    header, round_=self.round_, pandas=self.pandas, group=group
                )
            )


class Saved(Word):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, slice(-1, 0))

    def __call__(self, key: str) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        for md in db.get_client().get_metadata(self.key):
            header = md.col_header
            if md.row_header:
                header += '$' + md.row_header
            stack.append(SavedColumn(header, md=md))


# Quotation / combinator words


class Combinator(Word):
    def __init__(
        self,
        name: str,
        vocab: Vocabulary,
        slice_: slice = slice(None),
        num_quotations: int = 1,
    ):
        self.num_quotations = num_quotations
        super().__init__(name, vocab, slice_)

    def get_quotations(self) -> list[Word]:
        quotations = []
        current = self.prev
        needed = self.num_quotations
        while needed != 0 and current and isinstance(current, Quotation):
            quotations.append(current.quoted)
            needed -= 1
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
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, num_quotations=2)

    def __call__(self, count: int = 1) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if len(stack) >= self.count:
            self.get_quotations()[0].build_stack(stack)
        else:
            self.get_quotations()[1].build_stack(stack)


class IfHeaders(Combinator):
    def __call__(self, predicate: Callable[[list[str]], bool]) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        if self.predicate([col.header for col in stack]):
            self.get_quotations()[0].build_stack(stack)


class IfHeadersElse(Combinator):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, num_quotations=2)

    def __call__(self, predicate: Callable[[list[str]], bool]) -> Word:
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
        assert len(stack) % self.every == 0, (
            f'Stack length {len(stack)} not evenly divisible by every {self.every}'
        )
        copied = stack[:]
        stack.clear()
        quoted = self.get_quotations()[0]
        for t in zip(*[iter(copied)] * self.every):
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
            for i, col in enumerate(stack):
                if header == col.header:
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            quoted.build_stack(filtered_stack)
            new_stack += filtered_stack
            to_del.sort(reverse=True)
            for i in to_del:
                del stack[i]
        del stack[:]
        stack += new_stack


class Cleave(Combinator):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, num_quotations=-1)

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
    def __init__(self, vocab: Vocabulary, bound: list[Column]):
        self.bound = bound
        self.operate_count = 0
        super().__init__('bound', vocab)

    def operate(self, stack: list[Column]) -> None:
        if self.operate_count == 0:
            stack.extend(self.bound)
        else:
            bound_copy = [copy.copy(c) for c in self.bound]
            stack.extend(bound_copy)
        self.operate_count += 1


class Partial(Quotation, Combinator):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, slice(-1, None))

    def __add__(self, other: Word, copy_addend: bool = True) -> Word:
        return Word.__add__(self, other, copy_addend)

    def operate(self, stack: list[Column]) -> None:
        self.quoted = BoundWord(self.vocab, stack[:]) + self.get_quotations()[0]
        stack.clear()


class Compose(Quotation, Combinator):
    def __init__(self, name: str, vocab: Vocabulary):
        Combinator.__init__(self, name, vocab, num_quotations=2)

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
    def __init__(self, name: str, vocab: Vocabulary, method: ResampleMethod):
        self.method = method
        super().__init__(name, vocab)

    def operate(self, stack: list[Column]) -> None:
        for col in stack:
            col.resampler = self.method


@dataclass(kw_only=True, eq=False)
class PeriodicityColumn(Column):
    periodicity: Periodicity

    def set_input_range(self, inputs: list[Column]) -> None:
        inputs[0].set_range(self.range_.change_periodicity(self.periodicity))

    def operate(self) -> None:
        self.values[:] = np.nan
        resample(
            self.range_,
            self.values,
            self.inputs[0].range_,
            self.inputs[0].values,
            self.inputs[0].resampler,
        )


class SetPeriodicity(Word):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, slice(-1, None))

    def __call__(self, periodicity: str | Periodicity) -> Word:
        if isinstance(periodicity, str):
            self.periodicity = Periodicity[periodicity]
        else:
            self.periodicity = periodicity
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for i in inputs:
            col = PeriodicityColumn(i.header, periodicity=self.periodicity)
            col.shared.open_inputs.append(i)
            stack.append(col)


@dataclass(kw_only=True, eq=False)
class StartColumn(Column):
    start: datelike | int
    offset: int | None = None

    def set_input_range(self, inputs: list[Column]) -> None:
        if isinstance(self.start, int):
            self.offset = self.start
        else:
            self.offset = (
                self.range_.periodicity[self.start].ordinal - self.range_.start
            )
        input_range = Range(
            self.range_.start + self.offset, self.range_.stop, self.range_.periodicity
        )
        for col in inputs:
            col.set_range(input_range)

    def operate(self) -> None:
        if self.offset <= 0:
            self.values[:] = self.input_values[-self.offset :]
        else:
            self.values[self.offset :] = self.input_values
            self.values[: self.offset] = np.nan


class SetStart(Word):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, slice(-1, None))

    def __call__(self, start: datelike | int) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        group = GroupShared()
        for i in inputs:
            sib = StartColumn(
                i.header,
                start=self.start,
                group=group,
            )
            sib.shared.open_inputs.append(i)
            stack.append(sib)


@dataclass(kw_only=True, eq=False)
class FillFirstColumn(Column):
    lookback: int

    def set_input_range(self, inputs: list[Column]) -> None:
        self.lookback = -abs(self.lookback)
        input_range = self.range_.expand(self.lookback)
        for col in inputs:
            col.set_range(input_range)

    def operate(self) -> None:
        data = self.input_values
        out = self.values
        first_part = data[: -self.lookback + 1]
        row_ok = ~np.isnan(first_part).any(axis=1)
        good_idxs = np.flatnonzero(row_ok)
        if good_idxs.size > 0:
            out[0] = data[good_idxs[-1]]
        else:
            out[0] = np.nan
        out[1:] = data[-self.lookback + 1 :]


class FillFirst(Word):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, slice(-1, None))

    def __call__(self, lookback: int = 5) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        group = GroupShared()
        for i in inputs:
            sib = FillFirstColumn(
                i.header,
                lookback=self.lookback,
                group=group,
            )
            sib.shared.open_inputs.append(i)
            stack.append(sib)


@dataclass(kw_only=True, eq=False)
class FillColumn(Column):
    value: float

    def operate(self) -> None:
        values = self.values
        values[:] = self.input_values
        values[np.isnan(values)] = self.value


class Fill(Word):
    def __call__(self, value: float) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = stack[:]
        stack.clear()
        for i in inputs:
            col = FillColumn(i.header, value=self.value)
            col.shared.open_inputs.append(i)
            stack.append(col)


@dataclass(kw_only=True, eq=False)
class FFillColumn(Column):
    lookback: int
    leave_end: bool

    def set_input_range(self, inputs: list[Column]) -> None:
        input_range = self.range_.expand(int(-self.lookback))
        for col in inputs:
            col.set_range(input_range)

    def operate(self) -> None:
        x = self.input_values.ravel()
        idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        self.values[:] = x[idx][self.lookback :][:, None]
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
        for i in inputs:
            col = FFillColumn(
                i.header,
                lookback=self.lookback,
                leave_end=self.leave_end,
            )
            col.shared.open_inputs.append(i)
            stack.append(col)


@dataclass(kw_only=True, eq=False)
class JoinColumn(Column):
    date: datelike

    def set_input_range(self, inputs: list[Column]) -> None:
        r = self.range_
        cutover_index = r.periodicity[self.date].ordinal
        if r.stop <= cutover_index:
            inputs[0].set_range(r)
        elif r.start >= cutover_index:
            inputs[-1].set_range(r)
        else:
            inputs[0].set_range(Range(r.start, cutover_index, r.periodicity))
            inputs[1].set_range(Range(cutover_index, r.stop, r.periodicity))

    def operate(self) -> None:
        i = 0
        for input_ in self.inputs:
            if input_.range_ is not None:
                v = input_.values
                self.values[i : i + len(v)] = v
                i += len(v)


class Join(Word):
    def __init__(self, name: str, vocab: Vocabulary):
        super().__init__(name, vocab, slice_=slice(-2, None))

    def __call__(self, date: datelike) -> Word:
        self.date = date
        return super().__call__()

    def operate(self, stack: list[Column]) -> None:
        col = JoinColumn(
            stack[-2].header,
            date=self.date,
        )
        col.shared.open_inputs.append(stack.pop(-2))
        col.shared.open_inputs.append(stack.pop())
        stack.append(col)


@dataclass(kw_only=True, eq=False)
class ReductionColumn(Column):
    operation: Callable[[np.ndarray], np.ndarray]
    ignore_nans: bool

    def operate(self) -> None:
        inputs = self.input_values
        values = self.operation(inputs)[:, None]
        if self.ignore_nans:
            values[np.all(np.isnan(inputs), axis=1)] = np.nan
        else:
            values[np.any(np.isnan(inputs), axis=1)] = np.nan
        self.values[:] = values


class Reduction(Word):
    def __init__(
        self,
        name: str,
        vocab: Vocabulary,
        operation: Callable[[np.ndarray, np.ndarray], None],
    ):
        self.operation = operation
        super().__init__(name, vocab, slice(-2, None))

    def __call__(self, ignore_nans: bool = False) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        if inputs:
            col = ReductionColumn(
                inputs[0].header,
                operation=self.operation,
                ignore_nans=self.ignore_nans,
                name=self.name,
            )
            col.shared.open_inputs.extend(inputs)
            stack.append(col)


@dataclass(kw_only=True, eq=False)
class RollingColumn(Column):
    operation: Callable[[np.ndarray, int], np.ndarray]
    window: int
    reduction: bool = False
    lookback: int | None = None

    def set_input_range(self, inputs: list[Column]) -> None:
        self.lookback = max(5, int(self.window * 1.25))
        input_range = self.range_.expand(-self.lookback)
        for col in inputs:
            col.set_range(input_range)

    def operate(self) -> None:
        data = self.input_values
        mask = ~np.any(np.isnan(data), axis=1)
        idx = np.where(mask)[0]
        out = self.group_values
        if np.any(idx >= self.lookback):
            start = np.where(idx >= self.lookback)[0][0]
            out[(idx - self.lookback)[start:]] = self.operation(
                data[mask], self.window
            )[start:]
            if not np.all(mask):
                idx = np.where(~mask)[0]
                if np.any(idx >= self.lookback):
                    start = np.where(idx >= self.lookback)[0][0]
                    out[(idx - self.lookback)[start:]] = np.nan
        else:
            out[:] = np.nan


class Rolling(Word):
    def __init__(
        self, name: str, vocab: Vocabulary, operation: np.ufunc, slice_=slice(-1, None)
    ):
        self.operation = operation
        super().__init__(name, vocab, slice_=slice_)

    def __call__(self, window: int = 2) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        group = GroupShared()
        for i in inputs:
            sib = RollingColumn(
                i.header,
                operation=self.operation,
                window=self.window,
                group=group,
            )
            sib.shared.open_inputs.append(i)
            stack.append(sib)


class RollingReduction(Rolling):
    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        col = RollingColumn(
            inputs[0].header,
            window=self.window,
            operation=self.operation,
            reduction=True,
        )
        col.shared.open_inputs.extend(inputs)
        stack.append(col)


class GroupOperator(Word):
    def __init__(
        self,
        name: str,
        vocab: Vocabulary,
        operation: Callable[[np.ndarray, np.ndarray], None],
        slice_=slice(-1, None),
        ascending: bool = True,
        allow_group_drops: bool = True,
    ):
        self.ascending = ascending
        self.operation = operation
        self.allow_group_drops = allow_group_drops
        super().__init__(name, vocab, slice_)

    def operate(self, stack: list[Column]) -> None:
        inputs = [*stack]
        stack.clear()
        group = GroupShared()
        group.allow_drops = self.allow_group_drops
        for i in inputs:
            sib = GroupOperatorColumn(
                i.header,
                operation=self.operation,
                ascending=self.ascending,
                group=group,
                name=self.name,
            )
            sib.shared.open_inputs.append(i)
            stack.append(sib)


@dataclass(kw_only=True, eq=False)
class GroupOperatorColumn(Column):
    operation: Callable[[np.ndarray, np.ndarray], None]
    ascending: bool = True

    def operate(self) -> None:
        data = self.input_values
        if not self.ascending:
            data = data[::-1]
            self.operation(data, out=self.group_values[::-1])
        else:
            self.operation(data, out=self.group_values)


class Roll(Word):
    def operate(self, stack: list[Column]) -> None:
        stack.insert(0, stack.pop())


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
        if len(stack) == 0:
            return
        assert len(stack) % self.parts == 0, f'Stack not divisible by {self.parts}'
        count = len(stack) // self.parts
        last = 0
        lists = []
        for i in range(len(stack) + 1):
            if i % count == 0 and i != 0:
                lists.append(stack[i - count : i])
                last = i
        del stack[:last]
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
        assert start >= 0, f"Insufficient columns for hset('{','.join(self.headers)}')"
        for i in range(start, len(stack)):
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
    def __call__(self, header_func: Callable[[str], str]) -> Word:
        return super().__call__(locals())

    def operate(self, stack: list[Column]) -> None:
        for col in stack:
            col.header = self.header_func(col.header)


class HeaderAlphabetize(Word):
    @staticmethod
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

    def operate(self, stack: list[Column]) -> None:
        gen = self.alphabet_generator()
        for col in stack:
            col.header = next(gen)
