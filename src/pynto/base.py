from __future__ import annotations

import copy
import datetime
import logging
import re
import sys
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .vocabulary import Vocabulary

import numpy as np
import pandas as pd

from . import database as db
from .periods import Range, Period, Periodicity, datelike

DEBUG = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)


def toggle_debug():
    global DEBUG
    DEBUG = not DEBUG
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def debug_col_repr(c: Column, n: int) -> str:
    return (
        f'{" " * n}c{c.id_}/s{str(id(c.shared))[-4:]}'
        + f' g{str(id(c.group))[-4:]}({",".join([f"s{str(id(s))[-4:]}" for s in c.group.members])})'
        + f'-{c.name} "{c.header}" [{c.range_}]'
    )


def debug_stack(stack: list[Column], title: str = '', offset: int = 4) -> str:
    if stack:
        repr_ = (' ' * offset) + title
        offset += 4
        for c in stack:
            repr_ += debug_col_repr(c, offset) + '\n'
            repr_ += debug_stack(c.shared.open_inputs, 'open inputs', offset + 4)
            key = c.range_
            repr_ += debug_stack(
                (c.group.closed_inputs.get(key) if key is not None else None) or [],
                'group inputs',
                offset + 4,
            )
        return repr_ + '\n'
    else:
        return ''


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
    MIN = 5
    MAX = 6
    FIRST = 7
    FIRST_NOFILL = 8


@dataclass(eq=False)
class Shared:
    ordinal: int = 0
    open_inputs: list[Column] = field(default_factory=list)
    columns: set[Column] = field(default_factory=set)


@dataclass
class GroupShared:
    allow_drops: bool = True
    members: dict[Shared, None] = field(default_factory=dict)
    outputs: dict[tuple[Range, ResampleMethod], None | np.ndarray] = field(
        default_factory=dict
    )
    closed_inputs: dict[Range, list[Column]] = field(default_factory=dict)


@dataclass(eq=False)
class Column:
    header: str = ''
    range_: Range | None = None
    shared: Shared = field(default_factory=Shared)
    group: GroupShared = field(default_factory=GroupShared)
    resampler: ResampleMethod = ResampleMethod.LAST
    id_: int = field(default_factory=_IDs.get_next)
    name: str | None = None
    _is_copy: bool = False

    def __post_init__(self):
        self.shared.columns.add(self)
        if not self._is_copy:
            self.group.members[self.shared] = None
            if not self.name:
                self.name = self.__class__.__name__

    def operate(self) -> None:
        pass

    def get_bounds(self) -> tuple[datetime.date, datetime.date, Periodicity] | None:
        return None

    def compute(self) -> None:
        assert self.range_ is not None
        self.group.outputs[(self.range_, self.resampler)] = np.empty(
            (len(self.range_), len(self.group.members)), order='F'
        )
        self.operate()
        if self.range_ in self.group.closed_inputs:
            del self.group.closed_inputs[self.range_]

    @property
    def inputs(self) -> list[Column]:
        key = self.range_
        return (
            self.group.closed_inputs.get(key) if key is not None else None
        ) or self.shared.open_inputs

    @property
    def computed(self) -> bool:
        return (self.range_, self.resampler) in self.group.outputs

    @property
    def group_values(self) -> np.ndarray:
        assert self.range_ is not None
        result = self.group.outputs[(self.range_, self.resampler)]
        assert result is not None
        return result

    @property
    def values(self) -> np.ndarray:
        c = self.shared.ordinal
        return self.group_values[:, c : c + 1]

    def set_range(self, range_: Range) -> None:
        self.range_ = range_
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'   c{self.id_} -> {range_}')
        if range_ not in self.group.closed_inputs:
            inputs = []
            for i, shared in enumerate(self.group.members.keys()):
                inputs.extend([copy.copy(in_) for in_ in shared.open_inputs])
                for col in shared.columns:
                    col.shared.ordinal = i
            self.set_input_range(inputs)
            self.group.closed_inputs[(range_)] = inputs

    def set_input_range(self, inputs: list[Column]) -> None:
        assert self.range_ is not None
        for col in inputs:
            col.set_range(self.range_)

    @property
    def input_values(self) -> np.ndarray:
        assert self.range_ is not None, 'Cannot get input values before range set.'
        inputs = self.group.closed_inputs[self.range_]
        assert len(inputs) > 0, 'Empty inputs.'
        assert len(set([c.range_ for c in inputs])) == 1, (
            'All input ranges must be the same to use input_values funtion'
        )
        assert inputs[0].range_ is not None
        return np.concatenate([c.values for c in inputs]).reshape(  # type: ignore[no-any-return]
            (len(inputs[0].range_), len(inputs)), order='F'
        )

    def drop(self):
        self.shared.columns.remove(self)
        if self.group.allow_drops and len(self.shared.columns) == 0:
            del self.group.members[self.shared]

    def __copy__(self):
        id_ = _IDs.get_next()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'*Copying c{self.id_} -> c{id_}')
        return replace(self, id_=id_, _is_copy=True)

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return NotImplemented
        return self.id_ == other.id_

    def __del__(self):
        pass  # logger.debug('destroying '+ debug_col_repr(self, 4))


@dataclass(kw_only=True, eq=False)
class SavedColumn(Column):
    md: db.Metadata

    def get_bounds(self) -> tuple[datetime.date, datetime.date, Periodicity] | None:
        return (self.md[0][-1], self.md.expand()[-1][-1], self.md.periodicity)


class _ColsAccessor:
    def __init__(self, word: Word):
        self._word = word

    def __getitem__(self, key: int | slice | str | list[str]) -> Word:
        if isinstance(key, int):
            if key == -1:
                self._word.slice_ = slice(key, None)
            else:
                self._word.slice_ = slice(key, key + 1)
        elif isinstance(key, slice):
            self._word.slice_ = key
        elif isinstance(key, str):
            self._word.filters = [key]
        elif isinstance(key, list):
            self._word.filters = key
        else:
            raise IndexError('Invalid column indexer')
        return self._word


class Word:
    def __init__(
        self,
        name: str,
        vocab: Vocabulary,
        slice_: slice = slice(None),
        copy_selected: bool = False,
        discard_excluded: bool = False,
        inverse_selection: bool = False,
        raise_on_empty: bool = False,
    ):
        self.name = name
        self.vocab = vocab
        self.slice_ = slice_
        self.copy_selected = copy_selected
        self.discard_excluded = discard_excluded
        self.inverse_selection = inverse_selection
        self.raise_on_empty = raise_on_empty
        self.filters: list[str] | None = None
        self.next_: Word | None = None
        self.prev: Word | None = None
        self.called: bool = False

    def operate(self, stack: list[Column]) -> None:
        if self.raise_on_empty and len(stack) == 0:
            raise IndexError(f'Empty stack for {self.name}')
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Word:
        self.called = True
        locals_dict = args[0] if args else None
        if locals_dict:
            for key, value in locals_dict.items():
                if key not in ['__class__', 'self']:
                    setattr(self, key, value)
        return self

    def __add__(self, addend: Word | int | float, copy_addend: bool = True) -> Word:
        if isinstance(addend, (int, float)):
            from .words import Constant

            addend = Constant('c', self.vocab)(addend)
        this = self.copy_expression()
        if copy_addend:
            addend_tail = addend.copy_expression()
        else:
            addend_tail = addend
        addend_head = addend_tail._head()
        if this is not None:
            this.next_ = addend_head
            addend_head.prev = this
        return addend_tail

    def __radd__(self, addend: int | float) -> Word:
        if isinstance(addend, (int, float)):
            from .words import Constant

            const = Constant('c', self.vocab)(addend)
            return const.__add__(self)
        return NotImplemented

    def __invert__(self) -> Word:
        if (
            isinstance(self, Quotation)
            and hasattr(self, 'quoted')
            and self.prev is None
        ):
            return self.quoted.copy_expression()
        q = Quotation('q', self.vocab)
        q.quoted = self.copy_expression()
        return q

    @property
    def cols(self) -> _ColsAccessor:
        return _ColsAccessor(self)

    @property
    def copy(self) -> Word:
        self.copy_selected = True
        return self

    @property
    def discard(self) -> Word:
        self.discard_excluded = True
        return self

    def build_stack(self, stack: list[Column] | None = None) -> list[Column]:
        if stack is None:
            stack = []
            prefix = 0
        else:
            prefix = 8
        current: Word | None = self._head()
        logger.debug(f'{" " * prefix}Evaluating words')
        while current is not None:
            if not current.called:  # need to set defaults (if any)
                current()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{" " * (prefix + 3)}{current} slice {current.slice_}')
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'{" " * (prefix + 6)}selected=[{",".join([f"c{stack[i].id_}" for i in selected])}]'
                )
            current_stack = []
            to_delete = set()
            already_selected = set()
            for i in selected:
                if current.copy_selected or i in already_selected:
                    c = copy.copy(stack[i])
                    current_stack.append(c)
                else:
                    to_delete.add(i)
                    current_stack.append(stack[i])
                already_selected.add(i)

            if current.discard_excluded:
                excluded = set(range(len(stack))) - set(selected)
                for i in excluded:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f'{" " * (prefix + 6)}dropping {debug_col_repr(stack[i], 4)}'
                        )
                    stack[i].drop()  # no longer needed as siblings
                to_delete.update(excluded)
            for i in sorted(to_delete, reverse=True):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f'{" " * (prefix + 6)}remove from stack {debug_col_repr(stack[i], 4)}'
                    )
                del stack[i]
            current.operate(current_stack)
            stack.extend(current_stack)
            current = current.next_
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'{" " * (prefix + 3)}stack=[{",".join([f"c{c.id_}" for c in stack])}]'
                )
        return stack

    @property
    def stack(self) -> list[Column]:
        return self.build_stack()

    @property
    def columns(self) -> list[str]:
        return [col.header for col in self.stack]

    @property
    def last(self) -> Any:
        return Evaluator(self)[-1:].iloc[-1]

    @property
    def all(self) -> Any:
        return Evaluator(self)[:]

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
        call_word = self.vocab.resolve('call')
        assert call_word is not None
        call_word.slice_ = slice(-1, 0)
        return (~self) + call_word

    def __str__(self) -> str:
        ignore = (
            'prev next_ closed called generator slice_ filters copy_selected'
            + ' discard_excluded inverse_selection operation __doc__'
            + ' name allow_group_drops ascending raise_on_empty vocab'
        )
        str_args = []
        for k, v in self.__dict__.items():
            if k != self and k not in ignore.split():
                if isinstance(v, str):
                    str_args.append(f"{k}='{v[:2000]}'")
                    if len(v) > 2000:
                        str_args[-1] += '...'
                else:
                    str_args.append(f'{k}={str(v)}')
        s = self.name
        if str_args:
            s += f'({", ".join(str_args)})'
        if self.filters:
            s += f'.cols[{self.filters}]'
        else:
            s += f'.cols[{self.slice_.start or ""}:{self.slice_.stop or ""}:{self.slice_.step or ""}]'
        return s

    def __repr__(self) -> str:
        s = ''
        current = self
        while True:
            s = current.__str__() + s
            if current.prev is None:
                break
            else:
                s = '\n    + ' + s
                current = current.prev
        s = 'pt.' + s
        return s


class Quotation(Word):
    def __init__(self, name: str, vocab: Vocabulary, slice_: slice = slice(-1, 0)):
        super().__init__(name, vocab, slice_)

    def __call__(self, quoted: Word | None = None) -> Word:
        if quoted is None:
            return self
        else:
            this = self.copy_expression()
            this.quoted = quoted
            return this


def resample(
    to_range: Range,
    to_values: np.ndarray,
    from_range: Range,
    from_values: np.ndarray,
    method: ResampleMethod,
) -> None:
    idx, idx_input = from_range.resample_indicies(to_range, False)
    match method:
        case ResampleMethod.LAST:
            x = from_values.ravel()
            where = np.where(~np.isnan(x), np.arange(len(x)), 0)
            np.maximum.accumulate(where, out=where)
            from_values[:] = x[where][:, None]
            to_values[idx] = from_values[idx_input]
        case ResampleMethod.LAST_NOFILL:
            to_values[idx] = from_values[idx_input]
        case ResampleMethod.FIRST_NOFILL:
            idx_input = [0] + [x + 1 for x in idx_input[:-1]]
            to_values[idx] = from_values[idx_input]
        case ResampleMethod.FIRST:
            x = from_values.ravel()[::-1]
            where = np.where(~np.isnan(x), np.arange(len(x)), 0)
            np.maximum.accumulate(where, out=where)
            from_values[:] = x[where][::-1][:, None]
            idx_input = [0] + [x + 1 for x in idx_input[:-1]]
            to_values[idx] = from_values[idx_input]
        case ResampleMethod.SUM:
            sums = np.nancumsum(from_values)
            to_values[idx[0]] = from_values[idx_input[0]]
            to_values[idx[1:]] = (sums[idx_input[1:]] - sums[idx_input[:-1]])[:, None]
        case ResampleMethod.AVG:
            sums = np.nancumsum(from_values)
            counts = np.nancumsum(from_values / from_values)
            to_values[idx[0]] = from_values[idx_input[0]]
            to_values[idx[1:]] = (sums[idx_input[1:]] - sums[idx_input[:-1]])[:, None]
            to_values[idx[1:]] /= (counts[idx_input[1:]] - counts[idx_input[:-1]])[
                :, None
            ]
        case ResampleMethod.MIN:
            x = from_values.ravel()[::-1]
            where = np.where(~np.isnan(x), np.arange(len(x)), 0)
            np.maximum.accumulate(where, out=where)
            idx_input = [0] + [x + 1 for x in idx_input[:-1]]
            to_values[idx] = np.minimum.reduceat(x[where][::-1], idx_input)[:, None]
        case ResampleMethod.MAX:
            x = from_values.ravel()[::-1]
            where = np.where(~np.isnan(x), np.arange(len(x)), 0)
            np.maximum.accumulate(where, out=where)
            idx_input = [0] + [x + 1 for x in idx_input[:-1]]
            to_values[idx] = np.maximum.reduceat(x[where][::-1], idx_input)[:, None]


class Evaluator:
    def __init__(self, word: Word, values_only: bool = False):
        self.word = word
        self.values_only = values_only

    def __getitem__(self, key: slice | int | str | Range) -> Any:
        _IDs.reset()
        stack = self.word.build_stack()
        min_, max_, per = None, None, None
        working = stack[:]
        while working:
            col = working.pop()
            bounds = col.get_bounds()
            if bounds:
                min_ = bounds[0] if min_ is None else min(min_, bounds[0])
                max_ = bounds[1] if max_ is None else max(max_, bounds[1])
                per = bounds[2]
            working.extend(col.shared.open_inputs)
        if isinstance(key, Range):
            range_ = key
        elif isinstance(key, datelike) or isinstance(key, int):
            p = per if per else Periodicity.B
            _per = p[key]
            assert isinstance(_per, Period)
            range_ = _per.to_range()
        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            p = Periodicity[step] if step else per if per else Periodicity.B
            if isinstance(start, int) and max_ is not None:
                if start < 0:
                    start = p[max_] + start
                else:
                    assert min_ is not None
                    start = p[min_] + start
            if isinstance(stop, int) and max_ is not None:
                if stop < 0:
                    stop = p[max_] + stop
                else:
                    assert min_ is not None
                    stop = p[min_] + stop
            range_ = p[start or min_ : stop or max_]  # type: ignore[misc]
        else:
            raise TypeError('Unsupported indexer')
        logger.debug('Setting ranges')
        for col in stack:
            try:
                col.set_range(range_)
            except Exception as e:
                logger.error(f'Error setting range for {col}')
                raise e
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Stack\n{debug_stack(stack)}')
        working = stack[:]
        flat: deque[Column] = deque()
        while working:
            col = working.pop()
            flat.append(col)
            working.extend(col.inputs)
        flat.reverse()
        saveds = [col for col in flat if isinstance(col, SavedColumn)]
        if saveds:
            p = db.get_client().connection.pipeline()
            offsets: list[int] = []
            needed_saveds: list[SavedColumn] = []
            resample_ranges: list[Range | None] = []
            for col in saveds:
                if col.range_ and not col.computed:
                    col.compute()  # initialize array
                    if col.range_.periodicity != col.md.periodicity:
                        r = col.range_.change_periodicity(col.md.periodicity)
                        resample_ranges.append(r)
                    else:
                        r = col.range_
                        resample_ranges.append(None)
                    offsets.append(db.get_client()._req(col.md, r.start, r.stop, p))
                    needed_saveds.append(col)
            for col, offset, bytes_, r_ in zip(
                needed_saveds, offsets, p.execute(), resample_ranges
            ):
                col.values[:] = np.nan
                if len(bytes_) > 0:
                    data = np.frombuffer(bytes_, col.md.type_.dtype)
                    if r_ is not None:
                        assert col.range_ is not None
                        from_values = np.full((len(r_), 1), np.nan, order='F')
                        from_values[offset : offset + len(data), 0] = data
                        resample(col.range_, col.values, r_, from_values, col.resampler)
                    else:
                        col.values[offset : offset + len(data), 0] = data
        logger.debug('Processing flat')
        while flat:
            col = flat.popleft()
            if col.computed or not col.range_:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('    skip' + debug_col_repr(col, 1))
                continue
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('    calc' + debug_col_repr(col, 1))
                col.compute()
            except Exception as e:
                raise ValueError(f'Unable to operate {col!r}') from e
            # col.inputs = []
        if not stack:
            return None
        values = np.concatenate([col.values for col in stack]).reshape(
            (len(range_), len(stack)), order='F'
        )
        if self.values_only:
            return values
        return pd.DataFrame(
            values, columns=[col.header for col in stack], index=range_.to_index()
        )
