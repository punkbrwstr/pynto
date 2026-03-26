from __future__ import annotations

import logging
import re
import sys
from functools import partial
from typing import Any, Callable

import bottleneck as bn  # type: ignore[import-untyped]
import numpy as np

from .base import Word, ResampleMethod
from .operations import (
    dec_op,
    expanding_diff,
    expanding_lag,
    expanding_mean,
    expanding_ret,
    expanding_std,
    expanding_var,
    inc_op,
    rank,
    rolling_cor,
    rolling_cov,
    rolling_diff,
    rolling_ewma,
    rolling_ews,
    rolling_ewv,
    rolling_lag,
    rolling_ret,
    rolling_zsc,
    zero_first_op,
    zero_to_na_op,
)
from .words import (
    Call,
    Cleave,
    Compose,
    Constant,
    ConstantRange,
    Daycount,
    FFill,
    Fill,
    FillFirst,
    FromPandas,
    GroupOperator,
    HMap,
    HSort,
    HeaderAlphabetize,
    HeaderApply,
    HeaderFormat,
    HeaderReplace,
    HeaderSet,
    HeaderSetAll,
    IfExists,
    IfExistsElse,
    IfHeaders,
    IfHeadersElse,
    Interleave,
    Join,
    Map,
    Partial,
    PeriodOrdinal,
    Quotation,
    RandomNormal,
    Reduction,
    Repeat,
    Resample,
    Reverse,
    Roll,
    Rolling,
    RollingReduction,
    Saved,
    SetPeriodicity,
    SetStart,
    Timestamp,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)


def expanding_wrapper(
    func: Callable[..., Any],
) -> Callable[[np.ndarray, np.ndarray], None]:
    def wrapper(
        a: np.ndarray, out: np.ndarray, func: Callable[..., Any] = func
    ) -> None:
        mask = np.all(~np.isnan(a), axis=1)
        if not np.all(~mask):
            out[mask] = func(a[mask])
        out[~mask] = np.nan

    return wrapper


type Entry = tuple[str, str, Callable[..., Word]]


class Vocabulary(dict[str, Entry]):
    def resolve(self, name: str) -> Word | None:
        if re.match(r'c\d[_\d]*', name) is not None:
            return Constant('c', self)(float(name[1:].replace('_', '.')))
        elif re.match(r'c_\d[_\d]*', name) is not None:
            return Constant('c', self)(-float(name[2:].replace('_', '.')))
        elif re.match(r'r\d+', name) is not None:
            return Constant('c', self)(*range(int(name[1:])))
        elif name in self:
            return self[name][-1](name, self)
        else:
            return None

    def __getattr__(self, name: str) -> Any:
        if name == 'resolve':
            return self.__getattribute__('resolve')
        word = self.resolve(name)
        if word is not None:
            return word
        else:
            return self.__getattribute__(name)


vocab: Vocabulary = Vocabulary()


cat = 'Column Creation'
vocab['c'] = (cat, 'Pushes constant columns for each of _values_', Constant)
vocab['r'] = (
    cat,
    'Pushes constant columns for each whole number from 0 to _n_ - 1',
    ConstantRange,
)
vocab['nan'] = (
    cat,
    'Pushes a constant nan-valued column',
    lambda name, vocab: Constant(name, vocab)(np.nan),
)
vocab['randn'] = (
    cat,
    'Pushes a column with values from a random normal distribution',
    RandomNormal,
)
vocab['timestamp'] = (
    cat,
    'Pushes a column with the timestamp of the end of the period',
    Timestamp,
)
vocab['period_ordinal'] = (
    cat,
    'Pushes a column with the period ordinal',
    PeriodOrdinal,
)
vocab['day_count'] = (
    cat,
    'Pushes a column with the number of days in the period',
    Daycount,
)
vocab['load'] = (cat, 'Pushes columns saved to internal DB as _key_', Saved)
vocab['from_pandas'] = (
    cat,
    'Pushes columns from Pandas DataFrame or Series _pandas_',
    FromPandas,
)

cat = 'Stack Manipulation'
vocab['id'] = (cat, 'Identity/no-op', lambda name, vocab: Word(name, vocab))
vocab['pull'] = (
    cat,
    'Brings selected columns to the top',
    lambda name, vocab: Word(name, vocab, raise_on_empty=True),
)
vocab['dup'] = (
    cat,
    'Duplicates columns',
    lambda name, vocab: Word(
        name, vocab, slice_=slice(-1, None), copy_selected=True, raise_on_empty=True
    ),
)
vocab['keep'] = (
    cat,
    'Removes non-selected columns',
    lambda name, vocab: Word(name, vocab, discard_excluded=True),
)
vocab['drop'] = (
    cat,
    'Removes selected columns',
    lambda name, vocab: Word(
        name,
        vocab,
        inverse_selection=True,
        discard_excluded=True,
        copy_selected=False,
        slice_=slice(-1, None),
    ),
)
vocab['nip'] = (
    cat,
    'Removes non-selected columns, defaulting selection to top',
    lambda name, vocab: Word(
        name, vocab, discard_excluded=True, slice_=slice(-1, None)
    ),
)
vocab['roll'] = (cat, 'Permutes selected columns', Roll)
vocab['swap'] = (
    cat,
    'Swaps top and bottom selected columns',
    lambda name, vocab: Roll(name, vocab, slice(-2, None)),
)
vocab['rev'] = (cat, 'Reverses the order of selected columns', Reverse)
vocab['interleave'] = (
    cat,
    'Divides columns in _parts_ groups and interleaves the groups',
    Interleave,
)
vocab['hsort'] = (cat, 'Sorts columns by header', HSort)

vocab['q'] = (
    'Quotation',
    'Wraps the following words until *p* as a quotation, or '
    + 'wraps _quoted_ expression as a quotation',
    Quotation,
)

cat = 'Header manipulation'
vocab['hset'] = (cat, 'Set headers to _*headers_ ', HeaderSet)
vocab['hsetall'] = (
    cat,
    'Set headers to _*headers_ repeating, if necessary',
    HeaderSetAll,
)
vocab['hformat'] = (cat, 'Apply _format_spec_ to headers', HeaderFormat)
vocab['hreplace'] = (cat, 'Replace _old_ with _new_ in headers', HeaderReplace)
vocab['happly'] = (cat, 'Apply _header_func_ to headers_', HeaderApply)
vocab['halpha'] = (cat, 'Set headers to alphabetical values', HeaderAlphabetize)

cat = 'Combinators'
vocab['call'] = (cat, 'Applies quotation', Call)
vocab['map'] = (cat, 'Applies quotation in groups of _every_', Map)
vocab['repeat'] = (cat, 'Applies quotation _times_ times', Repeat)
vocab['hmap'] = (
    cat,
    'Applies quotation to stacks created grouping columns by header',
    HMap,
)
vocab['cleave'] = (cat, 'Applies all preceding quotations', Cleave)
vocab['ifexists'] = (
    cat,
    'Applies quotation if stack has at least _count_ columns',
    IfExists,
)
vocab['ifexistselse'] = (
    cat,
    'Applies top quotation if stack has at least _count_ columns'
    + ', otherwise applies second quotation',
    IfExistsElse,
)
vocab['ifheaders'] = (
    cat,
    'Applies top quotation if list of column headers fulfills _predicate_',
    IfHeaders,
)
vocab['ifheaderselse'] = (
    cat,
    'Applies quotation if list of column headers fulfills _predicate_'
    + ', otherwise applies second quotation',
    IfHeadersElse,
)
vocab['partial'] = (cat, 'Pushes stack columns to the front of quotation', Partial)
vocab['compose'] = (cat, 'Combines quotations', Compose)

cat = 'Data cleanup'
vocab['fill'] = (cat, 'Fills nans with _value_ ', Fill)
vocab['ffill'] = (
    cat,
    'Fills nans with previous values, looking back _lookback_ before range '
    + 'and leaving trailing nans unless not _leave_end_',
    FFill,
)
vocab['fillfirst'] = (
    cat,
    'Fills first row with previous non-nan value, looking back _lookback_ '
    + ' before range',
    FillFirst,
)
vocab['sync'] = (
    cat,
    'Align available data by setting all values to NaN when any values is NaN',
    lambda name, vocab: GroupOperator(
        name, vocab, expanding_wrapper(lambda a: a), slice_=slice(None)
    ),
)
vocab['join'] = (cat, 'Joins two columns at _date_', Join)
vocab['zero_first'] = (
    cat,
    'Changes first value to zero',
    lambda name, vocab: GroupOperator(name, vocab, zero_first_op),
)
vocab['zero_to_na'] = (
    cat,
    'Changes zeros to nans',
    lambda name, vocab: GroupOperator(name, vocab, zero_to_na_op),
)
cat = 'Resample methods'
vocab['resample_sum'] = (
    cat,
    'Sets periodicity resampling method to sum',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.SUM),
)
vocab['resample_last'] = (
    cat,
    'Sets periodicity resampling method to last',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.LAST),
)
vocab['resample_avg'] = (
    cat,
    'Sets periodicity resampling method to avg',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.AVG),
)
vocab['resample_lastnofill'] = (
    cat,
    'Sets periodicity resampling method to last with no fill',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.LAST_NOFILL),
)
vocab['resample_first'] = (
    cat,
    'Sets periodicity resampling method to first',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.FIRST),
)
vocab['resample_firstnofill'] = (
    cat,
    'Sets periodicity resampling method to first',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.FIRST_NOFILL),
)
vocab['resample_min'] = (
    cat,
    'Sets periodicity resampling method to min',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.MIN),
)
vocab['resample_max'] = (
    cat,
    'Sets periodicity resampling method to max',
    lambda name, vocab: Resample(name, vocab, ResampleMethod.MAX),
)
vocab['set_periodicity'] = (
    cat,
    'Changes column periodicity to _periodicity_, then resamples',
    SetPeriodicity,
)
vocab['set_start'] = (cat, 'Changes period start to _start_, then resamples', SetStart)


_funcs = [
    (
        'add',
        'Addition',
        partial(bn.nansum, axis=1),
        partial(bn.move_sum, axis=0),
        np.add.accumulate,
    ),
    (
        'sub',
        'Subtraction',
        partial(np.subtract.reduce, axis=1),
        None,
        np.subtract.accumulate,
    ),
    (
        'mul',
        'Multiplication',
        partial(np.multiply.reduce, axis=1),
        None,
        np.multiply.accumulate,
    ),
    ('div', 'Division', partial(np.divide.reduce, axis=1), None, None),
    ('pow', 'Power', partial(np.power.reduce, axis=1), None, None),
    ('mod', 'Modulo', partial(np.mod.reduce, axis=1), None, None),
    (
        'avg',
        'Arithmetic average',
        partial(bn.nanmean, axis=1),
        partial(bn.move_mean, axis=0, min_count=2),
        expanding_mean,
    ),
    (
        'std',
        'Standard deviation',
        partial(bn.nanstd, axis=1),
        partial(bn.move_std, axis=0),
        expanding_std,
    ),
    (
        'var',
        'Variance',
        partial(bn.nanvar, axis=1),
        partial(bn.move_var, axis=0),
        expanding_var,
    ),
    (
        'min',
        'Minimum',
        partial(bn.nanmax, axis=1),
        partial(bn.move_min, axis=0),
        np.minimum.accumulate,
    ),
    (
        'max',
        'Maximum',
        partial(bn.nanmin, axis=1),
        partial(bn.move_max, axis=0),
        np.maximum.accumulate,
    ),
    (
        'med',
        'Median',
        partial(bn.nanmedian, axis=1),
        partial(bn.move_median, axis=1),
        None,
    ),
    ('lag', 'Lag', None, rolling_lag, expanding_lag),
    ('dif', 'Lagged difference', None, rolling_diff, expanding_diff),
    ('ret', 'Lagged return', None, rolling_ret, expanding_ret),
    ('ewm', 'Exponentially-weighted moving average', None, rolling_ewma, None),
    ('ewv', 'Exponentially-weighted variance', None, rolling_ewv, None),
    ('ews', 'Exponentially-weighted standard deviation', None, rolling_ews, None),
    ('zsc', 'Z-score', None, rolling_zsc, None),
]
for code, desc, red, roll, scan in _funcs:
    if red:
        vocab[code] = (
            'Row-wise Reduction',
            desc,
            lambda name, vocab, func=red: Reduction(name, vocab, func),
        )
        vocab[f'n{code}'] = (
            'Row-wise Reduction Ignoring NaNs',
            desc,
            lambda name, vocab, func=red: Reduction(name, vocab, func)(True),
        )
    if roll:
        vocab[f'r{code}'] = (
            'Rolling Window',
            desc,
            lambda name, vocab, func=roll: Rolling(name, vocab, func),
        )
    if scan:
        vocab[f'c{code}'] = (
            'Cumulative',
            desc,
            lambda name, vocab, func=scan: GroupOperator(
                name, vocab, expanding_wrapper(func)
            ),
        )
        vocab[f'rc{code}'] = (
            'Reverse Cumulative',
            desc,
            lambda name, vocab, func=scan: GroupOperator(
                name, vocab, expanding_wrapper(func), ascending=False
            ),
        )
vocab['rcov'] = (
    'Rolling Window',
    'Covariance',
    lambda name, vocab: RollingReduction(
        name, vocab, rolling_cov, slice_=slice(-2, None)
    ),
)
vocab['rcor'] = (
    'Rolling Window',
    'Correlation',
    lambda name, vocab: RollingReduction(
        name, vocab, rolling_cor, slice_=slice(-2, None)
    ),
)
vocab['ewm_mean'] = vocab.pop('rewm')
vocab['ewm_var'] = vocab.pop('rewv')
vocab['ewm_std'] = vocab.pop('rews')


cat = 'One-for-one functions'
vocab['neg'] = (
    cat,
    'Additive inverse',
    lambda name, vocab: GroupOperator(name, vocab, np.negative),
)
vocab['inv'] = (
    cat,
    'Multiplicative inverse',
    lambda name, vocab: GroupOperator(name, vocab, np.reciprocal),
)
vocab['abs'] = (
    cat,
    'Absolute value',
    lambda name, vocab: GroupOperator(name, vocab, np.abs),
)
vocab['sqrt'] = (
    cat,
    'Square root',
    lambda name, vocab: GroupOperator(name, vocab, np.sqrt),
)
vocab['log'] = (
    cat,
    'Natural log',
    lambda name, vocab: GroupOperator(name, vocab, np.log),
)
vocab['exp'] = (
    cat,
    'Exponential',
    lambda name, vocab: GroupOperator(name, vocab, np.exp),
)
vocab['lnot'] = (
    cat,
    'Logical not',
    lambda name, vocab: GroupOperator(name, vocab, np.logical_not),
)
vocab['expm1'] = (
    cat,
    'Exponential minus one',
    lambda name, vocab: GroupOperator(name, vocab, np.expm1),
)
vocab['log1p'] = (
    cat,
    'Natural log of increment',
    lambda name, vocab: GroupOperator(name, vocab, np.log1p),
)
vocab['sign'] = (cat, 'Sign', lambda name, vocab: GroupOperator(name, vocab, np.sign))
vocab['rank'] = (
    cat,
    'Row-wise rank',
    lambda name, vocab: GroupOperator(
        name, vocab, rank, slice_=slice(None), allow_group_drops=False
    ),
)
vocab['inc'] = (
    cat,
    'Increment',
    lambda name, vocab: GroupOperator(name, vocab, inc_op),
)
vocab['dec'] = (
    cat,
    'Decrement',
    lambda name, vocab: GroupOperator(name, vocab, dec_op),
)
