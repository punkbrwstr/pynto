import re
import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from lima.time import get_index, get_date

Column = namedtuple('Column',['header', 'trace', 'row_function'])
_Range = namedtuple('_Range',['start', 'end', 'periodicity'])


def get_range(start, end, periodicity):
    return _Range(get_index(periodicity, start),
                    get_index(periodicity, end), periodicity)

def range_size(r):
    return r.end - r.start + 1

def to_date_range(r):
    return pd.date_range(get_date(r.periodicity,r.start),
                get_date(r.periodicity,r.end), freq=r.periodicity)

def _e():
    funcs = []
    def exp(*args,**kwargs):
        nonlocal funcs
        if 'quoted_stack' in kwargs:
            stack = kwargs['quoted_stack']
            for func in funcs:
                #print(f'calling quoted {func.__name__}')
                func(stack)
        elif len(args) == 1 and isinstance(args[0],list):
            # quotation puts itself into stack
            args[0].append(Column('quotation','quotation', exp))
        elif len(args) == 1 and isinstance(args[0],pd.core.indexes.datetimes.DatetimeIndex):
            # evaluating quotation itself with combinator
            return exp
        else:
            for arg in args:
                if isinstance(arg,(int, float, complex)):
                    def lit(stack,arg=arg):
                        def lit_col(date_range):
                            return np.full(range_size(date_range),arg)
                        stack.append(Column('constant',str(arg),lit_col))
                    arg = lit
                #print(f'adding func {arg.__name__}')
                funcs.append(arg)
            #print(f'funcs: [{",".join([func.__name__ for func in funcs])}]')
        trace = 'trace' in kwargs and kwargs['trace']
        if 'eval' in kwargs:
            date_range = kwargs['eval']
            stack = []
            for func in funcs:
                #print(f'calling {func.__name__}')
                func(stack)
            cols = []
            headers = []
            for s in stack:
                rows = s.row_function(date_range)
                headers.append(s.trace if trace else s.header)
                if not isinstance(rows,(np.ndarray, np.generic)):
                  rows = np.full(range_size(date_range),str(type(rows)))
                cols.append(rows)
            mat = np.vstack(cols).T
            return pd.DataFrame(mat, columns=headers, index=to_date_range(date_range))
        return exp
    return exp

def e(*args,**kwargs):
    return _e()(*args,**kwargs)

# Stack manipulation
def dup(stack):
    stack.append(stack[-1])

def roll(stack):
    stack.insert(0,stack.pop())

def swap(stack):
    stack.insert(-1,stack.pop())

def drop(stack):
    stack.pop()

def clear(stack):
    stack.clear()

def interleave(count=None, split_into=2):
    def interleave(stack, count=count, split_into=split_into):
        count = len(stack) // split_into if count is None else count
        place,last = 0,0
        lists = []
        for i in range(len(stack)+1):
            if i % count == 0 and i != 0:
               lists.append(stack[i-count:i])
               last = i
        del(stack[:last])
        stack += [val for tup in zip(*lists) for val in tup]
    return interleave

def pull(start,end=None,clear=False):
    def pull(stack,start=start,end=end):
        end = -start - 1 if end is None else -end
        start = len(stack) if start == 0 else -start
        pulled = stack[end:start]
        if clear:
            del(stack[:])
        else:
            del(stack[end:start])
        stack += pulled
    return pull

def filter(start,end=None,clear=False):
    return pull(start,end,clear=True)

def hpull(*args,clear=False):
    def hpull(stack):
        filtered_stack = []
        for header in args:
            to_del = []
            for i,col in enumerate(stack):
                #print(f'checking {col.header}')
                if not re.match(header,col.header) is None:
                    #print(f'matched {col.header}')
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        if clear:
            del(stack[:])
        stack += filtered_stack
    return hpull

def hfilter(*args):
    return hpull(*args,clear=True)

def compose(*args):
    def composed(stack):
        for func in args:
            func(stack)
    return composed

# Operators
def _binary_operator(name, operation):
    def binary_operator(stack):
        col2 = stack.pop()
        col1 = stack.pop()
        def binary_operator_col(date_range):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                values1 = col1.row_function(date_range)
                values2 = col2.row_function(date_range)
                return np.where(np.logical_or(np.isnan(values1), np.isnan(values2)),
                                    np.nan, operation(values1,values2))
        stack.append(Column(col1.header,f'e({col1.trace}),e({col2.trace}),{name}',binary_operator_col))
    return binary_operator

def _unary_operator(name, operation):
    def unary_operator(stack):
        col = stack.pop()
        def unary_operator_col(date_range):
            return operation(col.row_function(date_range))
        stack.append(Column(col.header,f'{col.trace},{name},',unary_operator_col))
    return unary_operator

add = _binary_operator('add', np.add)
sub = _binary_operator('sub', np.subtract)
mul = _binary_operator('mul', np.multiply)
div = _binary_operator('div', np.divide)
mod = _binary_operator('mod', np.mod)
exp = _binary_operator('exp', np.power)
eq = _binary_operator('eq', lambda x,y: np.where(np.equal(x,y),1,0 ))
ne = _binary_operator('ne', lambda x,y: np.where(np.not_equal(x,y),1,0))
ge = _binary_operator('ge', lambda x,y: np.where(np.greater_equal(x,y),1,0))
gt = _binary_operator('gt', lambda x,y: np.greater(x,y))
le = _binary_operator('le', lambda x,y: np.where(np.less_equal(x,y),1,0))
lt = _binary_operator('lt', lambda x,y: np.where(np.less(x,y),1,0))


absv = _unary_operator('absv', np.abs)
sqrt = _unary_operator('sqrt', np.sqrt)
zeroToNa = _unary_operator('zeroToNa', lambda x: np.where(np.equal(x,0),np.nan,x))

# Windows
def rolling(window=2, exclude_nans=True, lookback_multiplier=2):
    def rolling(stack):
        col = stack.pop()
        def rolling_col(date_range,lookback_multiplier=lookback_multiplier):
            if not exclude_nans:
                lookback_multiplier = 1
            lookback = (window - 1) * lookback_multiplier
            expanded_range = get_range(date_range.start - lookback,
                                    date_range.end,date_range.periodicity)
            expanded = col.row_function(expanded_range)
            mask = ~np.isnan(expanded) if exclude_nans else np.full(expanded.shape,True)
            no_nans = expanded[mask]
            # Indexes of no_nan values in expanded
            indexes = np.add.accumulate(np.where(mask))[0]
            if no_nans.shape[-1] - window < 0:
                #warnings.warn('Insufficient non-nan values in lookback.')
                no_nans = np.hstack([np.full(window - len(no_nans),np.nan),no_nans])
            shape = no_nans.shape[:-1] + (no_nans.shape[-1] - window + 1, window)
            strides = no_nans.strides + (no_nans.strides[-1],)
            windows = np.lib.stride_tricks.as_strided(no_nans, shape=shape, strides=strides)
            td = np.full((expanded.shape[0],window), np.nan)
            td[indexes[-windows.shape[0]:],:] = windows
            return td[lookback:]
        stack.append(Column(col.header,f'{col.trace},rolling({window})',rolling_col))
    return rolling


def expanding(stack):
    col = stack.pop()
    def expanding_col(date_range):
        return col.row_function(date_range)
    stack.append(Column(col.header, f'{col.trace},expanding', expanding_col))

def crossing(stack):
    cols = stack[:]
    stack.clear()
    headers = ','.join([col.header for col in cols])
    def crossing_col(date_range):
        return np.column_stack([col.row_function(date_range) for col in cols])
    stack.append(Column('cross', f'{headers},crossing',crossing_col))

def _window_operator(name, twod_operation, oned_operation):
    def window_operator(stack):
        col = stack.pop()
        def window_operator_col(date_range):
            #nan_on_last_nan = col.row_function.__name__ != 'crossing_col'
            values = col.row_function(date_range)
            if len(values.shape) == 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    return np.where(np.all(np.isnan(values),axis=1), np.nan,
                                twod_operation(values, axis=1))
            else:
                return oned_operation(values)
        stack.append(Column(col.header, f'{col.trace},{name}', window_operator_col))
    return window_operator

def _expanding_mean(x):
    nan_mask = np.isnan(x)
    cumsum = np.add.accumulate(np.where(nan_mask, 0, x))
    count = np.add.accumulate(np.where(nan_mask, 0, x / x))
    return np.where(nan_mask, np.nan, cumsum) / count

def _expanding_var(x):
    nan_mask = np.isnan(x)
    cumsum = np.add.accumulate(np.where(nan_mask, 0, x))
    cumsumOfSquares = np.add.accumulate(np.where(nan_mask, 0, x * x))
    count = np.add.accumulate(np.where(nan_mask, 0, x / x))
    return (cumsumOfSquares - cumsum * cumsum / count) / (count - 1)

def _make_expanding(ufunc):
    def expanding(x):
        mask = np.isnan(x)
        return np.where(mask, np.nan, ufunc.accumulate(np.where(mask,0,x)))
    return expanding

wsum = _window_operator('wsum',np.nansum, _make_expanding(np.add))
wmean = _window_operator('wmean',np.nanmean, _expanding_mean)
wprod = _window_operator('wprod',np.nanprod, _make_expanding(np.multiply))
wvar = _window_operator('wvar', np.nanvar, _expanding_var)
wstd = _window_operator('wstd', np.nanstd, _expanding_var)
wchange = _window_operator('wchange',lambda x, axis: x[:,-1] - x[:,0],
                                        lambda x, axis: x - x[0])
wpct_change = _window_operator('wpct_change',lambda x, axis: x[:,-1] / x[:,0] - 1,
                                        lambda x, axis: x / x[0] - 1)
wlog_change = _window_operator('wlog_change', lambda x, axis: np.log(x[:,-1] / x[:,0]),
                                        lambda x, axis: np.log( x / x[0]))
first = _window_operator('first',lambda x: x[:,0], lambda x: np.full(a.shape,a[0]))
last = _window_operator('last',lambda x: x[:,-1], lambda x: x)


def wlag(number):
    return compose(rolling(number+1),wfirst)

def wzscore():
    return compose(e(std),e(last),e(mean),cleave(3),sub,swap,div)

# Combinators
def call(depth=None,copy=False):
    def call(stack,depth=depth,copy=copy):
        quote = stack.pop()
        depth = len(stack) if depth is None else depth
        if depth != 0:
            this_stack = stack[-depth:]
            if not copy:
                del(stack[-depth:])
        else:
            this_stack = []
        quote.row_function(quoted_stack=this_stack)
        stack.extend(this_stack)
    return call

def keep(depth=None):
    return call(depth,True)

def every(step,copy=False):
    def every(stack):
        quote = stack.pop()
        copied_stack = stack[:]
        if len(copied_stack) % step != 0:
            raise ValueError('Stack not evenly divisible by step')
        if not copy:
            del(stack[:])
        for t in zip(*[iter(copied_stack)]*step):
            this_stack = list(t)
            quote.row_function(quoted_stack=this_stack)
            stack += this_stack
    return every

def cleave(num_quotations, depth=None, copy=False):
    def cleave(stack, depth=depth):
        quotes = stack[-num_quotations:]
        del(stack[-num_quotations:])
        depth = len(stack) if depth is None else depth
        copied_stack = stack[-depth:] if depth != 0 else []
        if not copy and depth != 0:
            del(stack[-depth:])
        #copied_stack = stack[:]
        #del(stack[:])
        for quote in quotes:
            this_stack = copied_stack[:]
            quote.row_function(quoted_stack=this_stack)
            stack += this_stack
    return cleave


def each(start=0, end=None, step=1, copy=False):
    def each(stack,start=start,end=end,step=step, copy=copy):
        quote = stack.pop()
        end = 0 if end is None else -end
        start = len(stack) if start == 0 else -start
        selected = stack[end:start]
        if len(selected) % step != 0:
            raise ValueError('Stack not evenly divisible by step')
        if not copy:
            del(stack[end:start])
        for t in zip(*[iter(selected)]*step):
            this_stack = list(t)
            quote.row_function(quoted_stack=this_stack)
            stack += this_stack
    return each

# Header
def hset(*args):
    def hset(stack):
        start = len(stack) - len(args)
        for i in range(start,len(stack)):
            stack[i] = Column(args[i - start], stack[i].trace, stack[i].row_function)
    return hset

def hformat(format_string):
    def hset(stack):
        col = stack.pop()
        stack.append(Column(format_string.format(col.header),
                        col.trace, col.row_function))
    return hset

def happly(func):
    def happly(stack):
        col = stack.pop()
        stack.append(Column(func(col.header),
                        col.trace, col.row_function))
    return happly

# Data cleaning
def fill(value):
    def fill(stack, value=value):
        col = stack.pop()
        def fill(date_range):
            x = col.row_function(date_range)
            x[np.isnan(x)] = 0.0
            return x
        stack.append(Column(col.header,f'{col.trace},fill',fill))
    return fill

def ffill(stack):
    col = stack.pop()
    def ffill(date_range):
        x = col.row_function(date_range)
        idx = np.where(~np.isnan(x),np.arange(len(x)),0)
        np.maximum.accumulate(idx,out=idx)
        return x[idx]
    stack.append(Column(col.header,f'{col.trace},ffill',ffill))

def join(date):
    def join(stack,date=date):
        col2 = stack.pop()
        col1 = stack.pop()
        def join_col(date_range,date=date):
            date = get_index(date_range.periodicity, date)
            if date_range.end < date:
                return col1.row_function(date_range)
            if date_range.start >= date:
                return col2.row_function(date_range)
            parts = []
            parts.append(col1.row_function(get_range(date_range.start, date - 1, date_range.periodicity)))
            parts.append(col2.row_function(get_range(date, date_range.end, date_range.periodicity)))
            return np.concatenate(parts)
        stack.append(Column('join',f'{col2.trace} {col2.trace} join({date})',join_col))
    return join

