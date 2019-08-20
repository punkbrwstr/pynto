import re
import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from pynto.time import Range, get_index

Column = namedtuple('Column',['header', 'trace', 'rows'])

class _Word(object):

    def __init__(self, func):
        self.func = func

    def __add__(self, other):
        if isinstance(other, list):
            assert len(other) == 1
            other = _Word(other)
        elif not hasattr(other, 'func'):
            other = other()
        self.next = other
        other.prev = self
        return other

    def __le__(self, date_range):
        return self.evaluate([], date_range)

    def evaluate(self, stack, date_range=None):
        current = self
        while hasattr(current, 'prev'):
            current = current.prev
        while True:
            if isinstance(current.func, list):
                stack.append(Column('quot','quot', current.func))
            else:
                if not hasattr(current, 'func'):
                    current = current()
                current.func(stack)
            if not hasattr(current, 'next'):
                break
            current = current.next
        if not date_range is None:
            if isinstance(date_range, tuple):
                date_range = Range(*date_range)
            else:
                date_range = Range(date_range)
            return pd.DataFrame(np.column_stack([col.rows(date_range) for col in stack]),
                        columns=[col.header for col in stack],
                        index=date_range.to_pandas())

class const(_Word):
    '''Constant column of value.'''
    def __init__(self, value=1.):
        def const(stack, value=value):
            def const_col(date_range):
                return np.full(len(date_range), value)
            stack.append(Column('constant', str(value), const_col))
        super().__init__(const)

class const_range(_Word):
    '''Constant column of value.'''
    def __init__(self, n):
        def const_range(stack, n=n):
            for i in range(n):
                def const_col(date_range, i=i):
                    return np.full(len(date_range), i)
                stack.append(Column('constant', str(i), const_col))
        super().__init__(const_range)

# Operators
class _BinaryOperator(_Word):
    def __init__(self, name, operation):
        def binary_operator(stack):
            col2 = stack.pop()
            col1 = stack.pop()
            def binary_operator_col(date_range):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    values1 = col1.rows(date_range)
                    values2 = col2.rows(date_range)
                    return np.where(np.logical_or(np.isnan(values1), np.isnan(values2)),
                                        np.nan, operation(values1,values2))
            stack.append(Column(col1.header,f'e({col1.trace}),e({col2.trace}),{name}',binary_operator_col))
        super().__init__(binary_operator)
            
def add(): return _BinaryOperator('add', np.add)
def sub(): return _BinaryOperator('sub', np.subtract)
def mul(): return _BinaryOperator('mul', np.multiply)
def div(): return _BinaryOperator('div', np.divide)
def mod(): return _BinaryOperator('mod', np.mod)
def exp(): return _BinaryOperator('exp', np.power)
def eq(): return _BinaryOperator('eq', np.equal)
def ne(): return _BinaryOperator('ne', np.not_equal)
def ge(): return _BinaryOperator('ge', np.greater_equal)
def gt(): return _BinaryOperator('gt', np.greater)
def le(): return _BinaryOperator('le', np.less_equal)
def lt(): return _BinaryOperator('lt', np.less)

class _UnaryOperator(_Word):
    def __init__(self, name, operation):
        def unary_operator(stack):
            col = stack.pop()
            def unary_operator_col(date_range):
                return operation(col.rows(date_range))
            stack.append(Column(col.header,f'{col.trace},{name},',unary_operator_col))
        super().__init__(unary_operator)

def absv(): return _Unary_operator('absv', np.abs)
def sqrt(): return _Unary_operator('sqrt', np.sqrt)
def zeroToNa(): return _Unary_operator('zeroToNa', lambda x: np.where(np.equal(x,0),np.nan,x))


# Stack manipulation
class dup(_Word):
    def __init__(self):
        def dup(stack):
            stack.append(stack[-1])
        super().__init__(dup)

class roll(_Word):
    def __init__(self):
        def roll(stack):
            stack.insert(0,stack.pop())
        super().__init__(roll)

class swap(_Word):
    def __init__(self):
        def swap(stack):
            stack.insert(-1,stack.pop())
        super().__init__(swap)

class drop(_Word):
    def __init__(self):
        def drop(stack):
            stack.pop()
        super().__init__(drop)

class clear(_Word):
    def __init__(self):
        def clear(stack):
            stack.clear()
        super().__init__(clear)

class interleave(_Word):
    def __init__(self, count=None, split_into=2):
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
        super().__init__(interleave)

class pull(_Word):
    def __init__(self, start, end=None, clear=False):
        def pull(stack,start=start,end=end):
            end = -start - 1 if end is None else -end
            start = len(stack) if start == 0 else -start
            pulled = stack[end:start]
            if clear:
                del(stack[:])
            else:
                del(stack[end:start])
            stack += pulled
        super().__init__(pull)

class hpull(_Word):
    def __init__(self, *args, clear=False):
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
        super().__init__(hpull)

def hfilter(*args, clear=False):
    return hpull(*args, clear=True)

class emwa(_Word):
    def __init__(self, window, fill_nans=True):
        def ewma(stack):
            col = stack.pop()
            def ewma_col(date_range, window=window, fill_nans=fill_nans):
                alpha = 2 /(window + 1.0)
                alpha_rev = 1-alpha

                data = col.rows(date_range)
                idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
                nans = np.where(idx == -1,np.nan,1) if fill_nans else np.where(np.isnan(data),np.nan,1)
                data = data[~np.isnan(data)]
                if len(data) == 0:
                    return np.full(idx.shape[0],np.nan)
                n = data.shape[0]
        
                pows = alpha_rev**(np.arange(n+1))
        
                scale_arr = 1/pows[:-1]
                offset = data[0]*pows[1:]
                pw0 = alpha*alpha_rev**(n-1)
        
                mult = data*pw0*scale_arr
                cumsums = mult.cumsum()
                out = offset + cumsums*scale_arr[::-1]
                return out[idx] * nans
            stack.append(Column(col.header,f'{col.trace},ewma,',ewma_col))
        super().__init__(ewma)


# Windows
class rolling(_Word):
    def __init__(self, window=2, exclude_nans=True, lookback_multiplier=2):
        def rolling(stack):
            col = stack.pop()
            def rolling_col(date_range,lookback_multiplier=lookback_multiplier):
                if not exclude_nans:
                    lookback_multiplier = 1
                lookback = (window - 1) * lookback_multiplier
                expanded_range = Range(date_range.start - lookback,
                                        date_range.end,date_range.periodicity)
                expanded = col.rows(expanded_range)
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
        super().__init__(rolling)

class crossing(_Word):
    def __init__(self):
        def crossing(stack):
            cols = stack[:]
            stack.clear()
            headers = ','.join([col.header for col in cols])
            def crossing_col(date_range):
                return np.column_stack([col.rows(date_range) for col in cols])
            stack.append(Column('cross', f'{headers},crossing',crossing_col))
        super().__init__(crossing)

class _WindowOperator(_Word):
    def __init__(self, name, twod_operation, oned_operation):
        def window_operator(stack):
            col = stack.pop()
            def window_operator_col(date_range):
                values = col.rows(date_range)
                if len(values.shape) == 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        return np.where(np.all(np.isnan(values),axis=1), np.nan,
                                    twod_operation(values, axis=1))
                else:
                    return oned_operation(values)
            stack.append(Column(col.header, f'{col.trace},{name}', window_operator_col))
        super().__init__(window_operator)

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

def wsum(): return _WindowOperator('wsum',np.nansum, _make_expanding(np.add))
def wmean(): return _WindowOperator('wmean',np.nanmean, _expanding_mean)
def wprod(): return _WindowOperator('wprod',np.nanprod, _make_expanding(np.multiply))
def wvar(): return _WindowOperator('wvar', np.nanvar, _expanding_var)
def wstd(): return _WindowOperator('wstd', np.nanstd, _expanding_var)
def wchange(): return _WindowOperator('wchange',lambda x, axis: x[:,-1] - x[:,0],
                                        lambda x, axis: x - x[0])
def wpct_change(): return _WindowOperator('wpct_change',lambda x, axis: x[:,-1] / x[:,0] - 1,
                                        lambda x, axis: x / x[0] - 1)
def wlog_change(): return _WindowOperator('wlog_change', lambda x, axis: np.log(x[:,-1] / x[:,0]),
                                        lambda x, axis: np.log( x / x[0]))
def wfirst(): return _WindowOperator('first',lambda x, axis: x[:,0], lambda x: np.full(a.shape,a[0]))
def wlast(): return _WindowOperator('last',lambda x, axis: x[:,-1], lambda x: x)


def wlag(number):
    return rolling(number+1) + wfirst()

def wzscore():
    return _Word([wstd]) + _Word([wlast]) + _Word([wmean]) + cleave(3) + sub + swap + div

# Combinators

def _get_quote_word(quote):
    assert quote.header == 'quot'
    quote_word = quote.rows[0]
    if not hasattr(quote_word, 'func'):
        quote_word = quote_word()
    return quote_word


class call(_Word):
    def __init__(self, depth=None, copy=False):
        def call(stack, depth=depth, copy=copy):
            quote = _get_quote_word(stack.pop())
            depth = len(stack) if depth is None else depth
            if depth != 0:
                this_stack = stack[-depth:]
                if not copy:
                    del(stack[-depth:])
            else:
                this_stack = []
            quote.evaluate(this_stack)
            stack.extend(this_stack)
        super().__init__(call)

#def every(step,copy=False):
#    def every(stack):
#        quote = stack.pop()
#        copied_stack = stack[:]
#        if len(copied_stack) % step != 0:
#            raise ValueError('Stack not evenly divisible by step')
#        if not copy:
#            del(stack[:])
#        for t in zip(*[iter(copied_stack)]*step):
#            this_stack = list(t)
#            quote.row_function(quoted_stack=this_stack)
#            stack += this_stack
#    return every

class each(_Word):
    def __init__(self, start=0, end=None, every=1, copy=False):
        def each(stack,start=start,end=end,every=every, copy=copy):
            quote = _get_quote_word(stack.pop())
            end = 0 if end is None else -end
            start = len(stack) if start == 0 else -start
            selected = stack[end:start]
            if len(selected) % every != 0:
                raise ValueError('Stack not evenly divisible by every')
            if not copy:
                del(stack[end:start])
            for t in zip(*[iter(selected)]*every):
                this_stack = list(t)
                quote.evaluate(this_stack)
                stack += this_stack
        super().__init__(each)

class cleave(_Word):
    def __init__(self, num_quotations, depth=None, copy=False):
        def cleave(stack, depth=depth):
            quotes = [_get_quote_word(quote) for quote in stack[-num_quotations:]]
            del(stack[-num_quotations:])
            depth = len(stack) if depth is None else depth
            copied_stack = stack[-depth:] if depth != 0 else []
            if not copy and depth != 0:
                del(stack[-depth:])
            for quote in quotes:
                this_stack = copied_stack[:]
                quote.evaluate(this_stack)
                stack += this_stack
        super().__init__(cleave)



# Header
class hset(_Word):
    def __init__(self, *args):
        def hset(stack):
            start = len(stack) - len(args)
            for i in range(start,len(stack)):
                stack[i] = Column(args[i - start], stack[i].trace, stack[i].rows)
        super().__init__(hset)

class hformat(_Word):
    def __init__(self, format_string):
        def hformat(stack):
            col = stack.pop()
            stack.append(Column(format_string.format(col.header),
                            col.trace, col.rows))
        super().__init__(hformat)

class happly(_Word):
    def __init__(self, func):
        def happly(stack):
            col = stack.pop()
            stack.append(Column(func(col.header), col.trace, col.rows))
        super().__init__(happly)

# Data cleaning
class fill(_Word):
    def __init__(self, value):
        def fill(stack, value=value):
            col = stack.pop()
            def fill(date_range):
                x = col.rows(date_range)
                x[np.isnan(x)] = value
                return x
            stack.append(Column(col.header,f'{col.trace},fill',fill))
        super().__init__(fill)

class ffill(_Word):
    def __init__(self):
        def ffill(stack):
            col = stack.pop()
            def ffill(date_range):
                x = col.rows(date_range)
                idx = np.where(~np.isnan(x),np.arange(len(x)),0)
                np.maximum.accumulate(idx,out=idx)
                return x[idx]
            stack.append(Column(col.header,f'{col.trace},ffill',ffill))
        super().__init__(ffill)

class join(_Word):
    def __init__(self, date):
        def join(stack,date=date):
            col2 = stack.pop()
            col1 = stack.pop()
            def join_col(date_range,date=date):
                date = get_index(date_range.periodicity, date)
                if date_range.end < date:
                    return col1.rows(date_range)
                if date_range.start >= date:
                    return col2.rows(date_range)
                parts = []
                parts.append(col1.rows(Range(date_range.start, date - 1, date_range.periodicity)))
                parts.append(col2.rows(Range(date, date_range.end, date_range.periodicity)))
                return np.concatenate(parts)
            stack.append(Column('join',f'{col2.trace} {col2.trace} join({date})',join_col))
        super().__init__(join)

