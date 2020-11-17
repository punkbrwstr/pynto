import re
import copy
import warnings
import numbers
import numpy as np
import numpy.ma as ma
import pandas as pd
import traceback
from pynto.ranges import Range, get_index
from pynto.tools import *
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

#Column = namedtuple('Column',['header', 'trace', 'rows'])

@dataclass
class Column:
    header : str
    trace: str
    rows_function: Callable[[Range], np.ndarray]

    def rows(self, range_):
        try:
            return self.rows_function(range_)
        except Exception as e:
            raise ValueError(f'{e} in rows for column "{self.header}" ({self.trace})') from e
            

class _Word:

    def __init__(self, name):
        self.name = name

    def __or__(self, other):
        if isinstance(other, numbers.Number):
            other = _c()(other)
        this = self._copy()
        other = other._copy()
        other = other._head()[0]
        this.next = other
        other.prev = this
        return other._tail()
    
    def _copy(self):
        cls = self.__class__
        copied = cls.__new__(cls)
        copied.__dict__.update(self.__dict__)
        first = copied
        while hasattr(copied, 'prev'):
            cls = copied.prev.__class__
            prev = cls.__new__(cls)
            prev.__dict__.update(copied.prev.__dict__)
            copied.prev = prev
            prev.next = copied
            copied = prev
        return first

    def __getitem__(self, key):
        row_range = key if isinstance(key, Range) else Range.from_indexer(key)
        return self._evaluate([], row_range)

    def _evaluate(self, stack, row_range=None):
        assert not hasattr(self, 'quoted'), 'Cannot evaluate quotation.'
        current = self._head()[0]
        while True:
            if hasattr(current, 'quoted'):
                stack.append(Column('quotation','quotation', current.quoted))
            else:
                if not hasattr(current, 'args'):
                    current = current()
                try:
                    current._operation(stack, current.args)
                except Exception as e:
                    traceback.print_exc()
                    raise SyntaxError(f'{e} in word {current.__repr__(True)}') from e
            if not hasattr(current, 'next'):
                break
            current = current.next
        if not row_range is None:
            if len(stack) == 0:
                return None
            values = np.column_stack([col.rows(row_range) for col in stack])
            return pd.DataFrame(values,
                        columns=[col.header for col in stack], index=row_range.to_index())
        else:
            return [col.header for col in stack]

    def __invert__(self):
        return _quote(self)

    def __repr__(self, no_recurse=False):
        s = ''
        current = self._head()[0]
        while True:
            if hasattr(current, 'quoted'):
                q = str(current.quoted)
                s += '~' + q if len(current.quoted) == 1 else '~(' + q + ')'
            else:
                s += current.name
                if hasattr(current, 'args'):
                    a = []
                    for k,v in current.args.items():
                        if k != 'self':
                            a.append(str(k) + '=' + str(v))
                    s += '(' + ', '.join(a) + ')'
            if no_recurse or not hasattr(current, 'next'):
                break
            else:
                s += ' | '
                current = current.next
        return s

    def __len__(self):
        return self._head()[1]

    def _head(self):
        count = 1
        current = self
        while hasattr(current, 'prev'):
            count += 1
            current = current.prev
        return (current, count)

    def _tail(self):
        current = self
        while hasattr(current, 'next'):
            current = current.next
        return current

    def __call__(self, args):
        this = self._copy()
        del(args['__class__'])
        del(args['self'])
        this.args = args
        return this

    def columns(self):
        return self._evaluate([], None)

class _NoArgWord(_Word):
    def __init__(self, name, stack_function):
        self.stack_function = stack_function
        super().__init__(name)

    def __call__(self):
        return super().__call__(locals())

    def _operation(self, stack, args):
        self.stack_function(stack)

class _quote(_Word):

    def __init__(self, quoted):
        self.quoted = quoted
        super().__init__('quote')

    def __call__(self):
        return super().__call__(locals())

dummy = _NoArgWord('dummy', lambda stack: None)

class _c(_Word):

    def __init__(self):
        super().__init__('c')

    def __call__(self, *values):
        return super().__call__(locals())

    def _operation(self, stack, args):
        for value in args['values']:
            def const_col(row_range, value=value):
                return np.full(len(row_range), value)
            stack.append(Column('constant', str(value), const_col))
c = _c()

def _timestamp(stack):
    def timestamp_col(row_range):
        assert row_range.range_type == 'datetime', "Cannot get timestamp for int step range"
        return np.array(row_range.to_index().astype('int'))
    stack.append(Column('timestamp','timestamp',timestamp_col))
timestamp = _NoArgWord('timestamp', _timestamp)

class _c_range(_Word):

    def __init__(self):
        super().__init__('c_range')

    def __call__(self, value):
        return super().__call__(locals())

    def _operation(self, stack, args):
        for i in range(args['value']):
            def const_col(row_range, i=i):
                return np.full(len(row_range), i)
            stack.append(Column('constant', str(self.args['value']), const_col))

c_range = _c_range()

def _frame_to_columns(stack, frame):
    if isinstance(frame.index, pd.core.indexes.datetimes.DatetimeIndex):
        index_type = 'datetime'
        if frame.index.freq is None:
            frame.index.freq = pd.infer_freq(frame.index)
    else:
        index_type = 'int'
    for header, col in frame.iteritems():
        def frame_col(r, col=col, index_type=index_type):
            assert not (index_type == 'int' and r.range_type == 'datetime'), 'Cannot evaluate int-indexed frame over datetime range'
            if r.range_type == 'int':
                if r.start is None:
                    r.start = 0
                elif r.start < 0:
                    r.start += len(col.index)
                if r.stop is None:
                    r.stop = len(col.index)
                elif r.stop < 0:
                    r.stop += len(col.index)
                values = col.values[r.start:r.stop:r.step]
            else:
                if col.index.freq is None:
                    try:
                        col.index.freq = pd.infer_freq(col.index)
                    except:
                        raise Exception('Unable to determine periodicity of pandas data.')
                if not r.start:
                    r.start = get_index(col.index.freq.name, col.index[0])
                if not r.stop:
                    r.stop = 1 + get_index(col.index.freq.name, col.index[-1])
                if not r.step:
                    r.step = col.index.freq.name
                if col.index.freq.name != r.step:
                    col =  col.resample(r.to_index()).asfreq()
                start, end = get_index(r.step, col.index[0]), get_index(r.step, col.index[-1]) + 1
                values = []
                if r.start < start:
                    values.append(np.full(min(start - r.start, len(r)), np.nan))
                if r.start < end and r.stop >= start:
                    values.append(col.values[max(r.start - start,0):min(r.stop - start, len(col))])
                if len(values) > 0:
                    values = np.concatenate(values)
                else:
                    values = pd.Series()
            if len(values) < len(r):
                values = np.concatenate([values, np.full(len(r) - len(values), np.nan)])
            return values
        stack.append(Column(header,f'csv {header}',frame_col))


class _pandas(_Word):

    def __init__(self):
        super().__init__('pandas')

    def __call__(self, frame_or_series):
        return super().__call__(locals())

    def _operation(self, stack, args):
        frame = args['frame_or_series']
        if isinstance(frame, pd.core.series.Series):
            frame = frame.toframe()
        _frame_to_columns(stack, frame)

pandas = _pandas()

class _csv(_Word):
    def __init__(self):
        super().__init__('csv')

    def __call__(self, csv_file, index_col=0, header='infer'):
        return super().__call__(locals())

    def _operation(self, stack, args):
        frame = pd.read_csv(args['csv_file'], index_col=args['index_col'],
                                    header=args['header'], parse_dates=True)
        _frame_to_columns(stack, frame)

csv = _csv()

def _get_binary_operator(name, op):
    def stack_function(stack):
        col2 = stack.pop()
        col1 = stack.pop()
        def binary_operator_col(row_range):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                values1 = col1.rows(row_range)
                values2 = col2.rows(row_range)
                return np.where(np.logical_or(np.isnan(values1), np.isnan(values2)),
                                    np.nan, op(values1,values2))
        stack.append(Column(col1.header,f'{col1.trace} | {col2.trace} | {name}',binary_operator_col))
    return _NoArgWord(name, stack_function)

def _get_unary_operator(name, op):
    def stack_function(stack):
        col = stack.pop()
        def unary_operator_col(row_range):
            return op(col.rows(row_range))
        stack.append(Column(col.header,f'{col.trace} | {name}',unary_operator_col))
    return _NoArgWord(name, stack_function)

add = _get_binary_operator('add', np.add)
sub = _get_binary_operator('sub', np.subtract)
mul = _get_binary_operator('mul', np.multiply)
power = _get_binary_operator('power', np.power)
div = _get_binary_operator('div', np.divide)
mod = _get_binary_operator('mod', np.mod)
eq = _get_binary_operator('eq', np.equal)
ne = _get_binary_operator('ne', np.not_equal)
ge = _get_binary_operator('ge', np.greater_equal)
gt = _get_binary_operator('gt', np.greater)
le = _get_binary_operator('le', np.less_equal)
lt = _get_binary_operator('lt', np.less)
neg = _get_unary_operator('neg', np.negative)
inv = _get_unary_operator('inv', np.reciprocal)
absv = _get_unary_operator('absv', np.abs)
sqrt = _get_unary_operator('sqrt', np.sqrt)
exp = _get_unary_operator('exp', np.exp)
log = _get_unary_operator('log', np.log)
zeroToNa = _get_unary_operator('zeroToNa', lambda x: np.where(np.equal(x,0),np.nan,x))

# Stack manipulation
dup = _NoArgWord('dup', lambda stack: stack.append(stack[-1]))
roll = _NoArgWord('roll', lambda stack: stack.insert(0,stack.pop()))
swap = _NoArgWord('swap', lambda stack: stack.insert(-1,stack.pop()))
drop = _NoArgWord('drop', lambda stack: stack.pop())
rev = _NoArgWord('rev', lambda stack: stack.reverse())
clear = _NoArgWord('dup', lambda stack: stack.clear())
hsort = _NoArgWord('hsort', lambda stack: stack.sort(key=lambda c: c.header))

class _interleave(_Word):

    def __init__(self): super().__init__('interleave')
    def __call__(self, count=None, split_into=2): return super().__call__(locals())

    def _operation(self, stack, args):
        count = args['count'] if args['count'] else len(stack) // args['split_into']
        last = 0
        lists = []
        for i in range(len(stack)+1):
            if i % count == 0 and i != 0:
                lists.append(stack[i-count:i])
                last = i
        del(stack[:last])
        stack += [val for tup in zip(*lists) for val in tup]
interleave = _interleave()

class _pull(_Word):

    def __init__(self): super().__init__('pull')
    def __call__(self, start, end=None, clear=False): return super().__call__(locals())

    def _operation(self, stack, args):
        end = -args['start'] - 1 if args['end'] is None else -args['end']
        start = len(stack) if args['start'] == 0 else -args['start']
        pulled = stack[end:start]
        if args['clear']:
            del(stack[:])
        else:
            del(stack[end:start])
        stack += pulled
pull = _pull()

class _hpull(_Word):
    def __init__(self): super().__init__('hpull')
    def __call__(self, *headers, clear=False, exact_match=False):
        return super().__call__(locals())
    def _operation(self, stack, args):
        filtered_stack = []
        for header in args['headers']:
            to_del = []
            matcher = lambda c: header == c.header if args['exact_match'] else re.match(header,col.header) is not None
            for i,col in enumerate(stack):
                if matcher(col):
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        if args['clear']:
            del(stack[:])
        stack += filtered_stack
hpull = _hpull()
def hfilter(*headers, exact_match=False):
    return _hpull()(*headers, clear=True, exact_match=exact_match)

class _ewma(_Word):
    def __init__(self): super().__init__('ewma')
    def __call__(self, window, fill_nans=True): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        window = args['window']
        def ewma_col(row_range, window=window):
            alpha = 2 /(args['window'] + 1.0)
            data = col.rows(row_range)
            idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
            nans = np.where(~np.isnan(data),1,np.nan)
       #     starting_nans = np.where(idx == -1,np.nan,1)
            data = data[~np.isnan(data)]
            if len(data) == 0:
                return np.full(idx.shape[0],np.nan)
            out = ewma_vectorized_safe(data,alpha)
            #return out[idx] * starting_nans
            return out[idx] * nans
        stack.append(Column(col.header,f'{col.trace} | ewma({window})',ewma_col))
ewma = _ewma()

# Combinators
class _call(_Word):
    def __init__(self): super().__init__('call')
    def __call__(self, depth=None, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation', 'call needs a quotation on top of stack'
        quoted = stack.pop().rows_function
        depth = len(stack) if args['depth'] is None else args['depth']
        if depth != 0:
            this_stack = stack[-depth:]
            if not args['copy']:
                del(stack[-depth:])
        else:
            this_stack = []
        quoted._evaluate(this_stack)
        stack.extend(this_stack)
call = _call()

class _curry(_Word):
    def __init__(self): super().__init__('call')
    def __call__(self, depth=1, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation', 'curry needs a quotation on top of stack'
        quoted = stack.pop()
        depth = args['depth']
        if depth != 0:
            this_stack = stack[-depth:]
            if not args['copy']:
                del(stack[-depth:])
        else:
            this_stack = []
        quoted.rows_function = _NoArgWord('curried', lambda stack: stack.extend(this_stack)) | quoted.rows_function
        stack.append(quoted)
curry = _curry()

class _each(_Word):
    def __init__(self): super().__init__('each')
    def __call__(self, start=0, end=None, every=1, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation'
        quote = stack.pop().rows_function
        end = 0 if args['end'] is None else -args['end']
        start = len(stack) if args['start'] == 0 else -args['start']
        selected = stack[end:start]
        assert len(selected) % args['every'] == 0, 'Stack not evenly divisible by every'
        if not args['copy']:
            del(stack[end:start])
        for t in zip(*[iter(selected)]*args['every']):
            this_stack = list(t)
            quote._evaluate(this_stack)
            stack += this_stack
each = _each()

def _heach(stack):
    assert stack[-1].header == 'quotation'
    quote = stack.pop().rows_function
    new_stack = []
    for header in set([c.header for c in stack]):
        to_del, filtered_stack = [], []
        for i,col in enumerate(stack):
            if header == col.header:
                filtered_stack.append(stack[i])
                to_del.append(i)
        quote._evaluate(filtered_stack)
        new_stack += filtered_stack
        to_del.sort(reverse=True)
        for i in to_del:
            del(stack[i])
    del(stack[:])
    stack += new_stack
heach = _NoArgWord('heach',_heach)

class _cleave(_Word):
    def __init__(self): super().__init__('cleave')
    def __call__(self, num_quotations=-1, depth=None, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        if args['num_quotations'] < 0:
            args['num_quotations'] = len(stack)
        quotes = [quote.rows_function for quote in stack[-args['num_quotations']:]]
        del(stack[-args['num_quotations']:])
        depth = len(stack) if args['depth'] is None else args['depth']
        copied_stack = stack[-depth:] if depth != 0 else []
        if not args['copy'] and depth != 0:
            del(stack[-depth:])
        for quote in quotes:
            this_stack = copied_stack[:]
            quote._evaluate(this_stack)
            stack += this_stack
cleave = _cleave()


# Header
class _hset(_Word):
    def __init__(self): super().__init__('hset')
    def __call__(self, *headers): return super().__call__(locals())
    def _operation(self, stack, args):
        headers = args['headers']
        if len(headers) == 1 and headers[0].find(',') != -1:
            headers = headers[0].split(',')
        start = len(stack) - len(headers)
        for i in range(start,len(stack)):
            stack[i] = Column(headers[i - start], stack[i].trace, stack[i].rows_function)
hset = _hset()

class _hformat(_Word):
    def __init__(self): super().__init__('hformat')
    def __call__(self, format_string): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(args['format_string'].format(col.header),
                            col.trace, col.rows))
hformat = _hformat()

class _happly(_Word):
    def __init__(self): super().__init__('happly')
    def __call__(self, header_function): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(args['header_function'](col.header), col.trace, col.rows_function))
happly = _happly()

# Windows
class _rolling(_Word):
    def __init__(self): super().__init__('rolling')
    def __call__(self, window=2, exclude_nans=True, periodicity=None, lookback_multiplier=2): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        def rolling_col(row_range,window=args['window'],
                periodicity=args['periodicity'],
                exclude_nans=args['exclude_nans'],
                lookback_multiplier=args['lookback_multiplier']):
            if not exclude_nans:
                lookback_multiplier = 1
            lookback = (window - 1) * lookback_multiplier
            if periodicity is None or periodicity == row_range.step:
                resample = False
                expanded_range = copy.copy(row_range)
            else:
                assert row_range.range_type == 'datetime', "Cannot change periodicity for int step range"
                resample = True
                expanded_range = Range.from_dates(row_range.start_date(),
                                        row_range.end_date(), periodicity)
            if ((not expanded_range.start is None and expanded_range.start >= lookback)
                    or expanded_range.range_type == 'datetime'):
                expanded_range.start = expanded_range.start - lookback
            else:
                expanded_range.start = 0
            expanded = col.rows(expanded_range)
            if row_range.stop is None:
                row_range.stop = expanded_range.stop
            #length = len(expanded_range)
            #if expanded.shape[0] < length + lookback:
                #fill = np.full(length + lookback - expanded.shape[0], np.nan)
                #expanded = np.concatenate([fill,col.rows(expanded_range)])
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
            if not resample:
                return td[lookback:]
            else:
                expanded_range.start = expanded_range.start + lookback
                return pd.DataFrame(td[lookback:],
                            index=expanded_range.to_index()).reindex(row_range.to_index())
        stack.append(Column(col.header,f'{col.trace} | rolling({args["window"]})',rolling_col))
rolling = _rolling()

class _expanding(_Word):
    def __init__(self): super().__init__('expanding')
    def __call__(self, start_date): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        def expanding_col(row_range,start_date=args['start_date']):
            index = get_index(row_range.step, start_date)
            if row_range.start is None:
                row_range.start = index
                return col.rows(row_range)
            expanded_range = copy.copy(row_range)
            offset = expanded_range.start - index
            expanded_range.start = index
            values = col.rows(expanded_range).view(ma.MaskedArray)
            values[:offset] = ma.masked
            if row_range.stop is None:
                row_range.stop = expanded_range.stop
            return values
        stack.append(Column(col.header,f'{col.trace} | expanding({args["start_date"]})',expanding_col))
expanding = _expanding()

def _crossing_op(stack):
    cols = stack[:]
    stack.clear()
    headers = ','.join([str(col.header) for col in cols])
    def crossing_col(row_range):
        return np.column_stack([col.rows(row_range) for col in cols])
    stack.append(Column(cols[0].header, f'{headers},crossing',crossing_col))

crossing = _NoArgWord('crossing', _crossing_op)

def _rev_expanding_op(stack):
    col = stack.pop()
    def rev_expanding_col(row_range):
        return col.rows(row_range)[::-1]
    stack.append(Column(col.header, f'{col.header} | rev_expanding',rev_expanding_col))

rev_expanding = _NoArgWord('rev_expanding', _rev_expanding_op)

def _get_window_operator(name,  twod_operation, oned_operation):
    def _operation(stack):
            col = stack.pop()
            def window_operator_col(row_range):
                values = col.rows(row_range)
                if len(values.shape) == 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        return np.where(np.all(np.isnan(values),axis=1), np.nan,
                                    twod_operation(values, axis=1))
                else:
                    cum = oned_operation(values, axis=None)
                    if values.strides[0] < 0:
                        return cum[::-1]
                    if isinstance(values,ma.MaskedArray):
                        return ma.array(cum,mask=values.mask).compressed()
                    return cum 
            stack.append(Column(col.header, f'{col.trace} | {name}', window_operator_col))
    return _NoArgWord(name, _operation)


wsum = _get_window_operator('wsum',np.nansum, make_expanding(np.add))
wmean = _get_window_operator('wmean',np.nanmean, expanding_mean)
wmax = _get_window_operator('wmax',np.nanmax, make_expanding(np.maximum))
wmin = _get_window_operator('wmin',np.nanmin, make_expanding(np.minimum))
wprod = _get_window_operator('wprod',np.nanprod, make_expanding(np.multiply))
wvar = _get_window_operator('wvar', np.nanvar, expanding_var)
wstd = _get_window_operator('wstd', np.nanstd, expanding_var)
wchange = _get_window_operator('wchange',lambda x, axis: x[:,-1] - x[:,0], lambda x, axis: x - x[0])
wpct_change = _get_window_operator('wpct_change',lambda x, axis: x[:,-1] / x[:,0] - 1, lambda x, axis: x / x[0] - 1)
wlog_change = _get_window_operator('wlog_change', lambda x, axis: np.log(x[:,-1] / x[:,0]), lambda x, axis: np.log( x / x[0]))
wfirst = _get_window_operator('first',lambda x, axis: x[:,0], lambda x: np.full(x.shape,x[0]))
wlast = _get_window_operator('last',lambda x, axis: x[:,-1], lambda x: x)


def wlag(number):
    return rolling(number+1) | wfirst()

wzscore = ~wstd | ~wlast | ~wmean | cleave(3, depth=1) | sub | swap | div


def _cumsum(stack):
        col = stack.pop()
        def cumsum_col(row_range):
            v = col.rows(row_range)
            nans = np.isnan(v)
            cumsum = np.nancumsum(v)
            v[nans] = -np.diff(np.concatenate([[0.],cumsum[nans]]))
            return np.cumsum(v)
        stack.append(Column(col.header, f'{col.trace} | cumsum', cumsum_col))
cumsum = _NoArgWord('cumsum', _cumsum)


# Data cleaning
class _fill(_Word):
    def __init__(self): super().__init__('fill')
    def __call__(self, value): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        def fill(row_range, value=args['value']):
            x = col.rows(row_range).copy()
            x[np.isnan(x)] = value
            return x
        stack.append(Column(col.header,f'{col.trace} | fill',fill))
fill = _fill()

class _ffill(_Word):
    def __init__(self): super().__init__('ffill')
    def __call__(self, lookback=0): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        def ffill(row_range):
            lookback = abs(args['lookback']) 
            if row_range.start is None and lookback != 0:
                col.rows(row_range)
            expanded_range = copy.copy(row_range)
            expanded_range.start -= lookback
            x = col.rows(expanded_range)
            if row_range.stop is None:
                row_range.stop = expanded_range.stop
            idx = np.where(~np.isnan(x),np.arange(len(x)),0)
            np.maximum.accumulate(idx,out=idx)
            return x[idx][lookback:]
        stack.append(Column(col.header,f'{col.trace} | ffill',ffill))
ffill = _ffill()

class _join(_Word):
    def __init__(self): super().__init__('join')
    def __call__(self, date): return super().__call__(locals())
    def _operation(self, stack, args):
        col2 = stack.pop()
        col1 = stack.pop()
        def join_col(row_range,date=args['date']):
            if row_range.range_type == 'datetime':
                date = get_index(row_range.step, date)
            if row_range.stop is not None and row_range.stop < date:
                return col1.rows(row_range)
            if row_range.start and row_range.start >= date:
                return col2.rows(row_range)
            r_first = copy.copy(row_range)
            r_first.stop = date
            r_second = copy.copy(row_range)
            r_second.start = date
            v_first = col1.rows(r_first)
            v_second = col2.rows(r_second)
            if row_range.stop is None:
                row_range.stop = r_second.stop
            if row_range.start is None:
                row_range.start = r_first.start
            return np.concatenate([v_first, v_second])
        stack.append(Column('join',f'{col1.trace} | {col2.trace}  | join({args["date"]})',join_col))
join = _join()
