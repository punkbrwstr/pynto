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
from dataclasses import dataclass, field
from typing import Callable

#import psutil
from multiprocessing import Pool
import multiprocessing.pool

_MULTIPROCESSING = True
_CACHING = True

def disable_multiprocessing(disable: bool = True):
    global _MULTIPROCESSING
    _MULTIPROCESSING = not disable


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

#Column = namedtuple('Column',['header', 'trace', 'rows'])
def get_rows(col_and_range):
    return col_and_range[0].rows(col_and_range[1])

@dataclass
class Column:
    header : str
    trace: str
    rows_function: Callable[[Range,dict], np.ndarray]
    args: dict = field(default_factory=list)
    copies: int = 1
    _rows_cache : dict = field(default_factory=dict)

    def rows(self, range_):
        try:
            if not _CACHING:
                return self.rows_function(range_, self.args)
            key = (range_.start, range_.stop, range_.step, range_.range_type)
            if key in self._rows_cache:
                rows = self._rows_cache[key]
            else: 
                rows = self.rows_function(range_, self.args)
                if self.copies > 1:
                    self._rows_cache[key] = rows
            if self.copies == 1:
                self._rows_cache.clear()
            else:
                self.copies -= 1
            return rows
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
                try:
                    if not hasattr(current, 'args'):
                        current = current()
                    current._operation(stack, current.args)
                except Exception as e:
                    traceback.print_exc()
                    error_msg = f' in word "{current}"'
                    if hasattr(current, 'prev'):
                        error_msg += ' preceded by "'
                        prev_expr = repr(current.prev)
                        if len(prev_expr) > 150:
                            error_msg += '...'
                        error_msg += prev_expr[-150:] + '"'
                    import sys
                    raise type(e)((str(e) + error_msg).replace("'",'"')).with_traceback(sys.exc_info()[2])

                    raise SyntaxError(repr(e) + error_msg) from e
            if not hasattr(current, 'next'):
                break
            current = current.next
        if not row_range is None:
            if len(stack) == 0:
                return None
            if _MULTIPROCESSING:
                pool = NestablePool(multiprocessing.cpu_count()-1)
                #Pool(psutil.cpu_count(logical=False)-1)
                values = np.column_stack(pool.map(get_rows, [(col, row_range) for col in stack]))
            else:
                values = np.column_stack([col.rows(row_range) for col in stack])
            return pd.DataFrame(values,
                        columns=[col.header for col in stack], index=row_range.to_index())
        else:
            return [col.header for col in stack]

    def __invert__(self):
        return _quote(self)

    def __str__(self):
        if hasattr(self, 'quoted'):
            q = str(self.quoted)
            s = '~' + q if len(self.quoted) == 1 else '~(' + q + ')'
        else:
            s = self.name
            if hasattr(self, 'args'):
                s += '(' + ', '.join([f'{k}={str(v)[:2000]}' for k,v in self.args.items() if k != self]) + ')'
        return s

    def __repr__(self, no_recurse=False):
        s = ''
        current = self
        while True:
            s = str(current) + s
            if not hasattr(current, 'prev'):
                break
            else:
                s = ' | ' + s
                current = current.prev
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
    def __init__(self, name, stack_function, stack_function_args={}):
        self.stack_function = stack_function
        self.stack_function_args = stack_function_args
        self.stack_function_args['name'] = name
        super().__init__(name)

    def __call__(self):
        return super().__call__(locals())

    def _operation(self, stack, args):
        self.stack_function(stack, self.stack_function_args)

class _quote(_Word):

    def __init__(self, quoted):
        self.quoted = quoted
        super().__init__('quote')

    def __call__(self):
        return super().__call__(locals())
def _dummy_stack_function(stack, args):
    pass
dummy = _NoArgWord('dummy', _dummy_stack_function)

def _const_col(row_range, args):
    return np.full(len(row_range), args['value'])

class _c(_Word):

    def __init__(self):
        super().__init__('c')

    def __call__(self, *values):
        return super().__call__(locals())

    def _operation(self, stack, args):
        for value in args['values']:
            stack.append(Column('constant', str(value), _const_col, {'value': value}))
c = _c()

def _timestamp_col(row_range, args):
    assert row_range.range_type == 'datetime', "Cannot get timestamp for int step range"
    return np.array(row_range.to_index().astype('int'))

def _timestamp_stack_function(stack, args):
    stack.append(Column('timestamp','timestamp',_timestamp_col))
timestamp = _NoArgWord('timestamp', _timestamp_stack_function)


class _c_range(_Word):

    def __init__(self):
        super().__init__('c_range')

    def __call__(self, value):
        return super().__call__(locals())

    def _operation(self, stack, args):
        for value in range(args['value']):
            stack.append(Column('constant', str(self.args['value']), _const_col, {'value': value}))
c_range = _c_range()

def _frame_col(r, args):
    col = args['col']
    index_type = args['index_type']
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

def _frame_to_columns(stack, frame):
    if isinstance(frame.index, pd.core.indexes.datetimes.DatetimeIndex):
        index_type = 'datetime'
        if frame.index.freq is None:
            frame.index.freq = pd.infer_freq(frame.index)
    else:
        index_type = 'int'
    for header, col in frame.iteritems():
        stack.append(Column(header,f'csv {header}',_frame_col, {'col': col, 'index_type': index_type}))


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

def _binary_operator_col(row_range, args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values1 = args['col1'].rows(row_range)
        values2 = args['col2'].rows(row_range)
        return np.where(np.logical_or(np.isnan(values1), np.isnan(values2)), np.nan,
                    args['op'](values1,values2))

def _binary_operator_stack_function(stack, args):
    col2 = stack.pop()
    col1 = stack.pop()
    stack.append(Column(col1.header,f'{col1.trace} | {col2.trace} | {args["name"]}',
                    _binary_operator_col,
                    {'col1': col1, 'col2': col2, 'op': args['op']}))

def _get_binary_operator(name, op):
    return _NoArgWord(name, _binary_operator_stack_function, {'op': op})

def _unary_operator_col(row_range, args):
    return args['op'](args['col'].rows(row_range))

def _unary_operator_stack_function(stack, args):
    col = stack.pop()
    stack.append(Column(col.header,f'{col.trace} | {args["name"]}',_unary_operator_col,
                            {'op': args['op'], 'col': col}))

def _get_unary_operator(name, op):
    return _NoArgWord(name, _unary_operator_stack_function, {'op': op})

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
def _zero_to_na_op(x):
    return np.where(np.equal(x,0),np.nan,x)
zero_to_na = _get_unary_operator('zero_to_na', _zero_to_na_op)
def _is_na_op(x):
    return np.where(np.isnan(x), 1, 0)
is_na = _get_unary_operator('is_na', _is_na_op)

def _logical_not_op(x):
    return np.where(np.logical_not(x), 1, 0)
logical_not = _get_unary_operator('logical_not', _logical_not_op)

# Stack manipulation
def _dup_stack_function(stack, args):
    stack.append(stack[-1])
dup = _NoArgWord('dup', _dup_stack_function)

def _roll_stack_function(stack, args):
    stack.insert(0,stack.pop())
roll = _NoArgWord('roll', _roll_stack_function)

def _swap_stack_function(stack, args):
    stack.insert(-1,stack.pop())
swap = _NoArgWord('swap', _swap_stack_function)

def _drop_stack_function(stack, args):
    stack.pop()
drop = _NoArgWord('drop', _drop_stack_function)

def _rev_stack_function(stack, args):
    stack.reverse()
rev = _NoArgWord('rev', _rev_stack_function)

def _clear_stack_function(stack, args):
    stack.clear()
clear = _NoArgWord('dup', _clear_stack_function)

def _hsort_stack_function(stack, args):
    stack.sort(key=lambda c: c.header)
hsort = _NoArgWord('hsort', _hsort_stack_function)

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

def _ewma_col(row_range, args):
    alpha = 2 /(args['window'] + 1.0)
    data = args['col'].rows(row_range)
    idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
    nans = np.where(~np.isnan(data),1,np.nan)
#     starting_nans = np.where(idx == -1,np.nan,1)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.full(idx.shape[0],np.nan)
    out = ewma_vectorized_safe(data,alpha)
    #return out[idx] * starting_nans
    return out[idx] * nans

class _ewma(_Word):
    def __init__(self): super().__init__('ewma')
    def __call__(self, window, fill_nans=True): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        window = args['window']
        stack.append(Column(col.header,f'{col.trace} | ewma({window})',
                _ewma_col, {'window': window, 'col': col}))
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

def _partial_stack_function(stack, args):
    stack.extend(args['stack'])
class _partial(_Word):
    def __init__(self): super().__init__('call')
    def __call__(self, depth=1, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation', 'partial needs a quotation on top of stack'
        quoted = stack.pop()
        depth = args['depth']
        if depth != 0:
            this_stack = stack[-depth:]
            if not args['copy']:
                del(stack[-depth:])
            else:
                for col in this_stack:
                    col.copies += 1
        else:
            this_stack = []
        quoted.rows_function = _NoArgWord('partial', _partial_stack_function, {'stack': this_stack}) | quoted.rows_function
        stack.append(quoted)
partial = _partial()

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
        else:
            for col in selected:
                col.copies += 1
        for t in zip(*[iter(selected)]*args['every']):
            this_stack = list(t)
            quote._evaluate(this_stack)
            stack += this_stack
each = _each()

def _heach_stack_function(stack, args):
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
heach = _NoArgWord('heach',_heach_stack_function)

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
        else:
            for col in copied_stack:
                col.copies += 1
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
            stack[i] = Column(headers[i - start], stack[i].trace,
                                stack[i].rows_function, stack[i].args)
hset = _hset()

class _hformat(_Word):
    def __init__(self): super().__init__('hformat')
    def __call__(self, format_string): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(args['format_string'].format(col.header),
                            col.trace, col.rows_function, col.args))
hformat = _hformat()

class _happly(_Word):
    def __init__(self): super().__init__('happly')
    def __call__(self, header_function): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(args['header_function'](col.header), col.trace,
                            col.rows_function, col.args))
happly = _happly()

# Windows
def _rolling_col(row_range,args):
    col = args['col']
    window = args['window']
    periodicity = args['periodicity']
    exclude_nans = args['exclude_nans']
    lookback_multiplier = args['lookback_multiplier']
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
class _rolling(_Word):
    def __init__(self): super().__init__('rolling')
    def __call__(self, window=2, exclude_nans=True, periodicity=None, lookback_multiplier=2): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        col_args = args.copy()
        col_args.update({'col': col})
        stack.append(Column(col.header,f'{col.trace} | rolling({args["window"]})',_rolling_col, col_args))
rolling = _rolling()

def expanding_col(row_range,args):
    start_date=args['start_date']
    col = args['col']
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

class _expanding(_Word):
    def __init__(self): super().__init__('expanding')
    def __call__(self, start_date): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(col.header,f'{col.trace} | expanding({args["start_date"]})',
            expanding_col, {'start_date': args['start_date'], 'col': col}))
expanding = _expanding()

def _crossing_col(row_range, args):
    if _MULTIPROCESSING:
        pool = NestablePool(multiprocessing.cpu_count()-1) #Pool(psutil.cpu_count(logical=False)-1)
        return np.column_stack(pool.map(get_rows, [(col, row_range) for col in args['cols']]))
    else:
        return np.column_stack([col.rows(row_range) for col in args['cols']])

def _crossing_op(stack, stack_args):
    cols = stack[:]
    stack.clear()
    #headers = ','.join([str(col.header) for col in cols])
    stack.append(Column(cols[0].header, f'crossing',_crossing_col, {'cols':  cols}))
crossing = _NoArgWord('crossing', _crossing_op)

def _rev_expanding_col(row_range, args):
    return args['col'].rows(row_range)[::-1]

def _rev_expanding_op(stack, args):
    col = stack.pop()
    stack.append(Column(col.header, f'{col.header} | rev_expanding',
                        _rev_expanding_col, {'col': col}))
rev_expanding = _NoArgWord('rev_expanding', _rev_expanding_op)

def _window_operator_col(row_range, args):
    values = args['col'].rows(row_range)
    if len(values.shape) == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.where(np.all(np.isnan(values),axis=1), np.nan,
                        args['twod_operation'](values, axis=1))
    else:
        cum = args['oned_operation'](values, axis=None)
        if values.strides[0] < 0:
            return cum[::-1]
        if isinstance(values,ma.MaskedArray):
            return ma.array(cum,mask=values.mask).compressed()
        return cum 

def _window_operator_stack_function(stack, args):
    name = args['name']
    col = stack.pop()
    col_args = args.copy()
    col_args['col'] = col
    stack.append(Column(col.header, f'{col.trace} | {name}',
        _window_operator_col, col_args))

def _get_window_operator(name,  twod_operation, oned_operation):
    return _NoArgWord(name, _window_operator_stack_function,
                        {'twod_operation': twod_operation, 'oned_operation': oned_operation})

def _wsum_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.add.accumulate(np.where(mask,0,x)))
wsum = _get_window_operator('wsum',np.nansum, _wsum_oned_op)

def _wmax_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.maximum.accumulate(np.where(mask,0,x)))
wmax = _get_window_operator('wmax',np.nanmax, _wmax_oned_op)

def _wmin_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.minimum.accumulate(np.where(mask,0,x)))
wmin = _get_window_operator('wmin',np.nanmin, _wmin_oned_op)

def _wprod_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.multiply.accumulate(np.where(mask,0,x)))
wprod = _get_window_operator('wprod',np.nanprod, _wprod_oned_op)

wmean = _get_window_operator('wmean',np.nanmean, expanding_mean)
wvar = _get_window_operator('wvar', np.nanvar, expanding_var)
wstd = _get_window_operator('wstd', np.nanstd, expanding_var)

def _wchange_twod_op(x, axis):
    return x[:,-1] - x[:,0]
def _wchange_oned_op(x, axis):
    return x - x[0]
wchange = _get_window_operator('wchange', _wchange_twod_op, _wchange_oned_op)

def _wpct_change_twod_op(x, axis):
    return x[:,-1] / x[:,0] - 1
def _wpct_change_oned_op(x, axis):
    return x / x[0] - 1
wpct_change = _get_window_operator('wpct_change', _wpct_change_twod_op, _wpct_change_oned_op)

def _wlog_change_twod_op(x, axis):
    return np.log(x[:,-1] / x[:,0])
def _wlog_change_oned_op(x, axis):
    return np.log( x / x[0])
wlog_change = _get_window_operator('wlog_change', _wlog_change_twod_op, _wlog_change_oned_op)

def _wfirst_twod_op(x, axis):
    return  x[:,0]
def _wfirst_oned_op(x, axis):
    return np.full(x.shape,x[0])
wfirst = _get_window_operator('first', _wfirst_twod_op, _wfirst_oned_op)

def _wlast_twod_op(x, axis):
    return  x[:,-1]
def _wlast_oned_op(x, axis):
    return  x
wlast = _get_window_operator('last', _wlast_twod_op, _wlast_oned_op)


def wlag(number):
    return rolling(number+1) | wfirst()

wzscore = ~wstd | ~wlast | ~wmean | cleave(3, depth=1) | sub | swap | div


def _cumsum_col(row_range, args):
    v = args['col'].rows(row_range)
    nans = np.isnan(v)
    cumsum = np.nancumsum(v)
    v[nans] = -np.diff(np.concatenate([[0.],cumsum[nans]]))
    return np.cumsum(v)

def _cumsum(stack, args):
    col = stack.pop()
    stack.append(Column(col.header, f'{col.trace} | cumsum', _cumsum_col, {'col': col}))
cumsum = _NoArgWord('cumsum', _cumsum)


# Data cleaning
def _fill_col(row_range, args):
    x = args['col'].rows(row_range).copy()
    x[np.isnan(x)] = args['value']
    return x
class _fill(_Word):
    def __init__(self): super().__init__('fill')
    def __call__(self, value): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(col.header,f'{col.trace} | fill',
                        _fill_col, {'col': col, 'value': args['value']}))
fill = _fill()

def _ffill_col(row_range, args):
    lookback = abs(args['lookback']) 
    if row_range.start is None and lookback != 0:
        args['col'].rows(row_range)
    expanded_range = copy.copy(row_range)
    expanded_range.start -= lookback
    x = args['col'].rows(expanded_range)
    if row_range.stop is None:
        row_range.stop = expanded_range.stop
    idx = np.where(~np.isnan(x),np.arange(len(x)),0)
    np.maximum.accumulate(idx,out=idx)
    return x[idx][lookback:]

class _ffill(_Word):
    def __init__(self): super().__init__('ffill')
    def __call__(self, lookback=0): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(col.header,f'{col.trace} | ffill',
                        _ffill_col, {'col': col, 'lookback': args['lookback']}))
ffill = _ffill()

def _join_col(row_range, args):
    date = args['date']
    if row_range.range_type == 'datetime':
        date = get_index(row_range.step, date)
    if row_range.stop is not None and row_range.stop < date:
        return args['col1'].rows(row_range)
    if row_range.start and row_range.start >= date:
        return args['col2'].rows(row_range)
    r_first = copy.copy(row_range)
    r_first.stop = date
    r_second = copy.copy(row_range)
    r_second.start = date
    v_first = args['col1'].rows(r_first)
    v_second = args['col2'].rows(r_second)
    if row_range.stop is None:
        row_range.stop = r_second.stop
    if row_range.start is None:
        row_range.start = r_first.start
    return np.concatenate([v_first, v_second])

class _join(_Word):
    def __init__(self): super().__init__('join')
    def __call__(self, date): return super().__call__(locals())
    def _operation(self, stack, args):
        col2 = stack.pop()
        col1 = stack.pop()
        stack.append(Column('join',f'{col1.trace} | {col2.trace}  | join({args["date"]})',
                                _join_col, {'col1': col1, 'col2': col2, 'date': args['date']}))
join = _join()
