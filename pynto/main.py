from __future__ import annotations
import re
import copy
import warnings
import numbers
import numpy as np
import numpy.ma as ma
import pandas as pd
import traceback
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any
from .ranges import Range
from .tools import *
from .database import get_client
from . import vocabulary
from . import periodicities

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

    def rows(self, range_: Range) -> np.ndarray:
        try:
            if not _CACHING:
                return self.rows_function(range_, self.args)
            key = (range_.start, range_.stop, str(range_.periodicity))
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
            print('Error in {self.header}')
            raise e
            raise ValueError(f'{e} in rows for column "{self.header}" ({self.trace})') from e
            
@dataclass
class Word:
    name: str
    next_: Word = None
    prev: Word = None
    quoted: Word = None
    args: dict = None

    def __getattr__(self, name):
        return self.__add__(getattr(vocabulary,name))

    def __add__(self, other: Word) -> Word:
        this = self._copy()
        other = other._copy()
        other = other._head()[0]
        this.next_ = other
        other.prev = this
        return other._tail()

    def __getitem__(self, key) -> pd.DataFrame:
        range_ = key if isinstance(key, Range) else Range.from_indexer(key)
        return self._evaluate([], range_)

    def __str__(self):
        if self.quoted:
            return f'quote({self.quoted.__repr__()})'
        else:
            s = self.name
            if self.args:
                s += '(' + ', '.join([f'{k}={str(v)[:2000]}' for k,v in self.args.items() if k != self]) + ')'
        return s

    def __repr__(self, no_recurse=False):
        s = ''
        current = self
        while True:
            s = str(current) + s
            if current.prev is None:
                break
            else:
                s = '.' + s
                current = current.prev
        return s

    def __len__(self):
        return self._head()[1]

    def __call__(self, args = {}) -> Word:
        this = self._copy()
        if '__class__' in args:
            del(args['__class__'])
        if 'self' in args:
            del(args['self'])
        this.args = args
        return this

    def _operation(self, stack, args):
        pass

    @property
    def columns(self):
        return self._evaluate([], None)
    
    def quote(self, quoted: Word):
        self.next_ = Quotation(quoted=quoted)
        self.next_.prev = self
        return self.next_
    
    def _copy(self) -> Word:
        cls = self.__class__
        copied = cls.__new__(cls)
        copied.__dict__.update(self.__dict__)
        first = copied
        while copied.prev is not None:
            cls = copied.prev.__class__
            prev = cls.__new__(cls)
            prev.__dict__.update(copied.prev.__dict__)
            copied.prev = prev
            prev.next_ = copied
            copied = prev
        return first

    def _evaluate(self, stack: List[Column], range_: Range = None):
        assert not self.quoted, 'Cannot evaluate quotation.'
        current = self._head()[0]
        while True:
            if current.quoted:
                stack.append(Column('quotation','quotation', current.quoted))
            else:
                try:
                    if not current.args:
                        current = current()
                    current._operation(stack, current.args)
                except Exception as e:
                    raise e
                    traceback.print_exc()
                    error_msg = f' in word "{current.name}"'
                    if current.prev is not None:
                        error_msg += ' preceded by "'
                        prev_expr = repr(current.prev)
                        if len(prev_expr) > 150:
                            error_msg += '...'
                        error_msg += prev_expr[-150:] + '"'
                    import sys
                    raise type(e)((str(e) + error_msg).replace("'",'"')).with_traceback(sys.exc_info()[2])

                    raise SyntaxError(repr(e) + error_msg) from e
            if current.next_ is None:
                break
            current = current.next_
        if not range_ is None:
            if len(stack) == 0:
                return None
            if _MULTIPROCESSING:
                pool = NestablePool(multiprocessing.cpu_count()-1)
                #Pool(psutil.cpu_count(logical=False)-1)
                values = np.column_stack(pool.map(get_rows, [(col, range_) for col in stack]))
            else:
                values = np.column_stack([col.rows(range_) for col in stack])
            return pd.DataFrame(values,
                        columns=[col.header for col in stack], index=range_.to_index())
        else:
            return [col.header for col in stack]

    def _head(self) -> Word:
        count = 1
        current = self
        while current.prev is not None:
            count += 1
            current = current.prev
        return (current, count)

    def _tail(self) -> Word:
        current = self
        while current.next_ is not None:
            current = current.next_
        return current

@dataclass
class Quotation(Word):
    name: str = 'quotation'
    
    def __call__(self, quoted):
        self.quoted=quoted
        return super().__call__(locals())

@dataclass
class NullaryWord(Word):
    def __init__(self,
            name,
            stack_function: Callable[[List[Column],Dict[str,Any]],None],
            stack_function_args: Dict[str, Any] = {}):
        self.stack_function = stack_function
        self.stack_function_args = stack_function_args
        self.stack_function_args['name'] = name
        super().__init__(name)

    def __call__(self):
        return self

    def _operation(self, stack, args):
        self.stack_function(stack, self.stack_function_args)

    def __repr__(self):
        return super().__repr__()

def dummy_stack_function(stack, args):
    pass

def const_col(range_, args):
    return np.full(len(range_), args['value'])

@dataclass
class Constant(Word):
    name: str = 'append'

    def __call__(self, *values):
        return super().__call__(locals())

    def _operation(self, stack, args):
        for value in args['values']:
            stack.append(Column('constant', str(value), const_col, {'value': value}))

def timestamp_col(range_, args):
    return np.array(range_.to_index().astype('int'))

def timestamp_stack_function(stack, args):
    stack.append(Column('timestamp','timestamp', timestamp_col))


@dataclass
class ConstantRange(Word):
    name: str = 'c_range'

    def __call__(self, value):
        return super().__call__(locals())

    def _operation(self, stack, args):
        for value in range(args['value']):
            stack.append(Column('constant', str(self.args['value']), const_col, {'value': value}))

def frame_col(range_, args):
    col = args['col']
    if col.index.freq is None:
        try:
            col.index.freq = pd.infer_freq(col.index)
        except:
            raise Exception('Unable to determine periodicity of pandas data.')
    periodicity = periodicities.from_pandas(col.index.freq.name)
    if periodicity != range_.periodicity:
        col =  col.resample(range_.to_index()).asfreq()
    start = range_.periodicity.get_index(col.index[0].date())
    end = range_.periodicity.get_index(col.index[-1].date()) + 1
    values = []
    if range_.start < start:
        values.append(np.full(min(start - range_.start, len(range_)), np.nan))
    if range_.start < end and range_.stop >= start:
        values.append(col.values[max(range_.start - start,0):min(range_.stop - start, len(col))])
    if len(values) > 0:
        values = np.concatenate(values)
    else:
        values = pd.Series()
    if len(values) < len(range_):
        values = np.concatenate([values, np.full(len(range_) - len(values), np.nan)])
    return values

def frame_to_columns(stack, frame):
    if frame.index.freq is None:
        frame.index.freq = pd.infer_freq(frame.index)
    for header, col in frame.iteritems():
        stack.append(Column(header,f'csv {header}', frame_col, {'col': col}))

@dataclass
class Pandas(Word):
    name: str = 'pandas'

    def __call__(self, frame_or_series):
        return super().__call__(locals())

    def _operation(self, stack, args):
        frame = args['frame_or_series']
        if isinstance(frame, pd.core.series.Series):
            frame = frame.toframe()
        frame_to_columns(stack, frame)

@dataclass
class CSV(Word):
    name: str = 'csv'

    def __call__(self, csv_file, index_col=0, header='infer'):
        return super().__call__(locals())

    def _operation(self, stack, args):
        frame = pd.read_csv(args['csv_file'], index_col=args['index_col'],
                                    header=args['header'], parse_dates=True)
        _frame_to_columns(stack, frame)

def binary_operator_col(range_, args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values1 = args['col1'].rows(range_)
        values2 = args['col2'].rows(range_)
        return np.where(np.logical_or(np.isnan(values1), np.isnan(values2)), np.nan,
                    args['op'](values1,values2))

def binary_operator_stack_function(stack, args):
    col2 = stack.pop()
    col1 = stack.pop()
    stack.append(Column(col1.header,f'{col1.trace}.{col2.trace}.{args["name"]}',
                    binary_operator_col,
                    {'col1': col1, 'col2': col2, 'op': args['op']}))

def get_binary_operator(name, op):
    return NullaryWord(name, binary_operator_stack_function, {'op': op})

def unary_operator_col(range_, args):
    return args['op'](args['col'].rows(range_))

def unary_operator_stack_function(stack, args):
    col = stack.pop()
    stack.append(Column(col.header,f'{col.trace}.{args["name"]}',unary_operator_col,
                            {'op': args['op'], 'col': col}))

def get_unary_operator(name, op):
    return NullaryWord(name, unary_operator_stack_function, {'op': op})

def zero_to_na_op(x):
    return np.where(np.equal(x,0),np.nan,x)
def is_na_op(x):
    return np.where(np.isnan(x), 1, 0)

def logical_not_op(x):
    return np.where(np.logical_not(x), 1, 0)

# Stack manipulation
def dup_stack_function(stack, args):
    stack.append(stack[-1])

def roll_stack_function(stack, args):
    stack.insert(0,stack.pop())

def swap_stack_function(stack, args):
    stack.insert(-1,stack.pop())

def drop_stack_function(stack, args):
    stack.pop()

def rev_stack_function(stack, args):
    stack.reverse()

def clear_stack_function(stack, args):
    stack.clear()

def hsort_stack_function(stack, args):
    stack.sort(key=lambda c: c.header)

@dataclass
class Interleave(Word):
    name: str = 'interleave'

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
@dataclass
class Pull(Word):
    name: str = 'pull'

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

@dataclass
class HeaderPull(Word):
    name: str = 'hpull'

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

@dataclass
class HeaderFilter(HeaderPull):
    name: str = 'hfilter'
    def __call__(self, *headers, clear=True, exact_match=False):
        return super().__call__(*headers, clear=True, exact_match=False)

def ewma_col(range_, args):
    alpha = 2 /(args['window'] + 1.0)
    data = args['col'].rows(range_)
    idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
    nans = np.where(~np.isnan(data),1,np.nan)
#     starting_nans = np.where(idx == -1,np.nan,1)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.full(idx.shape[0],np.nan)
    out = ewma_vectorized_safe(data,alpha)
    #return out[idx] * starting_nans
    return out[idx] * nans

@dataclass
class EWMA(Word):
    name: str = 'ewma'
    def __call__(self, window, fill_nans=True): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        window = args['window']
        stack.append(Column(col.header,f'{col.trace}.ewma({window})',
                ewma_col, {'window': window, 'col': col}))

# Combinators
@dataclass
class Call(Word):
    name: str = 'call'
    def __call__(self, depth=None, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation', 'call needs a quotation on top of stack'
        quoted = stack.pop().rows_function
        if 'depth' not in args or args['depth'] is None:
            depth = len(stack) 
        else:
            depth = args['depth']

        if depth != 0:
            this_stack = stack[-depth:]
            if 'copy' not in args or not args['copy']:
                del(stack[-depth:])
        else:
            this_stack = []
        quoted._evaluate(this_stack)
        stack.extend(this_stack)

def partial_stack_function(stack, args):
    stack.extend(args['stack'])

@dataclass
class Partial(Word):
    name: str = 'partial'
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
        quoted.rows_function = NullaryWord('partial', partial_stack_function, {'stack': this_stack}) + quoted.rows_function
        stack.append(quoted)

@dataclass
class Each(Word):
    name: str = 'each'
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
@dataclass
class Repeat(Word):
    name: str = 'repeat'
    def __call__(self, times): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation'
        quote = stack.pop().rows_function
        for _ in range(args['times']):
            quote._evaluate(stack)

def heach_stack_function(stack, args):
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

@dataclass
class Cleave(Word):
    name: str = 'cleave'
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


# Header
@dataclass
class HeaderSet(Word):
    name: str = 'hset'
    def __call__(self, *headers): return super().__call__(locals())
    def _operation(self, stack, args):
        headers = args['headers']
        if len(headers) == 1 and headers[0].find(',') != -1:
            headers = headers[0].split(',')
        start = len(stack) - len(headers)
        for i in range(start,len(stack)):
            stack[i] = Column(headers[i - start], stack[i].trace,
                                stack[i].rows_function, stack[i].args)

@dataclass
class HeaderFormat(Word):
    name: str = 'hformat'
    def __call__(self, format_string): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(args['format_string'].format(col.header),
                            col.trace, col.rows_function, col.args))

@dataclass
class HeaderApply(Word):
    name: str = 'happly'
    def __call__(self, header_function): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(args['header_function'](col.header), col.trace,
                            col.rows_function, col.args))

# Windows
def rolling_col(range_,args):
    col = args['col']
    window = args['window']
    periodicity = args['periodicity']
    exclude_nans = args['exclude_nans']
    lookback_multiplier = args['lookback_multiplier']
    if not exclude_nans:
        lookback_multiplier = 1
    lookback = (window - 1) * lookback_multiplier
    if periodicity is None or periodicity == str(range_.periodicity):
        resample = False
        expanded_range = copy.copy(range_)
    else:
        resample = True
        expanded_range = Range(range_.start_date(),
                                range_.end_date(), periodicity)
    expanded_range.start = expanded_range.start - lookback
    expanded = col.rows(expanded_range)
    if range_.stop is None:
        range_.stop = expanded_range.stop
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
                    index=expanded_range.to_index()).reindex(range_.to_index())
@dataclass
class Rolling(Word):
    name: str = 'rolling'
    def __call__(self, window=2, exclude_nans=True, periodicity=None, lookback_multiplier=2): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        col_args = args.copy()
        col_args.update({'col': col})
        stack.append(Column(col.header,f'{col.trace}.rolling({args["window"]})', rolling_col, col_args))

def expanding_col(range_,args):
    start_date=args['start_date']
    col = args['col']
    index = range_.periodicity.get_index(start_date)
    if range_.start is None:
        range_.start = index
        return col.rows(range_)
    expanded_range = copy.copy(range_)
    offset = expanded_range.start - index
    expanded_range.start = index
    values = col.rows(expanded_range).view(ma.MaskedArray)
    values[:offset] = ma.masked
    if range_.stop is None:
        range_.stop = expanded_range.stop
    return values

@dataclass
class Expanding(Word):
    name: str = 'expanding'
    def __call__(self, start_date): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(col.header,f'{col.trace}.expanding({args["start_date"]})',
            expanding_col, {'start_date': args['start_date'], 'col': col}))

def crossing_col(range_, args):
    if _MULTIPROCESSING:
        pool = NestablePool(multiprocessing.cpu_count()-1) #Pool(psutil.cpu_count(logical=False)-1)
        return np.column_stack(pool.map(get_rows, [(col, range_) for col in args['cols']]))
    else:
        return np.column_stack([col.rows(range_) for col in args['cols']])

def crossing_op(stack, stack_args):
    cols = stack[:]
    stack.clear()
    #headers = ','.join([str(col.header) for col in cols])
    stack.append(Column(cols[0].header, f'crossing',crossing_col, {'cols':  cols}))

def rev_expanding_col(range_, args):
    return args['col'].rows(range_)[::-1]

def rev_expanding_op(stack, args):
    col = stack.pop()
    stack.append(Column(col.header, f'{col.header}.rev_expanding',
                        rev_expanding_col, {'col': col}))

def window_operator_col(range_, args):
    values = args['col'].rows(range_)
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

def cumsum_col(range_, args):
    v = args['col'].rows(range_)
    nans = np.isnan(v)
    cumsum = np.nancumsum(v)
    v[nans] = -np.diff(np.concatenate([[0.],cumsum[nans]]))
    return np.cumsum(v)

def cumsum_stack_function(stack, args):
    col = stack.pop()
    stack.append(Column(col.header, f'{col.trace}.cumsum', cumsum_col, {'col': col}))

# Data cleaning
def fill_col(range_, args):
    x = args['col'].rows(range_).copy()
    x[np.isnan(x)] = args['value']
    return x

@dataclass
class Fill(Word):
    name: str = 'fill'
    def __call__(self, value): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(col.header,f'{col.trace}.fill(arg["value"])',
                        fill_col, {'col': col, 'value': args['value']}))

def ffill_col(range_, args):
    lookback = abs(args['lookback']) 
    if range_.start is None and lookback != 0:
        args['col'].rows(range_)
    expanded_range = copy.copy(range_)
    expanded_range.start -= lookback
    x = args['col'].rows(expanded_range)
    if range_.stop is None:
        range_.stop = expanded_range.stop
    idx = np.where(~np.isnan(x),np.arange(len(x)),0)
    np.maximum.accumulate(idx,out=idx)
    return x[idx][lookback:]

@dataclass
class FFill(Word):
    name: str = 'ffill'
    def __call__(self, lookback=0): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        stack.append(Column(col.header,f'{col.trace}.ffill',
                        ffill_col, {'col': col, 'lookback': args['lookback']}))

def join_col(range_, args):
    date = args['date']
    if not isinstance(date, int):
        date = range_.periodicity.get_index(date)
    if range_.stop is not None and range_.stop < date:
        return args['col1'].rows(range_)
    if range_.start and range_.start >= date:
        return args['col2'].rows(range_)
    r_first = copy.copy(range_)
    r_first.stop = date
    r_second = copy.copy(range_)
    r_second.start = date
    v_first = args['col1'].rows(r_first)
    v_second = args['col2'].rows(r_second)
    if range_.stop is None:
        range_.stop = r_second.stop
    if range_.start is None:
        range_.start = r_first.start
    return np.concatenate([v_first, v_second])

@dataclass
class Join(Word):
    name: str = 'join'
    def __call__(self, date): return super().__call__(locals())
    def _operation(self, stack, args):
        col2 = stack.pop()
        col1 = stack.pop()
        stack.append(Column('join',f'{col1.trace}.{col2.trace}.join({args["date"]})',
                                join_col, {'col1': col1, 'col2': col2, 'date': args['date']}))

def saved_col(range_, args):
    if range_.start is not None and range_.stop is not None and range_.periodicity is not None:
        return get_client().read_series_data(args['key'], range_.start,
                        range_.stop, range_.periodicity)[3]
    else:
        data = get_client().read_series_data(args['key'])
        values = data[3][range_.start: range_.stop: range_.periodicity]
        if range_.start is None:
            range_.start = data[0]
        elif range_.start < 0:
            range_.start = data[1] + range_.start
        else:
            range_.start = data[0] + range_.start

        if range_.stop is None:
            range_.stop = data[1]
        elif range_.stop < 0:
            range_.stop = data[1] + range_.stop
        else:
            range_.stop = data[0] + range_.stop
        range_.periodicity_code = data[2]
        return values

@dataclass
class Saved(Word):
    name: str = 'saved'
    def __call__(self, key): return super().__call__(locals())
    def _operation(self, stack, args):
        md = get_client()._read_metadata(args['key'])
        if not md.is_frame:
            stack.append(Column(args['key'], args['key'], saved_col, args))
        else:   
            for header in get_client().read_frame_headers(args['key']):
                col_args = args.copy()
                col_args.update({'key': f'{args["key"]}:{header}'})
                stack.append(Column(header, f'{args["key"]}:{header}', saved_col, col_args))

def window_operator_stack_function(stack, args):
    name = args['name']
    col = stack.pop()
    col_args = args.copy()
    col_args['col'] = col
    stack.append(Column(col.header, f'{col.trace}.{name}', window_operator_col, col_args))

def get_window_operator(name,  twod_operation, oned_operation):
    return NullaryWord(name,window_operator_stack_function,
                        {'twod_operation': twod_operation, 'oned_operation': oned_operation})

def sum_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.add.accumulate(np.where(mask,0,x)))

def max_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.maximum.accumulate(np.where(mask,0,x)))

def min_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.minimum.accumulate(np.where(mask,0,x)))

def prod_oned_op(x, axis):
    mask = np.isnan(x)
    return np.where(mask, np.nan, np.multiply.accumulate(np.where(mask,0,x)))

def change_twod_op(x, axis):
    return x[:,-1] - x[:,0]

def change_oned_op(x, axis):
    return x - x[0]

def pct_change_twod_op(x, axis):
    return x[:,-1] / x[:,0] - 1

def pct_change_oned_op(x, axis):
    return x / x[0] - 1

def log_change_twod_op(x, axis):
    return np.log(x[:,-1] / x[:,0])

def log_change_oned_op(x, axis):
    return np.log( x / x[0])

def first_twod_op(x, axis):
    return  x[:,0]

def first_oned_op(x, axis):
    return np.full(x.shape,x[0])

def last_twod_op(x, axis):
    return  x[:,-1]

def last_oned_op(x, axis):
    return  x

def lag(number):
    return rolling(number+1).first

