import re
import copy
import warnings
import numpy as np
import pandas as pd
import pynto.time as time
from pynto.tools import *
from collections import namedtuple

Column = namedtuple('Column',['header', 'trace', 'rows'])

class Range(object):

    def __init__(self, indexer):
        self.type = 'int'
        if isinstance(indexer, slice):
            self.start = indexer.start
            self.stop = indexer.stop
            self.step = indexer.step
            if self.start is None and self.stop is None:
                self.type = 'all'
            elif (self.start and isinstance(self.start, int)) or (self.stop and isinstance(self.stop, int)):
                if self.start is None:
                    self.start = 0
                if self.step is None:
                    self.step = 1
            else:
                self.type = 'datetime'
                if not self.step:
                    self.step = 'B'
                if self.start:
                    self.start = time.get_index(self.step, self.start)
                if self.stop:
                    self.stop = time.get_index(self.step, self.stop)
                else:
                    self.stop = time.get_index(self.step, time.now()) + 1
        else:
            self.start = indexer
            self.step = 1
            if not isinstance(self.start, int):
                self.type = 'datetime'
                self.start = time.get_index('B', self.start)
                self.step = 'B'
            self.stop = self.start + 1

    def _check(self):
        assert not (self.stop is None or self.start is None)

    def __len__(self):
        self._check()
        return self.stop - self.start

    def to_index(self):
        self._check()
        if isinstance(self.step,int):
            return range(self.start, self.stop, self.step)
        else:
            return pd.date_range(time.get_date(self.step,self.start),
                time.get_date(self.step,self.stop), freq=self.step)[:-1]


class _Word(object):

    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)
        other = other._head()[0]
        this.next = other
        other.prev = this
        return other

    def __getitem__(self, key):
        row_range = key if isinstance(key, Range) else Range(key)
        return self._evaluate([], row_range)

    def _evaluate(self, stack, row_range=None):
        assert not hasattr(self, 'quoted'), 'Cannot evaluate quotation.'
        current = self._head()[0]
        while True:
            print(current)
            if hasattr(current, 'quoted'):
                stack.append(Column('quotation','quotation', current.quoted))
            else:
                if not hasattr(current, 'args'):
                    current = current()
                current._operation(stack, current.args)
            if not hasattr(current, 'next'):
                break
            current = current.next
        if not row_range is None:
            return pd.DataFrame(np.column_stack([col.rows(row_range) for col in stack]),
                        columns=[col.header for col in stack], index=row_range.to_index())

    def __invert__(self):
        return _quote(self)

    def __repr__(self):
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
            if hasattr(current, 'next'):
                s += ' + '
                current = current.next
            else:
                break
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

    def __call__(self, args):
        this = copy.deepcopy(self)
        del(args['__class__'])
        del(args['self'])
        this.args = args
        return this

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
        self.quote = True
        self.quoted = quoted
        super().__init__('quote')

    def __call__(self):
        return super().__call__(locals())

class _c(_Word):

    def __init__(self):
        super().__init__('c')

    def __call__(self, value):
        return super().__call__(locals())

    def _operation(self, stack, args):
        def const_col(row_range):
            return np.full(len(row_range), self.args['value'])
        stack.append(Column('constant', str(self.args['value']), const_col))

c = _c()

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
        frame.index.freq = pd.infer_freq(frame.index)
    else:
        index_type = 'int'
    for header, col in frame.iteritems():
        def frame_col(row_range, col=col, index_type=index_type):
            assert not (index_type == 'int' and row_range.type == 'datetime'), 'Cannot evaluate int-indexed frame over datetime range'
            if row_range.type == 'int':
                if not row_range.start:
                    row_range.start = col.index[0]
                if not row_range.stop:
                    row_range.stop = len(col.index)
                if not row_range.step:
                    row_range.step = 1
                if row_range.start < 0:
                    row_range.start += len(col.index)
                if row_range.stop < 0:
                    row_range.stop += len(col.index)
                values =  col.values[row_range.start:row_range.stop:row_range.step]
            else:
                if not row_range.start:
                    row_range.start = time.get_index(col.index.freq.name, col.index[0])
                if not row_range.stop:
                    row_range.stop = 1 + time.get_index(col.index.freq.name, col.index[-1])
                if not row_range.step:
                    row_range.step = col.index.freq.name
                if col.index.freq.name == row_range.step:
                    start = time.get_index(row_range.step, col.index[0])
                    values =  col.values[row_range.start - start:row_range.stop - start]
                else:
                    values =  col.reindex(row_range.to_index(), method='ffill').values
            if len(values) < len(row_range):
                values = np.concatenate([values, np.full(len(row_range) - len(values), np.nan)])
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
    def _operation(stack):
        col2 = stack.pop()
        col1 = stack.pop()
        def binary_operator_col(row_range):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                values1 = col1.rows(row_range)
                values2 = col2.rows(row_range)
                return np.where(np.logical_or(np.isnan(values1), np.isnan(values2)),
                                    np.nan, self.op(values1,values2))
        stack.append(Column(col1.header,f'{col1.trace}) + {col2.trace} + {self.name}',binary_operator_col))
    return _NoArgWord(name, _operation)

def _get_unary_operator(name, op):
    def _operation(stack):
        col = stack.pop()
        def unary_operator_col(row_range):
            return self.operation(col.rows(row_range))
        stack.append(Column(col.header,f'{col.trace},{self.name},',unary_operator_col))
    return _NoArgWord(name, _operation)

add = _get_binary_operator('add', np.add)
sub = _get_binary_operator('sub', np.subtract)
mul = _get_binary_operator('mul', np.multiply)
div = _get_binary_operator('div', np.divide)
mod = _get_binary_operator('mod', np.mod)
exp = _get_binary_operator('exp', np.power)
eq = _get_binary_operator('eq', np.equal)
ne = _get_binary_operator('ne', np.not_equal)
ge = _get_binary_operator('ge', np.greater_equal)
gt = _get_binary_operator('gt', np.greater)
le = _get_binary_operator('le', np.less_equal)
lt = _get_binary_operator('lt', np.less)
absv = _get_unary_operator('absv', np.abs)
sqrt = _get_unary_operator('sqrt', np.sqrt)
zeroToNa = _get_unary_operator('zeroToNa', lambda x: np.where(np.equal(x,0),np.nan,x))

# Stack manipulation
dup = _NoArgWord('dup', lambda stack: stack.append(stack[-1]))
roll = _NoArgWord('roll', lambda stack: stack.insert(0,stack.pop()))
swap = _NoArgWord('swap', lambda stack: stack.insert(-1,stack.pop()))
drop = _NoArgWord('drop', lambda stack: stack.pop())
clear = _NoArgWord('dup', lambda stack: stack.clear())

class _interleave(_Word):

    def __init__(self): super().__init__('interleave')
    def __call__(self, count=None, split_into=2): return super().__call__(locals())

    def _operation(self, stack, args):
        count = args['count'] if 'count' in args else len(stack) // split_into
        place,last = 0,0
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
    def __call__(self, *headers, clear=False):
        return super().__call__(locals())
    def _operation(self, stack, args):
        print(args)
        filtered_stack = []
        for header in args['headers']:
            to_del = []
            for i,col in enumerate(stack):
                print(f'checking {col.header} against {header}')
                if not re.match(header,col.header) is None:
                    print(f'matched {col.header}')
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        if args['clear']:
            del(stack[:])
        stack += filtered_stack
hpull = _hpull()
hfilter = lambda headers: _hpull()(*headers, clear=True)

class _ewma(_Word):
    def __init__(self): super().__init__('ewma')
    def __call__(self, window, fill_nans=True): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        def ewma_col(row_range, window=window):
            alpha = 2 /(args['window'] + 1.0)
            alpha_rev = 1-alpha
            data = col.rows(row_range)
            idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
            starting_nans = np.where(idx == -1,np.nan,1)
            data = data[~np.isnan(data)]
            if len(data) == 0:
                return np.full(idx.shape[0],np.nan)
            out = ewma_vectorized_safe(data,alpha)
            return out[idx] * starting_nans
        stack.append(Column(col.header,f'{col.trace},ewma,',ewma_col))
ewma = _ewma()

# Windows
class _rolling(_Word):
    def __init__(self): super().__init__('rolling')
    def __call__(self, window=2, exclude_nans=True, lookback_multiplier=2).__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        def rolling_col(row_range,window=args['window'], exclude_nans=args['exclude_nans'],
                            lookback_multiplier=args['lookback_multiplier']):
            if not exclude_nans:
                lookback_multiplier = 1
            lookback = (window - 1) * lookback_multiplier
            expanded_range = copy.copy(row_range)
            if row_range.start >= lookback or row_range.type == 'datetime':
                expanded_range = row_range.start - lookback
                expanded = col.rows(expanded_range)
            else:
                expanded_range = 0
                fill = np.full(lookback - row_range.start, np.nan)
                expanded = np.concatenate([fill,col.rows(expanded_range)])
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
rolling = _rolling()

def _crossing_op(stack):
    cols = stack[:]
    stack.clear()
    headers = ','.join([col.header for col in cols])
    def crossing_col(row_range):
        return np.column_stack([col.rows(row_range) for col in cols])
    stack.append(Column('cross', f'{headers},crossing',crossing_col))

crossing = _NoArgWord('crossing', _crossing_op)

def _get_window_operator(name,  twod_operation, oned_operation):
    def _operation(self, stack, args):
            col = stack.pop()
            def window_operator_col(row_range):
                values = col.rows(row_range)
                if len(values.shape) == 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        return np.where(np.all(np.isnan(values),axis=1), np.nan,
                                    twod_operation(values, axis=1))
                else:
                    return oned_operation(values)
            stack.append(Column(col.header, f'{col.trace},{name}', window_operator_col))
    return _NoArgWord(name, _operation)


wsum = _get_window_operator('wsum',np.nansum, make_expanding(np.add))
wmean = _get_window_operator('wmean',np.nanmean, expanding_mean)
wprod = _get_window_operator('wprod',np.nanprod, make_expanding(np.multiply))
wvar = _get_window_operator('wvar', np.nanvar, expanding_var)
wstd = _get_window_operator('wstd', np.nanstd, expanding_var)
wchange = _get_window_operator('wchange',lambda x, axis: x[:,-1] - x[:,0], lambda x, axis: x - x[0])
wpct_change = _get_window_operator('wpct_change',lambda x, axis: x[:,-1] / x[:,0] - 1, lambda x, axis: x / x[0] - 1)
wlog_change = _get_window_operator('wlog_change', lambda x, axis: np.log(x[:,-1] / x[:,0]), lambda x, axis: np.log( x / x[0]))
wfirst = _get_window_operator('first',lambda x, axis: x[:,0], lambda x: np.full(a.shape,a[0]))
wlast = _get_window_operator('last',lambda x, axis: x[:,-1], lambda x: x)


def wlag(number):
    return rolling(number+1) + wfirst()

wzscore = quote(wstd) + [wlast] + [wmean] + cleave(3, depth=1) + sub + swap + div

# Combinators
class _call(_Word):
    def __init__(self): super().__init__('call')
    def __call__(self, depth=None, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation'
        quoted = stack.pop().rows
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


class _each(_Word):
    def __init__(self): super().__init__('each')
    def __call__(self, start=0, end=None, every=1, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        assert stack[-1].header == 'quotation'
        quote = stack.pop().rows
        end = 0 if args['end'] is None else -args['end']
        start = len(stack) if args['start'] == 0 else -args['start']
        selected = stack[end:start]
        assert len(selected) % args['every'] != 0, 'Stack not evenly divisible by every'
        if not args['copy']:
            del(stack[end:start])
        for t in zip(*[iter(selected)]*args['every']):
            this_stack = list(t)
            quote.evaluate(this_stack)
            stack += this_stack
each = _each()

class _cleave(_Word):
    def __init__(self): super().__init__('cleave')
    def __call__(self, num_quotations, depth=None, copy=False): return super().__call__(locals())
    def _operation(self, stack, args):
        quotes = [quote.rows for quote in stack[-args['num_quotations']:]]
        del(stack[-args['num_quotations']:])
        depth = len(stack) if args['depth'] is None else args['depth']
        copied_stack = stack[-depth:] if depth != 0 else []
        if not args['copy'] and depth != 0:
            del(stack[-depth:])
        for quote in quotes:
            this_stack = copied_stack[:]
            quote.evaluate(this_stack)
            stack += this_stack
cleave = _cleave()


# Header
class _hset(_Word):
    def __init__(self): super().__init__('hset')
    def __call__(self, *headers): return super().__call__(locals())
    def _operation(self, stack, args):
        start = len(stack) - len(args['headers'])
        for i in range(start,len(stack)):
            stack[i] = Column(args['headers'][i - start], stack[i].trace, stack[i].rows)
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
        stack.append(Column(args['header_function'](col.header), col.trace, col.rows))
happly = _happly()

# Data cleaning
class _fill(_Word):
    def __init__(self): super().__init__('fill')
    def __call__(self, value): return super().__call__(locals())
    def _operation(self, stack, args):
        col = stack.pop()
        def fill(row_range, value=args['value']):
            x = col.rows(row_range)
            x[np.isnan(x)] = value
            return x
        stack.append(Column(col.header,f'{col.trace},fill',fill))
fill = _fill()

def _ffill(stack):
    col = stack.pop()
    def ffill(row_range):
        x = col.rows(row_range)
        idx = np.where(~np.isnan(x),np.arange(len(x)),0)
        np.maximum.accumulate(idx,out=idx)
        return x[idx]
    stack.append(Column(col.header,f'{col.trace},ffill',ffill))
ffill = _NoArgWord('ffill',_ffill)

class _join(_Word):
    def __init__(self): super().__init__('join')
    def __call__(self, date): return super().__call__(locals())
    def _operation(self, stack, args):
        col2 = stack.pop()
        col1 = stack.pop()
        def join_col(row_range,date=args['date']):
            if row_range.type == 'datetime':
                date = get_index(row_range.step, date)
            if row_range.stop < date:
                return col1.rows(row_range)
            if row_range.start >= date:
                return col2.rows(row_range)
            parts = []
            r = copy.copy(row_range)
            r.stop = date - 1
            parts.append(col1.rows(r))
            r = copy.copy(row_range)
            r.start = date
            parts.append(col2.rows(r))
            return np.concatenate(parts)
        stack.append(Column('join',f'{col2.trace} {col2.trace} join({date})',join_col))
join = _join()
