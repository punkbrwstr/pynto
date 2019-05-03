import numpy as np
import pandas as pd
import re
from collections import namedtuple

Column = namedtuple('Column',['header', 'trace', 'row_function'])

def _p():
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
                            return np.full(len(date_range),arg)
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
                  rows = np.full(len(date_range),str(type(rows)))
                cols.append(rows)
            mat = np.vstack(cols).T
            return pd.DataFrame(mat, columns=headers, index=date_range)
        return exp
    return exp

def exp(*args,**kwargs):
    return _p()(*args,**kwargs)
                  
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
               
def swap(stack):
    stack.insert(-1,stack.pop())

def interleave(count):
    def interleave(stack):
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
                  
# Operators
def _binary_operator(name, operation):
    def binary_operator(stack):
        col2 = stack.pop()
        col1 = stack.pop()
        def binary_operator_col(date_range):
            return operation(col1.row_function(date_range),col2.row_function(date_range))
        stack.append(Column(col1.header,f'exp({col1.trace}),exp({col2.trace}),{name}',binary_operator_col))
    return binary_operator

def _unary_operator(name, operation):
    def unary_operator(stack):
        col = stack.pop()
        def unary_operator_col(date_range):
            return operation(col.row_function(date_range))
        stack.append(Column(col.header,f'{col.trace},{name},',unary_operator_col))
    return unary_operator
        
plus = _binary_operator('plus', np.add)
minus = _binary_operator('minus', np.subtract)
multiply = _binary_operator('multiply', np.multiply)
divide = _binary_operator('divide', np.divide)
mod = _binary_operator('mod', np.mod)
pow = _binary_operator('pow', np.power)
absolute = _unary_operator('absolute', np.abs)

# Windows
def rolling(window=2):
    def rolling(stack):
        col = stack.pop()
        def rolling_col(date_range):
            expanded_range = pd.date_range(date_range[0] - (window - 1) * date_range.freq,date_range[-1],freq=date_range.freq)
            expanded_data = col.row_function(expanded_range)
            def window_generator():
                for i in range(len(date_range)):
                    start = i
                    while start > 0 and np.count_nonzero(~np.isnan(expanded_data[start:i + window])) < window:
                        start -= 1
                    yield expanded_data[start:i + window]
            return window_generator
        stack.append(Column(col.header,f'{col.trace},rolling({window})',rolling_col))
    return rolling
                  

def expanding(stack):
    col = stack.pop()
    def expanding_col(date_range):
        data = col.row_function(date_range)
        def window_generator():
            for i in range(len(date_range)):
                yield data[0:i + 1]
        return window_generator
    stack.append(Column(col.header, f'{col.trace},expanding', expanding_col))
                  
def crossing(stack):
    cols = stack[:]
    stack.clear()
    headers = ','.join([col.header for col in cols])
    def crossing_col(date_range):
        data = [col.row_function(date_range) for col in cols]
        def window_generator():
            row = np.full(len(data),np.nan)
            for i in range(len(date_range)):
                for j in range(row.shape[0]):
                    row[j] = data[j][i]
                yield row
        return window_generator
    stack.append(Column('cross', f'{headers},crossing',crossing_col))

def fwd_rolling(window=2):
    def fwd_rolling(stack):
        col = stack.pop()
        def rolling_col(date_range):
            expanded_range = pd.date_range(date_range[0],(window - 1) * date_range.freq + date_range[-1],freq=date_range.freq)
            expanded_data = col.row_function(expanded_range)
            def window_generator():
                for i in range(len(date_range)):
                    end = i + window
                    while end <= len(expanded_data) and np.count_nonzero(~np.isnan(expanded_data[i:end])) < window:
                        end += 1
                    yield expanded_data[i:end]
            return window_generator
        stack.append(Column(col.header,f'{col.trace},rolling({window})',rolling_col))
    return fwd_rolling


def window_operator(name, operation):
    def window_operator(stack):
        col = stack.pop()
        def window_operator_col(date_range):
            generator = col.row_function(date_range)
            output = np.full(len(date_range),np.nan)
            for i, window in enumerate(generator()):
                output[i] = np.nan if np.isnan(window[-1]) else operation(window)
            return output
        stack.append(Column(col.header, f'{col.trace},{name}', window_operator_col))
    return window_operator

sum = window_operator('sum',np.nansum)
mean = window_operator('mean',np.nanmean)
std = window_operator('std',np.nanstd)
cumprod = window_operator('cumprod',np.nancumprod)
max = window_operator('cumprod',np.nancumprod)
min = window_operator('cumprod',np.nancumprod)
median = window_operator('cumprod',np.nancumprod)
pct_change = window_operator('pct_change',lambda a: a[-1] / a[0] -1)
log_change = window_operator('log_change',lambda a: np.log(a[-1]) - np.log(a[0]))
lag = window_operator('lag',lambda a: a[0])
last = window_operator('lag',lambda a: a[-1])


# Combinators
def call(depth=None,copy=False):
    def call(stack,depth=depth,copy=copy):
        quote = stack.pop()
        depth = len(stack) if depth is None else depth
        this_stack = stack[-depth:]
        if not copy:
            del(stack[-depth:])
        quote.row_function(quoted_stack=this_stack)
        stack += this_stack
    return call

def keep(depth=None):
    return call(depth,True)

def every(step,copy=False):
    def every(stack):
        quote = stack.pop()
        copied_stack = stack[:]
        if len(copied_stack) % step != 0:
            raise Exception('Stack not evenly divisible by step')
        if not copy:
            del(stack[:])
        for t in zip(*[iter(copied_stack)]*step):
            this_stack = list(t)
            quote.row_function(quoted_stack=this_stack)
            stack += this_stack
    return every

def cleave(depth):
    def cleave(stack):
        quotes = stack[-depth:]
        del(stack[-depth:])
        copied_stack = stack[:]
        del(stack[:])
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
            raise Exception('Stack not evenly divisible by step')
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
def ffill(stack):
    col = stack.pop()
    def ffill(date_range):
        x = col.row_function(date_range)
        idx = np.where(~np.isnan(x),np.arange(len(x)),0)
        np.maximum.accumulate(idx,out=idx)
        return x[idx]
    stack.append(Column(col.header,f'{col.trace},ffill',ffill))


# Utils

yest = lambda : pd.date_range(pd.datetime.today().date() - pd.tseries.offsets.BDay(),pd.datetime.today().date() - pd.tseries.offsets.BDay(),freq='B')

