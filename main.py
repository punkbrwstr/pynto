import numpy as np
import pandas as pd
import re

def p():
    funcs = []
    def exp(*args,**kwargs):
        nonlocal funcs
        if 'quoted_stack' in kwargs:
            stack = kwargs['quoted_stack']
            for func in funcs:
                #print(f'quote calling {func.__name__}')      
                #print(f'\tquote before stack: [{",".join([s[0] for s in stack])}]')
                func(stack)
                #print(f'\tquote after stack: [{",".join([s[0] for s in stack])}]')
        elif len(args) == 1 and isinstance(args[0],list):
            # quotation puts its on stack
            args[0].append(('quote',exp))
        elif len(args) == 1 and isinstance(args[0],pd.core.indexes.datetimes.DatetimeIndex):
            # evaluating quotation itself with combinator
            return exp
        else:
            for arg in args:
                if isinstance(arg,(int, float, complex)):
                    def lit(stack,arg=arg):
                        def lit_col(date_range):
                            return np.full(len(date_range),arg)
                        stack.append((str(arg),lit_col))
                    arg = lit
                print(f'adding func {arg.__name__}')
                funcs.append(arg)
            print(f'funcs: [{",".join([func.__name__ for func in funcs])}]')
        if 'eval' in kwargs:
            date_range = kwargs['eval']
            stack = []
            for func in funcs:
                #print(f'calling {func.__name__}')
                #print(f'\tbefore stack: [{",".join([s[0] for s in stack])}]')
                func(stack)
                #print(f'\tafter stack: [{",".join([s[0] for s in stack])}]')
            cols = {} 
            for s in stack:
                col = s[1](date_range)                
                if isinstance(col,(np.ndarray, np.generic)):
                  cols[s[0]] = col
                else:
                  cols[s[0]] = np.full(len(date_range),str(type(col)))
            df = pd.DataFrame(cols, index=date_range)
            return df
        return exp
    return exp
                  
# Stack manipulation
def dup(stack):
    stack.append(stack[-1])
                  
def roll(stack):
    stack.insert(0,stack.pop())
                  
def drop(stack):
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
                  
# Operators
def _binary_operator(name, operation):
    def binary_operator(stack):
        col2 = stack.pop()
        col1 = stack.pop()
        def binary_operator_col(date_range):
            return operation(col1[1](date_range),col2[1](date_range))
        stack.append((f'({col1[0]} {col2[0]} {name})',binary_operator_col))
    return binary_operator

def _unary_operator(name, operation):
    def unary_operator(stack):
        col1 = stack.pop()
        def unary_operator_col(date_range):
            return operation(col1[1](date_range))
        stack.append((f'({col1[0]} {name})',unary_operator_col))
    return unary_operator
        
plus = _binary_operator('plus', np.add)
minus = _binary_operator('minus', np.subtract)
multiply = _binary_operator('multiply', np.multiply)
divide = _binary_operator('divide', np.divide)
mod = _binary_operator('mod', np.mod)
exp = _binary_operator('exp', np.power)
absolute = _unary_operator('absolute', np.abs)

def rolling(window=2):
    def rolling(stack):
        col1 = stack.pop()
        def rolling_col(date_range):
            expanded_range = pd.date_range(date_range[0] - (window - 1) * date_range.freq,date_range[-1],freq=date_range.freq)
            expanded_data = col1[1](expanded_range)
            def window_generator():
                for i in range(len(date_range)):
                    start = i
                    while start > 0 and np.count_nonzero(~np.isnan(expanded_data[start:i + window])) < window:
                        start -= 1
                    yield expanded_data[start:i + window]
            return window_generator
        stack.append((f'({col1[0]} rolling({window}))',rolling_col))
    return rolling
                  

def expanding(stack):
    col1 = stack.pop()
    def expanding_col(date_range):
        data = col1[1](date_range)
        def window_generator():
            for i in range(len(date_range)):
                yield data[0:i + 1]
        return window_generator
    stack.append((f'({col1[0]} expanding)',expanding_col))
                  
def crossing(stack):
    cols = [stack]
    stack.clear()
    def crossing_col(date_range):
        data = [col[1](date_range) for col in cols]
        def window_generator():
            row = np.full(len(data),np.nan)
            for i in range(len(date_range)):
                for j in range(row.shape[0]):
                    row[j] = data[j][i]
                yield row
        return window_generator
    stack.append((f'(crossing)',crossing_col))


def window_operator(name, operation):
    def window_operator(stack):
        col1 = stack.pop()
        def window_operator_col(date_range):
            generator = col1[1](date_range)
            output = np.full(len(date_range),np.nan)
            for i, window in enumerate(generator()):
                output[i] = np.nan if np.isnan(window[-1]) else operation(window)
            return output
        stack.append((f'({col1[0]} {name})', window_operator_col))
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


# Combinators
def call(stack):
    quote = stack.pop()
    quote[1](quoted_stack=stack)

def each(start=0, end=-1, step=1, headers=[]):
    def each(stack,start=start,end=end,step=step,headers=headers):
        quote = stack.pop()
        output_stack = []
        if len(headers) == 0:
            end = end if end >= 0 else len(stack) + end + 1
            for i in range(start,end,step):
                last = i + step
                this_stack = stack[i:last]
                quote[1](quoted_stack=this_stack)
                output_stack += this_stack
            del(stack[start:last])
        else:
            if not isinstance(headers, list):
                headers = [headers]
            for header in headers:
                this_stack = []
                to_del = []
                for i,(h,f) in enumerate(stack):
                    print(f'checking {h}')
                    if not re.match(header,h) is None:
                        print(f'matched {h}')
                        this_stack.append(stack[i])
                        to_del.append(i)
                to_del.sort(reverse=True)
                for i in to_del: 
                    del(stack[i])
                quote[1](quoted_stack=this_stack)
                output_stack += this_stack
        stack += output_stack
    return each

# Header
def hset(*args):
    def hset(stack):
        start = len(stack) - len(args)
        for i in range(start,len(stack)): 
            stack[i] = (args[i - start],stack[i][1])
    return hset

def hformat(format_string):
    def hset(stack):
        for i in range(len(stack)): 
            stack[i] = (format_string.format(stack[i][0]),stack[i][1])
    return hset
