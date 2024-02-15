from __future__ import annotations
import re
import copy
import warnings
import numbers
import uuid
import numpy as np
import numpy.ma as ma
import pandas as pd
import traceback
from dataclasses import dataclass, field
from functools import partial,reduce
from operator import add
from typing import Callable, List, Dict, Any
from .ranges import Range
from .tools import *
from . import vocabulary
from . import periodicities


_CACHE: dict[tuple[uuid.UUID,pd.Range],np.array] = {}

class ColumnError(Exception):
    pass

@dataclass(repr=False)
class Column:
    header : str
    word_name: str
    function: Callable[[Range,Dict[str, Any],List[Column]], np.ndarray]
    args: Dict[str, Any] = field(default_factory=dict)
    stack: list[Column] = field(default_factory=list)
    id_: uuid.UUID = field(default_factory=uuid.uuid4)
    no_cache: bool = False

    def __getitem__(self, range_: Range) -> np.ndarray:
        try:
            if False:
                d =self.function(range_, self.args, self.stack) 
                repr_ = f'.{self.word_name}'
                if len(self.args) == 1:
                    repr_ += '(' + ', '.join([f'{repr(v)}' for k,v in self.args.items()]) + ')'
                if len(self.args) > 1:
                    repr_ += '(' + ', '.join([f'{k}={repr(v)}' for k,v in self.args.items()]) + ')'
                if len(d.shape) > 1:
                    d2 = d[:,0]
                else:
                    d2 = d
                print(repr_)
                print(pd.Series(d2, index=range_.to_index()))
                return d
            if self.no_cache:
                return self.function(range_, self.args, self.stack)
            if not (self.id_, range_) in _CACHE:
                _CACHE[(self.id_, range_)] = self.function(range_, self.args, self.stack)
            return _CACHE[(self.id_, range_)]
        except Exception as e:
            if not isinstance(e, ColumnError):
                raise ColumnError(f'{type(e).__name__}: {e} getting {range_} for' \
                        + f'column "{self.header}" ({repr(self)})') from e
            else:
                raise e

    def __repr__(self):
        repr_ = 'pt'
        if len(self.stack) > 0:
            for column in self.stack:
                repr_ += '.q('
                repr_ += repr(column)
                repr_ += ')'
            repr_ += f'.cleave({len(self.stack)})'
        repr_ += f'.{self.word_name}'
        if len(self.args) == 1:
            repr_ += '(' + ', '.join([f'{repr(v)}' for k,v in self.args.items()]) + ')'
        if len(self.args) > 1:
            repr_ += '(' + ', '.join([f'{k}={repr(v)}' for k,v in self.args.items()]) + ')'
        return repr_

def quotation_col(range_, args, _):
    return np.full(len(range_), np.nan)

@dataclass(repr=False)
class QuotationColumn(Column):
    header: str = 'quotation'
    word_name: str = 'q'
    function: Callable[[Range,Dict[str, Any],List[Column]], np.ndarray] = \
                quotation_col

    def __repr__(self):
        return f'pt.q({self.args["quoted"]})'

def _resolve(name: str) -> Word:
    if re.match('f\d[_\d]*',name) is not None:
        return Constant()(float(name[1:].replace('_','.')))
    elif re.match('f_\d[_\d]*',name) is not None:
        return Constant()(-float(name[2:].replace('_','.')))
    elif re.match('i\d+',name) is not None:
        return Constant()(int(name[1:].replace('_','.')))
    elif re.match('i_\d+',name) is not None:
        return Constant()(-int(name[2:].replace('_','.')))
    elif re.match('r\d+_\d+',name) is not None:
        start, end = name[1:].split('_')
        return ConstantRange()(start,end)
    elif re.match('r\d+',name) is not None:
        return ConstantRange()(name[1:])
    elif name in vocabulary._all_set:
        return getattr(vocabulary, name)()
    else:
        raise AttributeError


@dataclass(repr=False)
class Word:
    name: str
    next_: Word = None
    prev: Word = None
    args: dict = None
    open_quotes: int = 0
    _stack: Optional[List[Column]] = None

    @property
    def __(self):
        raise Exception('Deprecated')


    def __add__(self, other: Word, copy_left=True) -> Word:
        this = self.copy_expression()
        if copy_left:
            other_tail = other.copy_expression()
        else:
            other_tail = other
        other_head = other_tail._head()[0]
        this.next_ = other_head
        other_head.prev = this
        if this.open_quotes != 0:
            current = other_head
            while current.next_ is not None:
                current.open_quotes += this.open_quotes
                current = current.next_
            current.open_quotes += this.open_quotes
        return other_tail

    def concat(self, other:Word) -> Word:
        return self.__add__(other)

    def __getattr__(self, name):
        try:
            word = _resolve(name)
            return self.__add__(word, copy_left=False)
        except:
            return self.__getattribute__(name)

    @property
    def p(self):
        assert self.open_quotes > 0, 'No quote to close.'
        current = self
        open_quotes = self.open_quotes
        while current.prev is not None and current.prev.open_quotes == open_quotes:
            current.open_quotes = 0
            current = current.prev
        current.open_quotes -= 1 
        assert isinstance(current, Quotation), 'something wrong'
        current.next_.prev = None
        current = current(current.next_._tail())
        current.next_ = None
        return current


    def __dir__(self):
        return sorted(set(dir(vocabulary)))

    def __getitem__(self, key: Any) -> pd.DataFrame:
        if isinstance(key, Range):
            range_ = key 
        elif isinstance(key, int):
            range_ = Range(key)
        else:
            range_ = Range.from_indexer(key)
        if self._stack is None:
            self.evaluate()
        if len(self._stack) == 0:
            return None
        values = np.column_stack([col[range_] for col in self._stack])
        _CACHE.clear()
        return pd.DataFrame(values, columns=[col.header for col in self._stack], index=range_.to_index())

    @property
    def columns(self):
        if self._stack is None:
            self.evaluate()
        return [col.header for col in self._stack]

    @property
    def stack(self):
        if self._stack is None:
            self.evaluate()
        return self._stack

    def evaluate(self, stack: Optional[List[Column]] = None) -> None:
        assert self.open_quotes == 0, 'Unclosed quotation.  Cannot evaluate'
        start = self._head()[0]
        current = start
        if stack is None or len(stack) == 0:
            if stack is None:
                stack = []
            while True:
                if current._stack is not None:
                    stack.extend(copy.copy(current._stack))
                    start = current.next_
                if current.next_ is None:
                    break
                current = current.next_
            keep = True
        else:
            keep = False
        current = start
        while current is not None:
            try:
                if current.args is None:
                    # needed to initialize defaults if no args   
                    current = current()
                current.operate(stack)
            except Exception as e:
                traceback.print_exc()
                error_msg = f' in word "{current.__str__()}"'
                if current.prev is not None:
                    error_msg += ' preceded by "'
                    prev_expr = repr(current.prev)
                    if len(prev_expr) > 150:
                        error_msg += '...'
                    error_msg += prev_expr[-150:] + '"'
                import sys
                raise type(e)((str(e) + error_msg).replace("'",'"')).with_traceback(sys.exc_info()[2])

                raise SyntaxError(repr(e) + error_msg) from e
            current = current.next_
        if keep:
            self._stack = copy.copy(stack)

    def __str__(self):
        if self.args and 'quoted' in self.args:
            return f'quote({self.args["quoted"].__repr__()})'
        else:
            s = self.name
            if self.args:
                str_args = []
                for k,v in self.args.items():
                    if k != self:
                        if isinstance(v, str):
                            str_args.append(f"{k}='{v[:2000]}'")
                            if len(v) > 2000:
                                str_args[-1] += '...'
                        else:
                            str_args.append(f"{k}={str(v)}")
                s += '(' + ', '.join(str_args) + ')'
        return s

    def __repr__(self):
        s = ''
        current = self
        while True:
            s = current.__str__() + s
            if current.prev is None:
                break
            else:
                s = '.' + s
                current = current.prev
        s = 'pt.' + s
        return s

    def __len__(self):
        return self._head()[1]

    def __call__(self, args = {}) -> Word:
        this = self.copy_expression()
        for name in ['__class__','self']:
            args.pop(name, None)
        this.args = args
        return this

    def operate(self, stack):
        pass
    
    def __copy__(self) -> Word:
        cls = self.__class__
        copied = cls.__new__(cls)
        copied.__dict__.update(self.__dict__)
        return copied

    def copy_expression(self) -> Word:
        first = copy.copy(self)
        first._stack = None
        current = first
        while current.prev is not None:
            prev = copy.copy(current.prev)
            current.prev = prev
            prev.next_ = current
            current = prev
        return first

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

    @property
    def local(self) -> Word:
        return Quotation()(self).call(0)

@dataclass(repr=False)
class Quotation(Word):
    name: str = 'quotation'
    
    def __call__(self, quoted):
        this = self.copy_expression()
        this.args = {'quoted': quoted}
        return this

    def __getattr__(self, name):
        try:
            word = _resolve(name)
            if self.args is None:
                self.open_quotes += 1 
            return self.__add__(word, False)
        except:
            return self.__getattribute__(name)

    def __add__(self, other: Word, copy_left=True) -> Word:
        if self.args is None and copy_left:
            self.open_quotes += 1 
        return super().__add__(other, copy_left)


    def operate(self, stack):
        stack.append(QuotationColumn(args={'quoted': self.args['quoted']}))

@dataclass(repr=False)
class BaseWord(Word):
    name: str
    operate: Callable[[List[Column]],None]

def peek_col(range_, args, stack, output):
    values = stack[0][range_]
    output[args['col']] = pd.Series(values, index=range_.to_index(), name=args['header']) \
            
    return values

PEEK = {}

def get_peek(name):
    return pd.concat(PEEK[name], axis=1)

@dataclass(repr=False)
class Peek(Word):
    name: str = 'peek'

    def __call__(self, name):
        return super().__call__(locals())

    def operate(self, stack):
        global PEEK
        output = []
        PEEK[self.args['name']] = output
        old_stack = stack.copy()
        stack.clear()
        col_func = partial(peek_col, output=output)
        for i, col in enumerate(reversed(old_stack)):
            output.append(None)
            col_args = self.args.copy()
            col_args['col'] = i
            col_args['header'] = col.header
            stack.insert(0, Column(col.header, self.name, col_func, col_args, [col]))

def const_col(range_, args, _):
    return np.full(len(range_), args['values'])

@dataclass(repr=False)
class Constant(Word):
    name: str = 'c'

    def __call__(self, *values):
        return super().__call__(locals())

    def __str__(self):
        output = []
        for value in self.args['values']:
            if isinstance(value, int):
                output.append('i' + str(value).replace('-','_'))
            elif isinstance(value, float):
                output.append('f' + str(value).replace('-','_').replace('.','_'))
        return '.'.join(output)

    def operate(self, stack):
        for value in self.args['values']:
            stack.append(Column('c', 'c', const_col, {'values': value}, no_cache=True))

def timestamp_col(range_, args, _):
    return np.array(range_.to_index().view('int'))

def daycount_col(range_, args, _):
    return np.array(range_.day_counts())

@dataclass(repr=False)
class ConstantRange(Word):
    name: str = 'c_range'

    def __call__(self, end, start=0):
        return super().__call__(locals())

    def __str__(self):
        return f'r{self.args["start"]}_{self.args["end"]}'

    def operate(self, stack):
        for value in range(int(self.args['start']),int(self.args['end'])):
            stack.append(Column('c', 'c', const_col, {'values': value}, no_cache=True))

def frame_col(range_, args, stack):
    col = stack[0]
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

@dataclass(repr=False)
class Pandas(Word):
    name: str = 'pandas'
    def __call__(self, frame_or_series): return super().__call__(locals())

    def operate(self, stack):
        frame = self.args['frame_or_series']
        if isinstance(frame, pd.core.series.Series):
            frame = frame.toframe()
        if frame.index.freq is None:
            frame.index.freq = pd.infer_freq(frame.index)
        for header, col in frame.items():
            stack.append(Column(header, self.name, frame_col, self.args, [col], no_cache=True))

@dataclass(repr=False)
class CSV(Word):
    name: str = 'csv'

    def __call__(self, csv_file, index_col=0, header='infer'):
        return super().__call__(locals())

    def operate(self, stack):
        frame = pd.read_csv(self.args['csv_file'], index_col=self.args['index_col'],
                                    header=self.args['header'], parse_dates=True)
        if frame.index.freq is None:
            frame.index.freq = pd.infer_freq(frame.index)
        for header, col in frame.items():
            stack.append(Column(header, self.name, frame_col, self.args, [col], no_cache=True))

def unary_operator_col(range_, args, stack, op):
    return op(stack[0][range_])

def binary_operator_col(range_, args, stack, op):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lhs = stack[0][range_]
        rhs = stack[1][range_]
        return np.where(np.logical_or(np.isnan(lhs), np.isnan(rhs)), np.nan, op(lhs,rhs))

def unary_operator_stack_function(stack, name, op):
    col = stack.pop()
    stack.append(Column(col.header, name, partial(unary_operator_col, op=op),
                    {}, [col], no_cache=True))

def binary_operator_stack_function(stack, name, op):
    assert len(stack) > 1, 'Binary function requires two columns'
    col2 = stack.pop()
    col1 = stack.pop()
    stack.append(Column(col1.header, name, partial(binary_operator_col, op=op),
                    {},  [col1, col2], no_cache=True))

def get_unary_operator(name, op):
    return BaseWord(name, operate=partial(unary_operator_stack_function, name=name, op=op))

def get_binary_operator(name, op):
    return BaseWord(name, operate=partial(binary_operator_stack_function, name=name, op=op))

@dataclass(repr=False)
class BinaryOperator(Word):
    name: str
    operation: Union[np.ufunc,Callable[[np.ndarray,np.ndarray],np.ndarray]] = None

    def __call__(self, operand: float = None):
        return super().__call__(locals())

    def operate(self, stack):
        if self.args['operand'] is not None:
            col2 = Column('c', 'c', const_col,
                        {'values': self.args['operand']}, no_cache=True)
        else:
            col2 = stack.pop()
        col1 = stack.pop()
        stack.append(Column(col1.header, self.name,
                    partial(binary_operator_col, op=self.operation),
                    {},  [col1, col2], no_cache=True))

def zero_first_op(x):
    x = x.copy()
    x[0] = 0.0
    return x

def zero_to_na_op(x):
    return np.where(np.equal(x,0),np.nan,x)

def is_na_op(x):
    return np.where(np.isnan(x), 1, 0)

def logical_not_op(x):
    return np.where(np.logical_not(x), 1, 0)

def logical_and_op(x,y):
    return np.where(np.logical_and(x,y), 1, 0)

def logical_or_op(x,y):
    return np.where(np.logical_or(x,y), 1, 0)

def logical_xor_op(x,y):
    return np.where(np.logical_xor(x,y), 1, 0)

# Stack manipulation
def dup_stack_function(stack):
    stack.append(stack[-1])

def roll_stack_function(stack):
    stack.insert(0,stack.pop())

def swap_stack_function(stack):
    stack.insert(-1,stack.pop())

def drop_stack_function(stack):
    stack.pop()

def rev_stack_function(stack):
    stack.reverse()

def clear_stack_function(stack):
    stack.clear()

def hsort_stack_function(stack):
    stack.sort(key=lambda c: c.header)

@dataclass(repr=False)
class Interleave(Word):
    name: str = 'interleave'

    def __call__(self, count=None, split_into=2): return super().__call__(locals())

    def operate(self, stack):
        if len(stack) == 0: return
        count = self.args['count'] if self.args['count'] else len(stack) // self.args['split_into']
        last = 0
        lists = []
        for i in range(len(stack)+1):
            if i % count == 0 and i != 0:
                lists.append(stack[i-count:i])
                last = i
        del(stack[:last])
        stack += [val for tup in zip(*lists) for val in tup]

@dataclass(repr=False)
class Pop(Word):
    name: str = 'pop'

    def __call__(self, count=1): return super().__call__(locals())

    def operate(self, stack):
        del(stack[-abs(int(self.args['count'])):])

@dataclass(repr=False)
class Top(Word):
    name: str = 'top'

    def __call__(self, count=1): return super().__call__(locals())

    def operate(self, stack):
        del(stack[:-abs(int(self.args['count']))])

@dataclass(repr=False)
class Pull(Word):
    name: str = 'pull'

    def __call__(self, start, end=None, clear=False): return super().__call__(locals())

    def operate(self, stack):
        end = -self.args['start'] - 1 if self.args['end'] is None else -self.args['end']
        start = len(stack) if self.args['start'] == 0 else -self.args['start']
        pulled = stack[end:start]
        if self.args['clear']:
            del(stack[:])
        else:
            del(stack[end:start])
        stack += pulled

@dataclass(repr=False)
class HeaderPull(Word):
    name: str = 'hpull'

    def __call__(self, *headers, clear=False, exact_match=False):
        return super().__call__(locals())
    def operate(self, stack):
        filtered_stack = []
        for header in self.args['headers']:
            to_del = []
            matcher = lambda c: header == c.header if self.args['exact_match'] else re.match(header,col.header) is not None
            for i,col in enumerate(stack):
                if matcher(col):
                    filtered_stack.append(stack[i])
                    to_del.append(i)
            to_del.sort(reverse=True)
            for i in to_del:
                del(stack[i])
        if self.args['clear']:
            del(stack[:])
        stack += filtered_stack

@dataclass(repr=False)
class HeaderFilter(HeaderPull):
    name: str = 'hfilter'
    def __call__(self, *headers, clear=True, exact_match=False):
        return super().__call__(*headers, clear=True, exact_match=False)

def ewma_col(range_, args, stack):
    alpha = 2 /(args['window'] + 1.0)
    data = stack[0][range_]
    idx = np.cumsum(np.where(~np.isnan(data),1,0)) - 1
    nans = np.where(~np.isnan(data),1,np.nan)
#     starting_nans = np.where(idx == -1,np.nan,1)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.full(idx.shape[0],np.nan)
    out = ewma_vectorized_safe(data,alpha)
    #return out[idx] * starting_nans
    return out[idx] * nans

@dataclass(repr=False)
class EWMA(Word):
    name: str = 'ewma'
    def __call__(self, window, fill_nans=True): return super().__call__(locals())
    def operate(self, stack):
        col = stack.pop()
        stack.append(Column(col.header,'ewma', ewma_col, self.args, [col]))

# Combinators
@dataclass(repr=False)
class Call(Word):
    name: str = 'call'
    def __call__(self, depth=None, copy=False): return super().__call__(locals())
    def operate(self, stack):
        assert len(stack) == 0 or not isinstance(stack[-1],Quotation) , 'call needs a quotation on top of stack'
        quoted = stack.pop().args['quoted']
        if 'depth' not in self.args or self.args['depth'] is None:
            depth = len(stack) 
        else:
            depth = self.args['depth']

        if depth != 0:
            this_stack = stack[-depth:]
            if 'copy' not in self.args or not self.args['copy']:
                del(stack[-depth:])
        else:
            this_stack = []
        quoted.evaluate(this_stack)
        stack.extend(this_stack)

@dataclass(repr=False)
class IfExists(Word):
    name: str = 'ifexists'
    def __call__(self, count=1, else_=False, copy=False): return super().__call__(locals())
    def operate(self, stack):
        assert len(stack) == 0 or not isinstance(stack[-1],Quotation) , 'ifexists needs a quotation on top of stack'
        quoted = stack.pop().args['quoted']
        if self.args['else_']:
            assert len(stack) == 0 or not isinstance(stack[-1],Quotation) , 'ifexists needs two quotations on top of stack for else_=True'
            quoted_else = stack.pop().args['quoted']
        if len(stack) >= self.args['count']:
            quoted.evaluate(stack)
        elif self.args['else_']:
            quoted_else.evaluate(stack)

@dataclass(repr=False)
class If(Word):
    name: str = 'if_'
    def __call__(self, condition: Callable[[list[str]],bool]): return super().__call__(locals())
    def operate(self, stack):
        assert len(stack) == 0 or not isinstance(stack[-1],Quotation) , 'if_ needs a quotation on top of stack'
        quoted = stack.pop().args['quoted']
        if self.args['condition']([col.header for col in stack]):
            quoted.evaluate(stack)

@dataclass(repr=False)
class IfElse(Word):
    name: str = 'ifelse'
    def __call__(self, condition: Callable[[list[str]],bool]): return super().__call__(locals())
    def operate(self, stack):
        assert len(stack) < 2 or \
                not isinstance(stack[-1],Quotation) or \
                not isinstance(stack[-2],Quotation) \
                , 'if_ needs a quotation on top of stack'
        quoted = stack.pop().args['quoted']
        else_quoted = stack.pop().args['quoted']
        if self.args['condition']([col.header for col in stack]):
            quoted.evaluate(stack)
        else:
            else_quoted.evaluate(stack)

def partial_stack_function(stack, partial_stack):
    stack.extend(partial_stack)

@dataclass(repr=False)
class Partial(Word):
    name: str = 'partial'
    def __call__(self, depth=1, copy=False): return super().__call__(locals())
    def operate(self, stack):
        assert len(stack) == 0 or not isinstance(stack[-1],Quotation) , 'partial needs a quotation on top of stack'
        quote = stack.pop()
        depth = self.args['depth']
        if depth != 0:
            this_stack = stack[-depth:]
            if not self.args['copy']:
                del(stack[-depth:])
        else:
            this_stack = []
        stack_function = partial(partial_stack_function, partial_stack=this_stack)
        quote.args['quoted'] = BaseWord('partial', operate=stack_function) + quote.args['quoted']
        stack.append(quote)

@dataclass(repr=False)
class Compose(Word):
    name: str = 'compose'
    def __call__(self, num_quotations=0): return super().__call__(locals())
    def operate(self, stack):
        count = self.args['num_quotations']
        if self.args['num_quotations'] == 0:
            while isinstance(stack[-count-1],QuotationColumn):
                count +=1
        quoted = reduce(add, [quote.args['quoted'] for quote in stack[-count:]])
        del(stack[-count:])
        stack.append(QuotationColumn(args={'quoted': quoted}))

@dataclass(repr=False)
class Each(Word):
    name: str = 'each'
    def __call__(self, start=0, end=None, every=1, copy=False): return super().__call__(locals())
    def operate(self, stack):
        assert isinstance(stack[-1], QuotationColumn)
        quote = stack.pop().args['quoted']
        end = 0 if self.args['end'] is None else -self.args['end']
        start = len(stack) if self.args['start'] == 0 else -self.args['start']
        selected = stack[end:start]
        assert len(selected) % self.args['every'] == 0, f'Stack length {len(selected)} not evenly divisible by every {self.args["every"]}'
        if not self.args['copy']:
            del(stack[end:start])
        for t in zip(*[iter(selected)]*self.args['every']):
            this_stack = list(t)
            quote.evaluate(this_stack)
            stack += this_stack
@dataclass(repr=False)
class Repeat(Word):
    name: str = 'repeat'
    def __call__(self, times): return super().__call__(locals())
    def operate(self, stack):
        assert len(stack) == 0 or not isinstance(stack[-1],Quotation) , 'repeat needs a quotation on top of stack'
        quote = stack.pop().args['quoted']
        for _ in range(self.args['times']):
            quote.evaluate(stack)

def heach_stack_function(stack):
    assert stack[-1].header == 'quotation'
    quote = stack.pop().args['quoted']
    new_stack = []
    for header in set([c.header for c in stack]):
        to_del, filtered_stack = [], []
        for i,col in enumerate(stack):
            if header == col.header:
                filtered_stack.append(stack[i])
                to_del.append(i)
        quote.evaluate(filtered_stack)
        new_stack += filtered_stack
        to_del.sort(reverse=True)
        for i in to_del:
            del(stack[i])
    del(stack[:])
    stack += new_stack

@dataclass(repr=False)
class Cleave(Word):
    name: str = 'cleave'
    def __call__(self, num_quotations=0, depth=None, copy=False): return super().__call__(locals())
    def operate(self, stack):
        count = self.args['num_quotations']
        if self.args['num_quotations'] == 0:
            while isinstance(stack[-count-1],QuotationColumn):
                count +=1
        quotes = [quote.args['quoted'] for quote in stack[-count:]]
        del(stack[-count:])
        depth = len(stack) if self.args['depth'] is None else self.args['depth']
        copied_stack = stack[-depth:] if depth != 0 else []
        if not self.args['copy'] and depth != 0:
            del(stack[-depth:])
        for quote in quotes:
            this_stack = copied_stack[:]
            quote.evaluate(this_stack)
            stack += this_stack

# Header
def header_col(range_, args, stack):
    return stack[0][range_]

@dataclass(repr=False)
class HeaderSet(Word):
    name: str = 'hset'
    def __call__(self, *headers): return super().__call__(locals())
    def operate(self, stack):
        headers = self.args['headers']
        if len(headers) == 1 and headers[0].find(',') != -1:
            headers = headers[0].split(',')
        start = len(stack) - len(headers)
        for i in range(start,len(stack)):
            header = headers[i - start]
            stack[i] = Column(header, self.name, header_col,
                                {'headers': header}, [stack[i]], no_cache=True)

def hcopy_stack_function(stack):
    col1 = stack.pop()
    col2 = stack.pop()
    stack.append(col2)
    stack.append(Column(col2.header, 'hcopy', header_col, {}, [col1], no_cache=True))

@dataclass(repr=False)
class HeaderFormat(Word):
    name: str = 'hformat'
    def __call__(self, format_string): return super().__call__(locals())
    def operate(self, stack):
        col = stack.pop()
        header = self.args['format_string'].format(col.header)
        stack.append(Column(header, self.name, header_col, self.args, [col], no_cache=True))

@dataclass(repr=False)
class HeaderReplace(Word):
    name: str = 'hreplace'
    def __call__(self, old, new): return super().__call__(locals())
    def operate(self, stack):
        col = stack.pop()
        header = col.header.replace(self.args['old'],self.args['new'])
        stack.append(Column(header, self.name, header_col, self.args, [col], no_cache=True))

@dataclass(repr=False)
class HeaderApply(Word):
    name: str = 'happly'
    def __call__(self, header_function): return super().__call__(locals())
    def operate(self, stack):
        col = stack.pop()
        header = self.args['header_function'](col.header)
        stack.append(Column(header, self.name, header_col, {}, [col], no_cache=True))

# Windows
def rolling_col(range_, args, stack):
    col = stack[0]
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
        expanded_range = Range.change_periodicity(range_, periodicity)
    expanded_range.start = expanded_range.start - lookback
    expanded = col[expanded_range]
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
                    index=expanded_range.to_index()).reindex(range_.to_index()).values
@dataclass(repr=False)
class Rolling(Word):
    name: str = 'rolling'
    def __call__(self, window=2, exclude_nans=True, periodicity=None, lookback_multiplier=2): return super().__call__(locals())
    def operate(self, stack):
        col = stack.pop()
        stack.append(Column(col.header, self.name, rolling_col, self.args, [col]))

def expanding_col(range_, args, stack):
    start_date=args['start_date']
    col = stack[0]
    index = range_.periodicity.get_index(start_date)
    if range_.start is None:
        range_.start = index
        return col[range_]
    expanded_range = copy.copy(range_)
    offset = expanded_range.start - index
    expanded_range.start = index
    values = col[expanded_range].view(ma.MaskedArray)
    values[:offset] = ma.masked
    if range_.stop is None:
        range_.stop = expanded_range.stop
    return values

@dataclass(repr=False)
class Expanding(Word):
    name: str = 'expanding'
    def __call__(self, start_date): return super().__call__(locals())
    def operate(self, stack):
        col = stack.pop()
        stack.append(Column(col.header, self.name, expanding_col, self.args, [col]))

def crossing_col(range_, args, stack):
    return np.column_stack([col[range_] for col in stack])

def crossing_op(stack):
    cols = stack[:]
    stack.clear()
    stack.append(Column(cols[0].header, 'crossing', crossing_col, {}, cols))

def rev_expanding_col(range_, args, stack):
    return stack[0][range_][::-1]

def rev_expanding_op(stack):
    col = stack.pop()
    stack.append(Column(col.header, 'rev_expanding', rev_expanding_col, {}, [col]))

def cumsum_col(range_, args, stack):
    v = stack[0][range_]
    nans = np.isnan(v)
    cumsum = np.nancumsum(v)
    v[nans] = -np.diff(np.concatenate([[0.],cumsum[nans]]))
    return np.cumsum(v)

def cumsum_stack_function(stack):
    col = stack.pop()
    stack.append(Column(col.header, 'cumsum', cumsum_col, {}, [col]))

# Data cleaning
def fill_col(range_, args, stack):
    x = stack[0][range_].copy()
    x[np.isnan(x)] = args['value']
    return x

@dataclass(repr=False)
class Fill(Word):
    name: str = 'fill'
    def __call__(self, value): return super().__call__(locals())
    def operate(self, stack):
        col = stack.pop()
        stack.append(Column(col.header, self.name, fill_col, self.args, [col], no_cache=True))

def ffill_col(range_, args, stack):
    lookback = abs(args['lookback']) 
    expanded_range = copy.copy(range_)
    expanded_range.start -= lookback
    x = stack[0][expanded_range]
    idx = np.where(~np.isnan(x),np.arange(len(x)),0)
    np.maximum.accumulate(idx,out=idx)
    filled = x[idx][lookback:]
    if args['leave_end']:
        print(len(x))
        print(idx.max())
        if len(x) > idx.max() + 1:
            at_end = len(x) - idx.max() - 1
            filled[-at_end:] = np.nan
    return filled

@dataclass(repr=False)
class FFill(Word):
    name: str = 'ffill'
    def __call__(self, lookback=0, leave_end=False): return super().__call__(locals())
    def operate(self, stack):
        if len(stack) == 0: return
        col = stack.pop()
        stack.append(Column(col.header, self.name, ffill_col, self.args, [col]))

def join_col(range_, args, stack):
    date = args['date']
    if not isinstance(date, int):
        date = range_.periodicity.get_index(date)
    if range_.stop is not None and range_.stop < date:
        return stack[0][range_]
    if range_.start and range_.start >= date:
        return stack[1][range_]
    r_first = copy.copy(range_)
    r_first.stop = date
    r_second = copy.copy(range_)
    r_second.start = date
    v_first = stack[0][r_first]
    v_second = stack[1][r_second]
    if range_.stop is None:
        range_.stop = r_second.stop
    if range_.start is None:
        range_.start = r_first.start
    return np.concatenate([v_first, v_second])

@dataclass(repr=False)
class Join(Word):
    name: str = 'join'
    def __call__(self, date): return super().__call__(locals())
    def operate(self, stack):
        col2 = stack.pop()
        col1 = stack.pop()
        stack.append(Column(col1.header, self.name, join_col, self.args, [col1, col2], no_cache=True))


def window_operator_col(range_, args, stack, matrix_op, vector_op):
    values = stack[0][range_]
    if len(values.shape) == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.where(np.all(np.isnan(values),axis=1), np.nan,
                        matrix_op(values, axis=1))
    else:
        cum = vector_op(values, axis=None)
        if values.strides[0] < 0:
            return cum[::-1]
        if isinstance(values,ma.MaskedArray):
            return ma.array(cum,mask=values.mask).compressed()
        return cum 

def window_operator_stack_function(stack, name, matrix_op, vector_op):
    col = stack.pop()
    op = partial(window_operator_col, matrix_op=matrix_op, vector_op=vector_op) 
    stack.append(Column(col.header, name, op, {}, [col]))

def get_window_operator(name, matrix_op, vector_op):
    stack_function = partial(window_operator_stack_function, name=name, 
                                matrix_op=matrix_op, vector_op=vector_op) 
    return BaseWord(name, operate=stack_function)

def count_twod_op(x, axis):
    return np.sum(~np.isnan(x),axis=1)

def count_oned_op(x, axis):
    return np.cumsum(~np.isnan(x))

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

def firstvalid_oned_op(x, axis):
    valid = np.argwhere(~np.isnan(x))
    if len(valid) == 0:
        return np.full(x.shape, np.nan)
    where = np.min(valid)
    value = x[where]
    return np.concatenate([np.full(where,np.nan),np.full(len(x)-where,value)])

def firstvalid_twod_op(x, axis):
    return np.array([firstvalid_oned_op(row, axis) for row in x])


def last_twod_op(x, axis):
    return  x[:,-1]

def last_oned_op(x, axis):
    return  x

def _multicol_rows_function(range_: pt.Range,
                            args: Dict[str, Any],
                            stack: List[Column],
                            table_id: uuid.UUID,
                            table_function: Callable[[pt.Range, Dict[str, Any], List[Column]], np.ndarray],
                            ):#lock: AcquirerProxy) -> np.ndarray:
        if (table_id, range_) not in _CACHE:
            #print(f'computing table: {table_id}')
            _CACHE[(table_id, range_)] = table_function(range_, args, stack)
        return _CACHE[(table_id, range_)][:, args['column']]

@dataclass(repr=False)
class MultiColCachedWord(Word):
    name = None
    table_function: Callable[[pt.Range, Dict[str, Any], List[Column]], np.ndarray] = None
    table_columns: Callable[[Dict[str, Any], List[Column]], int] = None
    table_headers: Callable[[Dict[str, Any], List[Column], int], str] = None

    def operate(self, stack):
        inputs = stack.copy()
        stack.clear()
        row_function = partial(_multicol_rows_function, 
                table_id=uuid.uuid4(), table_function=self.table_function)
        for i in range(self.table_columns(self.args, inputs)):
            col_args = self.args.copy()
            col_args['column'] = i
            header = self.table_headers(col_args, inputs, i)
            stack.append(Column(header, self.name, row_function, col_args, inputs, no_cache=True))

def _rank_table_function(range_: pt.Range, args: Dict[str, Any], stack: List[Column]) -> np.ndarray:
    return pd.DataFrame(np.column_stack([c[range_] for c in stack])).rank(axis=1).values

def _rank_table_columns(args, stack):
    return len(stack)

def _rank_table_headers(args, stack, i):
    return stack[i].header

@dataclass(repr=False)
class Rank(MultiColCachedWord):
    def __post_init__(self):
        self.name = 'rank'
        self.table_function = _rank_table_function
        self.table_columns = _rank_table_columns
        self.table_headers = _rank_table_headers

    def __call__(self):
        return super().__call__()
