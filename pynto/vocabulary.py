import numpy as np
from .main import *
from .database import Saved

__all__ = [
'abs', 'add', 'annotations', 'begin', 'c', 'c_range',
'call', 'change', 'clear', 'cleave', 'copy', 'count',
'crossing', 'csv', 'cumsum', 'day_count', 'db', 'div',
'drop', 'dup', 'each', 'eq', 'ewma', 'exp', 'expanding',
'expanding_mean', 'expanding_std', 'expanding_var',
'ffill', 'fill', 'first', 'firstvalid', 'ge', 'gt',
'happly', 'heach', 'hfilter', 'hformat', 'hpull',
'hset', 'hsort', 'ifexists',
'interleave', 'inv', 'is_na', 'join', 'last', 'le',
'log', 'log_change', 'logical_and', 'logical_not',
'logical_or', 'logical_xor', 'lt', 'ma',
'max', 'mean', 'min', 'mod', 'mul', 'ne',
'neg', 'partial',
'pct_change', 'peek', 'pow', 'prod', 'pull',
'q', 'quote', 'rank', 'repeat', 'rev',
'rev_expanding', 'roll', 'rolling',
'sqrt', 'std',
'sub', 'sum', 'swap', 'timestamp', 'var',
'zero_first', 'zero_to_na', 'zscore']

def _define(name: str, word) -> None:
    globals()[name] = word
    __all__.append(name)

c = Constant()
timestamp = BaseWord('timestamp', operate=lambda stack: stack.append(Column('timestamp','timestamp', timestamp_col)))
day_count = BaseWord('day_count', operate=lambda stack: stack.append(Column('days','day_count', daycount_col)))
c_range = ConstantRange()
pandas = Pandas()
csv = CSV()
add = BinaryOperator('add', operation=np.add)
sub = BinaryOperator('sub', operation=np.subtract)
mul = BinaryOperator('mul', operation= np.multiply)
pow = BinaryOperator('pow', operation= np.power)
div = BinaryOperator('div', operation= np.divide)
mod = BinaryOperator('mod', operation= np.mod)
eq = BinaryOperator('eq', operation= np.equal)
ne = BinaryOperator('ne', operation= np.not_equal)
ge = BinaryOperator('ge', operation= np.greater_equal)
gt = BinaryOperator('gt', operation= np.greater)
le = BinaryOperator('le', operation= np.less_equal)
lt = BinaryOperator('lt', operation= np.less)
logical_and = BinaryOperator('logical_and', operation= logical_and_op)
logical_or = BinaryOperator('logical_or', operation= logical_or_op)
logical_xor = BinaryOperator('logical_xor', operation= logical_xor_op)

neg = get_unary_operator('neg', np.negative)
inv = get_unary_operator('inv', np.reciprocal)
abs = get_unary_operator('abs', np.abs)
sqrt = get_unary_operator('sqrt', np.sqrt)
exp = get_unary_operator('exp', np.exp)
log = get_unary_operator('log', np.log)
zero_to_na = get_unary_operator('zero_to_na', zero_to_na_op)
is_na = get_unary_operator('is_na', is_na_op)
zero_first = get_unary_operator('zero_first', zero_first_op)
logical_not = get_unary_operator('logical_not', logical_not_op)
roll = BaseWord('roll', operate=roll_stack_function)
swap = BaseWord('swap', operate=swap_stack_function)
drop = BaseWord('drop', operate=drop_stack_function)
dup = BaseWord('dup', operate=dup_stack_function)
rev = BaseWord('rev', operate=rev_stack_function)
clear = BaseWord('clear', operate=clear_stack_function)
hsort = BaseWord('hsort', operate=hsort_stack_function)
interleave = Interleave()
pull = Pull()
hpull = HeaderPull()
hfilter = HeaderFilter()
ewma = EWMA()
call = Call()
partial = Partial()
each = Each()
repeat = Repeat()
heach = BaseWord('heach',operate=heach_stack_function)
cleave = Cleave()
hset = HeaderSet()
hformat = HeaderFormat()
happly = HeaderApply()
rolling = Rolling()
expanding = Expanding()
crossing = BaseWord('crossing', operate=crossing_op)
rev_expanding = BaseWord('rev_expanding', operate=rev_expanding_op)
fill = Fill()
ffill = FFill()
join = Join()
cumsum = BaseWord('cumsum', operate=cumsum_stack_function)
sum = get_window_operator('sum',np.nansum, sum_oned_op)
count = get_window_operator('count',count_twod_op, count_oned_op)
max = get_window_operator('max',np.nanmax,max_oned_op)
min = get_window_operator('min',np.nanmin,min_oned_op)
prod = get_window_operator('prod',np.nanprod,prod_oned_op)
mean = get_window_operator('mean',np.nanmean, expanding_mean)
var = get_window_operator('var', np.nanvar, expanding_var)
std = get_window_operator('std', np.nanstd, expanding_var)
change = get_window_operator('change', change_twod_op, change_oned_op)
pct_change = get_window_operator('pct_change', pct_change_twod_op, pct_change_oned_op)
log_change = get_window_operator('log_change', log_change_twod_op, log_change_oned_op)
first = get_window_operator('first', first_twod_op, first_oned_op)
firstvalid = get_window_operator('firstvalid', firstvalid_twod_op, firstvalid_oned_op)
last = get_window_operator('last',last_twod_op,last_oned_op)
quote = Quotation()
q = Quotation()
begin = Word('')
peek = Peek()
ifexists = IfExists()
db = Saved()

zscore = quote(quote(std).quote(last).quote(mean).cleave(3, depth=1).sub.swap.div).call

rank = Rank('rank')
