from . import main
import numpy as np

__all__ = sorted(set([
'abs', 'add', 'annotations', 'begin', 'c', 'c_range',
'call', 'change', 'clear', 'cleave', 'compose', 'copy', 'count',
'crossing', 'csv', 'cumsum', 'day_count', 'db', 'div',
'drop', 'dup', 'each', 'eq', 'ewma', 'exp', 'expanding',
'expanding_mean', 'expanding_std', 'expanding_var',
'ffill', 'fill', 'first', 'firstvalid', 'ge', 'gt',
'happly','hcopy', 'heach', 'hfilter', 'hformat','hreplace','hpull',
'hset', 'hsort', 'ifexists','if_','ifelse',
'interleave', 'inv', 'is_na', 'join', 'last', 'le',
'log', 'log_change', 'logical_and', 'logical_not',
'logical_or', 'logical_xor', 'lt', 'ma','map',
'max', 'mean', 'min', 'mod', 'mul','nan', 'ne',
'neg', 'pandas', 'partial',
'pct_change', 'peek', 'pow', 'prod', 'pull','pop',
'q', 'quote', 'rank', 'repeat', 'rev',
'rev_expanding', 'roll', 'rolling',
'sqrt', 'std',
'sub', 'sum','sumnonan', 'swap', 'timestamp', 'top','var',
'zero_first', 'zero_to_na', 'zscore']))

_all_set = set(__all__)

def __dir__():
    return __all__

def _define(name: str, word) -> None:
    global __all__, _all_set
    __all__ = sorted(__all__ + [name])
    _all_set = set(__all__)
    globals()[name] = word

c = lambda: main.Constant()
nan = lambda: main.Constant()(np.nan)
timestamp = lambda: main.BaseWord('timestamp', operate=lambda stack: stack.append(main.Column('timestamp','timestamp', main.timestamp_col)))
day_count = lambda: main.BaseWord('day_count', operate=lambda stack: stack.append(main.Column('days','day_count', main.daycount_col)))
c_range = lambda: main.ConstantRange()
pandas = lambda: main.Pandas()
csv = lambda: main.CSV()
add = lambda: main.BinaryOperator('add', operation=np.add)
sub = lambda: main.BinaryOperator('sub', operation=np.subtract)
mul = lambda: main.BinaryOperator('mul', operation= np.multiply)
pow = lambda: main.BinaryOperator('pow', operation= np.power)
div = lambda: main.BinaryOperator('div', operation= np.divide)
mod = lambda: main.BinaryOperator('mod', operation= np.mod)
eq = lambda: main.BinaryOperator('eq', operation= np.equal)
ne = lambda: main.BinaryOperator('ne', operation= np.not_equal)
ge = lambda: main.BinaryOperator('ge', operation= np.greater_equal)
gt = lambda: main.BinaryOperator('gt', operation= np.greater)
le = lambda: main.BinaryOperator('le', operation= np.less_equal)
lt = lambda: main.BinaryOperator('lt', operation= np.less)
logical_and = lambda: main.BinaryOperator('logical_and', operation= logical_and_op)
logical_or = lambda: main.BinaryOperator('logical_or', operation= logical_or_op)
logical_xor = lambda: main.BinaryOperator('logical_xor', operation= logical_xor_op)

neg = lambda: main.get_unary_operator('neg', np.negative)
inv = lambda: main.get_unary_operator('inv', np.reciprocal)
abs = lambda: main.get_unary_operator('abs', np.abs)
sqrt = lambda: main.get_unary_operator('sqrt', np.sqrt)
exp = lambda: main.get_unary_operator('exp', np.exp)
log = lambda: main.get_unary_operator('log', np.log)
zero_to_na = lambda: main.get_unary_operator('zero_to_na', main.zero_to_na_op)
is_na = lambda: main.get_unary_operator('is_na', is_na_op)
zero_first = lambda: main.get_unary_operator('zero_first', main.zero_first_op)
logical_not = lambda: main.get_unary_operator('logical_not', main.logical_not_op)
roll = lambda: main.BaseWord('roll', operate=main.roll_stack_function)
swap = lambda: main.BaseWord('swap', operate=main.swap_stack_function)
drop = lambda: main.BaseWord('drop', operate=main.drop_stack_function)
dup = lambda: main.BaseWord('dup', operate=main.dup_stack_function)
rev = lambda: main.BaseWord('rev', operate=main.rev_stack_function)
clear = lambda: main.BaseWord('clear', operate=main.clear_stack_function)
hsort = lambda: main.BaseWord('hsort', operate=main.hsort_stack_function)
interleave = lambda: main.Interleave()
pull = lambda: main.Pull()
pop = lambda: main.Pop()
top = lambda: main.Top()
hpull = lambda: main.HeaderPull()
hfilter = lambda: main.HeaderFilter()
ewma = lambda: main.EWMA()
call = lambda: main.Call()
partial = lambda: main.Partial()
each = lambda: main.Each()
map = lambda: main.Each()
repeat = lambda: main.Repeat()
heach = lambda: main.BaseWord('heach',operate=main.heach_stack_function)
cleave = lambda: main.Cleave()
compose = lambda: main.Compose()
hset = lambda: main.HeaderSet()
hformat = lambda: main.HeaderFormat()
hreplace = lambda: main.HeaderReplace()
happly = lambda: main.HeaderApply()
hcopy = lambda: main.BaseWord('hcopy', operate=main.hcopy_stack_function)
rolling = lambda: main.Rolling()
expanding = lambda: main.Expanding()
crossing = lambda: main.BaseWord('crossing', operate=main.crossing_op)
rev_expanding = lambda: main.BaseWord('rev_expanding', operate=main.rev_expanding_op)
fill = lambda: main.Fill()
ffill = lambda: main.FFill()
join = lambda: main.Join()
cumsum = lambda: main.BaseWord('cumsum', operate=main.cumsum_stack_function)
sum = lambda: main.get_window_operator('sum',np.nansum, main.sum_oned_op)
count = lambda: main.get_window_operator('count',main.count_twod_op, main.count_oned_op)
max = lambda: main.get_window_operator('max',np.nanmax,main.max_oned_op)
min = lambda: main.get_window_operator('min',np.nanmin,main.min_oned_op)
prod = lambda: main.get_window_operator('prod',np.nanprod,main.prod_oned_op)
mean = lambda: main.get_window_operator('mean',np.nanmean, main.expanding_mean)
var = lambda: main.get_window_operator('var', np.nanvar, main.expanding_var)
std = lambda: main.get_window_operator('std', np.nanstd, main.expanding_std)
change = lambda: main.get_window_operator('change', main.change_twod_op, main.change_oned_op)
pct_change = lambda: main.get_window_operator('pct_change', main.pct_change_twod_op, main.pct_change_oned_op)
log_change = lambda: main.get_window_operator('log_change', main.log_change_twod_op, main.log_change_oned_op)
first = lambda: main.get_window_operator('first', main.first_twod_op, main.first_oned_op)
firstvalid = lambda: main.get_window_operator('firstvalid', main.firstvalid_twod_op, main.firstvalid_oned_op)
last = lambda: main.get_window_operator('last',main.last_twod_op,main.last_oned_op)
sumnonan = lambda: main.get_window_operator('sumnonan',np.sum, main.sumnonan_oned_op)
quote = lambda: main.Quotation()
q = lambda: main.Quotation()
begin = lambda: main.Word('')
peek = lambda: main.Peek()
ifexists = lambda: main.IfExists()
if_ = lambda: main.If()
ifelse = lambda: main.IfElse()

zscore = lambda: quote()(quote()(std()).quote(last()).quote(mean()).cleave(3, depth=1).sub.swap.div).call()

rank = lambda: main.Rank('rank')
