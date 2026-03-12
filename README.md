![pynto logo](resources/pynto.png)

## pynto: Data analysis in Python using stack-based programming

pynto is a Python package that lets you manipulate a data frame as a stack of columns, using the the expressiveness of the [concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language)/[stack-oriented](https://en.wikipedia.org/wiki/Stack-oriented_programming) paradigm.  

## How does it work?

With pynto you chain together functions called _words_ to formally specify how to calculate each column of your data frame.  The composed _words_ can be lazily evaluated over any range of rows to create your data frame.

_Words_ add, remove or modify columns.  They can operate on the entire stack or be limited to a certain columns using a _column indexer_.  Composed _words_ will operate in left-to-right order, with operators following their operands in _postfix_ (Reverse Polish Notation) style.  More complex operations can be specified using _quotations_, anonymous blocks of _words_ that do not operate immediately, and _combinators_, higher-order words that control the execution of _quotations_.


## What does it look like?
Here's a program to calculate deviations from moving average for each column in a table using the _combinator_/_quotation_ pattern.
```
>>> import pynto as pt 
>>> ma_dev = (                        # create a pynto expression by concatenating words to
>>>     pt.load('stock_prices')      # append columns to stack from the build-in database
>>>     .q                            # start a quotation 
>>>         .dup                      # push a copy of the top (leftmost) column of the stack
>>>         .ravg(20)                 # calculate 20-period moving average
>>>         .sub                      # subtract top column from second column 
>>>     .p                            # close the quotation
>>>     .map                          # use the map combinator to apply the quotation
>>> )                                 # to each column in the stack
>>>
>>> df = ma_dev.rows['2021-06-01':]         # evaluate over a range of rows to get a DataFrame
>>> pt.db['stocks_ma_dev'] = df             # save the results back to the database   
```

## Why pynto?
 - Expressive: Pythonic syntax; Combinatory logic for modular, reusable code 
 - Performant: Memoization to eliminate duplicate operations
 - Batteries included:  Built-in time series database
 - Interoperable: Seemlessly integration with Pandas/numpy

## Get pynto
```
pip install pynto
```

## Reference

### The Basics

## Constant literals
Add constant-value columns to the stack using literals that start with `c`, followed by a number with `-` and `.` characters replaced by `_`.  `r`_n_ adds whole number-value constant columns up to _n - 1_.
```
>>> # Compose _words_ that add a column of 10s to the stack, duplicate the column, 
>>> # and then multiply the columns together
>>> ten_squared = pt.c10_0.dup.mul         
```

## Row indexers
To evaluate your expression, you use a row indexer.  Specify rows by date range using the `.rows[`_start_`:`_stop (exclusive)_`:`_periodicity_`]` syntax. None slicing arguments default to the widest range available.  _int_ indices also work with the `.rows` indexer. `.first`, and `.last` are included for convenience.
```
>>> ten_squared.rows['2021-06-01':'2021-06-03','B']                   # evaluate over a two business day date range                                                   
                 c
2021-06-01     100.0
2021-06-02     100.0
```

## Quotations and Combinators
Combinators are higher-order functions that allow pynto to do more complicated things like branching and looping.  Combinators operate on quotations, expressions that are pushed to the stack instead of operating on the stack.  To push a quotation to the stack, put words in between `q` and `p` (or put an expression in the local namespace within the parentheses of `pt.q(_expression_)`).  THe `map` combinator evaluated a quotation at the top of the stack over each column below in the stack.
```
>>> pt.c9.c10.q.dup.mul.p.map.last
                 c         c
2021-06-02      81.0     100.0
```

## Headers
Each column has a string header.  `hset` sets the header to a new value.  Headers are useful for filtering or arranging columns.
```
>>> pt.c9.c10.q.dup.mul.p.map.hset('a','b').last
                 a         b
2021-06-02      81.0     100.0
```

## Column indexers
Column indexers specify the columns on which a _word_ operates, overiding the _word's_ default.  Postive _int_ indices start from the bottom (left) of the stack and negative indices start from the top.

By default `add` has a column indexer of [-2:]
```
>>> pt.r5.add.last
              c    c    c    c
2021-06-02  0.0  1.0  2.0  7.0
```
Change the column indexer of `add` to [:] to sum all columns
```
>>> pt.r5.add[:].last
               c
2025-06-02  10.0
```
You can also index columns by header, using regular expressions
```
>>> pt.r3.hset('a,b,c').add['(a|c)'].last
              b    a
2025-06-02  1.0  2.0
```

## Defining words
_Words_ in the local namespace can be composed using the  `+` operator.  
```
>>> squared = pt.dup.mul
>>> ten_squared2 = pt.c10_0 + squared    # same thing
```

_Words_ can also be defined globally in the pynto vocabulary.
```
>>> pt.define['squared'] = pt.dup.mul
>>> ten_squared3 = pt.c10_0.squared    # same thing
```


### The Database

pynto has built-in database functionality that lets you save DataFrames and Series to a Redis database.  The database saves the underlying numpy data in native byte format for zero-copy retrieval.   Each DataFrame column is saved as an independent key and can be retrieved or updated on its own.  The database also supports three-dimensional frames that have a two-level MultiIndex.

```
>>> pt.db['my_df'] = expr.rows['2021-06-01':'2021-06-03']
>>> pt.load('my_df').rows[:]
              constant  constant
2021-06-01      81.0     100.0
2021-06-02      81.0     100.0
```

## pynto built-in vocabulary



### Column Creation

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

c|[-1:]|_values_|Pushes constant columns for each of _values_

day_count|[-1:]||Pushes a column with the number of days in the period

from_pandas|[:]|_pandas_, _round__|Pushes columns from Pandas DataFrame or Series _pandas_

load|[-1:]||Pushes columns of a DataFrame saved to internal DB as _key_

nan|[-1:]|_values_|Pushes a constant nan-valued column

period_ordinal|[-1:]||Pushes a column with the period ordinal

r|[-1:]|_n_|Pushes constant columns for each whole number from 0 to _n_ - 1

randn|[-1:]||Pushes a column with values from a random normal distribution

timestamp|[-1:]||Pushes a column with the timestamp of the end of the period

### Stack Manipulation

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

drop|[-1:]||Removes selected columns

dup|[-1:]||Duplicates columns

hsort|[:]||Sorts columns by header

id|[:]||Identity/no-op

interleave|[:]|_parts_|Divides columns in _parts_ groups and interleaves the groups

keep|[:]||Removes non-selected columns

nip|[-1:]||Removes non-selected columns, defaulting selection to top

pull|[:]||Brings selected columns to the top

rev|[:]||Reverses the order of selected columns

roll|[:]||Permutes selected columns

swap|[-2:]||Swaps top and bottom selected columns

### Quotation

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

q|[-1:]|_quoted_, _this_|Wraps the following words until *p* as a quotation, or wraps _quoted_ expression as a quotation

### Header manipulation

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

halpha|[:]||Set headers to alphabetical values

happly|[:]|_header_func_|Apply _header_func_ to headers_

hformat|[:]|_format_spec_|Apply _format_spec_ to headers

hreplace|[:]|_old_, _new_|Replace _old_ with _new_ in headers

hset|[:]|_headers_|Set headers to _*headers_ 

hsetall|[:]|_headers_|Set headers to _*headers_ repeating, if necessary

### Combinators

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

call|[:]||Applies quotation

cleave|[:]|_num_quotations_|Applies all preceding quotations

compose|[:]|_num_quotations_|Combines quotations

hmap|[:]||Applies quotation to stacks created grouping columns by header

ifexists|[:]|_count_|Applies quotation if stack has at least _count_ columns

ifexistselse|[:]|_count_|Applies top quotation if stack has at least _count_ columns, otherwise applies second quotation

ifheaders|[:]|_predicate_|Applies top quotation if list of column headers fulfills _predicate_

ifheaderselse|[:]|_predicate_|Applies quotation if list of column headers fulfills _predicate_, otherwise applies second quotation

map|[:]|_every_|Applies quotation in groups of _every_

partial|[-1:]|_quoted_, _this_|Pushes stack columns to the front of quotation

repeat|[:]|_times_|Applies quotation _times_ times

### Data cleanup

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

ffill|[:]|_lookback_, _leave_end_|Fills nans with previous values, looking back _lookback_ before range and leaving trailing nans unless not _leave_end_

fill|[:]||Fills nans with _value_ 

fillfirst|[-1:]|_lookback_|Fills first row with previous non-nan value, looking back _lookback_  before range

join|[-2:]|_date_|Joins two columns at _date_

sync|[:]||Align available data by setting all values to NaN when any values is NaN

zero_first|[-1:]||Changes first value to zero

zero_to_na|[-1:]||Changes zeros to nans

### Resample methods

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

resample_avg|[:]||Sets periodicity resampling method to avg

resample_first|[:]||Sets periodicity resampling method to first

resample_firstnofill|[:]||Sets periodicity resampling method to first

resample_last|[:]||Sets periodicity resampling method to last

resample_lastnofill|[:]||Sets periodicity resampling method to last with no fill

resample_max|[:]||Sets periodicity resampling method to max

resample_min|[:]||Sets periodicity resampling method to min

resample_sum|[:]||Sets periodicity resampling method to sum

set_periodicity|[-1:]|_periodicity_|Changes column periodicity to _periodicity_, then resamples

set_start|[-1:]|_start_|Changes period start to _start_, then resamples

### Row-wise Reduction

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

add|[-2:]|_ignore_nans_|Addition

avg|[-2:]|_ignore_nans_|Arithmetic average

div|[-2:]|_ignore_nans_|Division

max|[-2:]|_ignore_nans_|Maximum

med|[-2:]|_ignore_nans_|Median

min|[-2:]|_ignore_nans_|Minimum

mod|[-2:]|_ignore_nans_|Modulo

mul|[-2:]|_ignore_nans_|Multiplication

pow|[-2:]|_ignore_nans_|Power

std|[-2:]|_ignore_nans_|Standard deviation

sub|[-2:]|_ignore_nans_|Subtraction

var|[-2:]|_ignore_nans_|Variance

### Row-wise Reduction Ignoring NaNs

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

nadd|[-2:]|_ignore_nans_|Addition

navg|[-2:]|_ignore_nans_|Arithmetic average

ndiv|[-2:]|_ignore_nans_|Division

nmax|[-2:]|_ignore_nans_|Maximum

nmed|[-2:]|_ignore_nans_|Median

nmin|[-2:]|_ignore_nans_|Minimum

nmod|[-2:]|_ignore_nans_|Modulo

nmul|[-2:]|_ignore_nans_|Multiplication

npow|[-2:]|_ignore_nans_|Power

nstd|[-2:]|_ignore_nans_|Standard deviation

nsub|[-2:]|_ignore_nans_|Subtraction

nvar|[-2:]|_ignore_nans_|Variance

### Rolling Window

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

ewm_mean|[-1:]|_window_|Exponentially-weighted moving average

ewm_std|[-1:]|_window_|Exponentially-weighted standard deviation

ewm_var|[-1:]|_window_|Exponentially-weighted variance

radd|[-1:]|_window_|Addition

ravg|[-1:]|_window_|Arithmetic average

rcor|[-2:]|_window_|Correlation

rcov|[-2:]|_window_|Covariance

rdif|[-1:]|_window_|Lagged difference

rlag|[-1:]|_window_|Lag

rmax|[-1:]|_window_|Maximum

rmed|[-1:]|_window_|Median

rmin|[-1:]|_window_|Minimum

rret|[-1:]|_window_|Lagged return

rstd|[-1:]|_window_|Standard deviation

rvar|[-1:]|_window_|Variance

rzsc|[-1:]|_window_|Z-score

### Cumulative

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

cadd|[-1:]||Addition

cavg|[-1:]||Arithmetic average

cdif|[-1:]||Lagged difference

clag|[-1:]||Lag

cmax|[-1:]||Maximum

cmin|[-1:]||Minimum

cmul|[-1:]||Multiplication

cret|[-1:]||Lagged return

cstd|[-1:]||Standard deviation

csub|[-1:]||Subtraction

cvar|[-1:]||Variance

### Reverse Cumulative

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

rcadd|[-1:]||Addition

rcavg|[-1:]||Arithmetic average

rcdif|[-1:]||Lagged difference

rclag|[-1:]||Lag

rcmax|[-1:]||Maximum

rcmin|[-1:]||Minimum

rcmul|[-1:]||Multiplication

rcret|[-1:]||Lagged return

rcstd|[-1:]||Standard deviation

rcsub|[-1:]||Subtraction

rcvar|[-1:]||Variance

### One-for-one functions

Word | Default Selector | Parameters | Description
:---|:---|:---|:---

abs|[-1:]||Absolute value

dec|[-1:]||Decrement

exp|[-1:]||Exponential

expm1|[-1:]||Exponential minus one

inc|[-1:]||Increment

inv|[-1:]||Multiplicative inverse

lnot|[-1:]||Logical not

log|[-1:]||Natural log

log1p|[-1:]||Natural log of increment

neg|[-1:]||Additive inverse

rank|[:]||Row-wise rank

sign|[-1:]||Sign

sqrt|[-1:]||Square root

