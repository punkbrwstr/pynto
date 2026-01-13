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
>>> ma_dev = pt.saved('stock_prices') \     # append columns to stack from the build-in database
>>>     .q                            \     # start a quotation 
>>>         .dup                      \     # push a copy of the top (leftmost) column of the stack
>>>         .ravg(20)                 \     # calculate 20-period moving average
>>>         .sub                      \     # subtract top column from second column 
>>>     .p                            \     # close the quotation
>>>     .map                          \     # use the map combinator to apply the quotation
>>>                                         # to each column in the stack
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
>>> pt.saved('my_df').rows[:]
              constant  constant
2021-06-01      81.0     100.0
2021-06-02      81.0     100.0
```

## pynto built-in vocabulary



## Column Creation

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| c | Pushes constant columns for each of _values_ | values: list[float] | [-1:0] |
| dc | Pushes a column with the number of days in the period |  | [-1:0] |
| nan | Pushes a constant nan-valued column | values: list[float] | [-1:0] |
| pandas | Pushes columns from Pandas DataFrame or Series _pandas_ | pandas: pd.DataFrame \| pd.Series, round_: bool = False | [:] |
| po | Pushes a column with the period ordinal |  | [-1:0] |
| r | Pushes constant columns for each whole number from 0 to _n_ - 1 | n: int | [-1:0] |
| randn | Pushes a column with values from a random normal distribution |  | [-1:0] |
| saved | Pushes columns saved to internal DB as _key_ | key: str | [-1:0] |
| ts | Pushes a column with the timestamp of the end of the period |  | [-1:0] |

## Combinators

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| call | Applies quotation |  | [:] |
| cleave | Applies all preceding quotations | num_quotations: int = -1 | [:] |
| compose | Combines quotations | num_quotations: int = 2 | [:] |
| hmap | Applies quotation to stacks created grouping columns by header |  | [:] |
| ifexists | Applies quotation if stack has at least _count_ columns | count: int = 1 | [:] |
| ifexistselse | Applies top quotation if stack has at least _count_ columns, otherwise applies second quotation | count: int = 1 | [:] |
| ifheaders | Applies top quotation if list of column headers fulfills _predicate_ | predicate: Callable[[list[str]], bool] | [:] |
| ifheaderselse | Applies quotation if list of column headers fulfills _predicate_, otherwise applies second quotation | predicate: Callable[[list[str]], bool] | [:] |
| map | Applies quotation in groups of _every_ | every: int = 1 | [:] |
| partial | Pushes stack columns to the front of quotation | quoted: Word \| None = None | [-1:] |
| repeat | Applies quotation _times_ times | times: int = 2 | [:] |

## Cumulative

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| cadd | Addition |  | [-1:] |
| cavg | Arithmetic average |  | [-1:] |
| cdif | Lagged difference |  | [-1:] |
| clag | Lag |  | [-1:] |
| cmax | Maximum |  | [-1:] |
| cmin | Minimum |  | [-1:] |
| cmul | Multiplication |  | [-1:] |
| cret | Lagged return |  | [-1:] |
| cstd | Standard deviation |  | [-1:] |
| csub | Subtraction |  | [-1:] |
| cvar | Variance |  | [-1:] |

## Data cleanup

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| ffill | Fills nans with previous values, looking back _lookback_ before range and leaving trailing nans unless not _leave_end_ | lookback: int = 10, leave_end: bool = True | [:] |
| fill | Fills nans with _value_  | value: float | [:] |
| join | Joins two columns at _date_ | date: datelike | [-2:] |
| per | Changes column periodicity to _periodicity_, then resamples | periodicity: str \| Periodicity | [-1:] |
| resample_avg | Sets periodicity resampling method to avg |  | [:] |
| resample_last | Sets periodicity resampling method to last |  | [:] |
| resample_sum | Sets periodicity resampling method to sum |  | [:] |
| start | Changes period start to _start_, then resamples | start: datelike, round_: bool = False | [:] |
| zero_first | Changes first value to zero |  | [-1:] |
| zero_to_na | Changes zeros to nans |  | [-1:] |

## Header manipulation

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| halpha | Set headers to alphabetical values |  | [:] |
| happly | Apply _header_func_ to headers_ | header_func: Callable[[str], str] | [:] |
| hformat | Apply _format_spec_ to headers | format_spec: str | [:] |
| hreplace | Replace _old_ with _new_ in headers | old: str, new: str = '' | [:] |
| hset | Set headers to _*headers_  | headers: str | [:] |
| hsetall | Set headers to _*headers_ repeating, if necessary | headers: str | [:] |

## One-for-one functions

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| abs | Absolute value |  | [-1:] |
| dec | Decrement |  | [-1:] |
| exp | Exponential |  | [-1:] |
| expm1 | Exponential minus one |  | [-1:] |
| inc | Increment |  | [-1:] |
| inv | Multiplicative inverse |  | [-1:] |
| lnot | Logical not |  | [-1:] |
| log | Natural log |  | [-1:] |
| log1p | Natural log of increment |  | [-1:] |
| neg | Additive inverse |  | [-1:] |
| rank | Row-wise rank |  | [:] |
| sign | Sign |  | [-1:] |
| sqrt | Square root |  | [-1:] |

## Quotation

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| q | Wraps the following words until *p* as a quotation, or wraps _quoted_ expression as a quotation | quoted: Word \| None = None | [-1:0] |

## Reverse Cumulative

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| rcadd | Addition |  | [-1:] |
| rcavg | Arithmetic average |  | [-1:] |
| rcdif | Lagged difference |  | [-1:] |
| rclag | Lag |  | [-1:] |
| rcmax | Maximum |  | [-1:] |
| rcmin | Minimum |  | [-1:] |
| rcmul | Multiplication |  | [-1:] |
| rcret | Lagged return |  | [-1:] |
| rcstd | Standard deviation |  | [-1:] |
| rcsub | Subtraction |  | [-1:] |
| rcvar | Variance |  | [-1:] |

## Rolling Window

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| radd | Addition | window: int = 2 | [-1:] |
| ravg | Arithmetic average | window: int = 2 | [-1:] |
| rcor | Correlation | window: int = 2 | [-2:] |
| rcov | Covariance | window: int = 2 | [-2:] |
| rdif | Lagged difference | window: int = 2 | [-1:] |
| rewm | Exponentially-weighted average | window: int = 2 | [-1:] |
| rews | Exponentially-weighted standard deviation | window: int = 2 | [-1:] |
| rewv | Exponentially-weighted variance | window: int = 2 | [-1:] |
| rlag | Lag | window: int = 2 | [-1:] |
| rmax | Maximum | window: int = 2 | [-1:] |
| rmed | Median | window: int = 2 | [-1:] |
| rmin | Minimum | window: int = 2 | [-1:] |
| rret | Lagged return | window: int = 2 | [-1:] |
| rstd | Standard deviation | window: int = 2 | [-1:] |
| rvar | Variance | window: int = 2 | [-1:] |
| rzsc | Z-score | window: int = 2 | [-1:] |

## Row-wise Reduction

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| add | Addition | ignore_nans: bool = False | [-2:] |
| avg | Arithmetic average | ignore_nans: bool = False | [-2:] |
| div | Division | ignore_nans: bool = False | [-2:] |
| max | Maximum | ignore_nans: bool = False | [-2:] |
| med | Median | ignore_nans: bool = False | [-2:] |
| min | Minimum | ignore_nans: bool = False | [-2:] |
| mod | Modulo | ignore_nans: bool = False | [-2:] |
| mul | Multiplication | ignore_nans: bool = False | [-2:] |
| pow | Power | ignore_nans: bool = False | [-2:] |
| std | Standard deviation | ignore_nans: bool = False | [-2:] |
| sub | Subtraction | ignore_nans: bool = False | [-2:] |
| var | Variance | ignore_nans: bool = False | [-2:] |

## Row-wise Reduction Ignoring NaNs

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| nadd | Addition | ignore_nans: bool = False | [-2:] |
| navg | Arithmetic average | ignore_nans: bool = False | [-2:] |
| ndiv | Division | ignore_nans: bool = False | [-2:] |
| nmax | Maximum | ignore_nans: bool = False | [-2:] |
| nmed | Median | ignore_nans: bool = False | [-2:] |
| nmin | Minimum | ignore_nans: bool = False | [-2:] |
| nmod | Modulo | ignore_nans: bool = False | [-2:] |
| nmul | Multiplication | ignore_nans: bool = False | [-2:] |
| npow | Power | ignore_nans: bool = False | [-2:] |
| nstd | Standard deviation | ignore_nans: bool = False | [-2:] |
| nsub | Subtraction | ignore_nans: bool = False | [-2:] |
| nvar | Variance | ignore_nans: bool = False | [-2:] |

## Stack Manipulation

| Word | Description | Parameters | Column Indexer |
|------|-------------|------------|----------------|
| drop | Removes selected columns |  | [-1:] |
| dup | Duplicates columns |  | [-1:] |
| filter | Removes non-selected columns |  | [:] |
| hsort | Sorts columns by header |  | [:] |
| id | Identity/no-op |  | [:] |
| interleave | Divides columns in _parts_ groups and interleaves the groups | parts: int = 2 | [:] |
| nip | Removes non-selected columns, defaulting selection to top |  | [-1:] |
| pull | Brings selected columns to the top |  | [:] |
| rev | Reverses the order of selected columns |  | [:] |
| roll | Permutes selected columns |  | [:] |
| swap | Swaps top and bottom selected columns |  | [-2:] |
