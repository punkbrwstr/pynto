## pynto: Time series analysis in Python using the concatenative paradigm

pynto is a Python package that lets you manipulate tabular data with the expressiveness and code reusability of [concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language) programming.  With pynto you define an _expression_ that formally specifies how to calculate the data in your table.  Expressions are made by chaining together functions called _words_.  It works like a pipeline: the output from one word becomes the input for the following word.  The table of data is treated like a stack of independent columns.  The rightmost column in the table is the top of the stack.  Words can add, remove or modify columns, but they are row-agnostic--expressions can be evaluated over any range of rows.  
## What does it look like?
```
>>> import pynto as pt 
>>> ma_dev = pt.saved('stock_prices') \         # define an expression that appends columns from the build-in database
>>>     .quote(pt.dup.rolling(20).mean.sub) \   # append a quoted expression to the stack
>>>     .each                                   # use the each combinator to apply the quotation to the previous columns
>>> df = ma_dev['2021-06-01':]                  # evaluate your expression over a date range to get a DataFrame
>>> pt.db['stocks_ma_dev'] = df                 # save the results back to the database   
```

## Why pynto?
 - Expressive: Pythonic syntax; Combinatory logic for modular, reusable code 
 - Performant: Column-wise multiprocessing; caching of duplicate operations
 - Batteries included:  Built-in Redis-based time series database
 - Interoperable: Seemlessly integration with Pandas

## Get pynto
```
pip install pynto
```
## Reference

### The Basics

Create expressions by chaining built-in words together using a fluent interface, or using the `+` operator for words in the local namespace.  When evaluated, words will operate in left-to-right order, with operators following their operands in _postfix_ style.  
```
>>> square = pt.dup.mul         # duplicates the top column, then multiplies top two columns together
```
The word `c` that adds a constant-value column to the stack.  Like many pynto words, `c` takes a _parameter_ in parentheses to specify the constant value `c(10.0)`. pynto can handle any NumPy data type, but all rows in a column have to have the same type.

```
>>> expr = pt.c(10.0) + square    # compose square expression with a columns of 10s
```
To evaluate your expression, specify a date range using the `[`_start_`:`_stop (exclusive)_`:`_periodicity_`]` syntax.  (Instances of `pt.Range` in the local namespace also work.)
```
>>> expr['2021-06-01':'2021-06-03','B']                   # evaluate over a two business day date range                                                   
             constant
2021-06-01     100.0
2021-06-02     100.0
```
Each column has a string header that can be modified.  `hset` sets the header to a new value.  Headers can be usefully for filtering or arranging columns.
```
>>> expr += pt.hset('ten squared')
>>> expr['2021-06-01':'2021-06-03','B']  
               ten squared
2021-06-01        100.0
2021-06-02        100.0
```
_Combinators_ are higher-order functions that allow pynto to do more complicated things like branching and looping.  Combinators operate on _quotations_, expressions that are pushed to the stack instead of operating on the stack.  To create a quotation put an expression within the parentheses of `pt.quote()`.
```
>>> expr = pt.c(9.).c(10.).quote(square).each
>>> expr['2021-06-01':'2021-06-02']
              constant  constant
2021-06-01      81.0     100.0
```

### The Database

pynto has built-in time series database functionality that lets you save pandas DataFrames and Series to a Redis database.  The database uses only String key/values, saving the underlying numpy data in native byte format for zero-copy retrieval.   Each DataFrame column is saved as an independent key and can be retrieved on its own.  The database also supports three-dimensional frames that have a two-level MultiIndex.  Saved data can be accessed through pynto expressions or standard methods

```
>>> pt.db['my_df'] = expr['2021-06-01':'2021-06-03']
>>> pt.saved('my_df')['2021-06-01':'2021-06-02']
              constant  constant
2021-06-01      81.0     100.0
>>> pt.db.read('my_df')
              constant  constant
2021-06-01      81.0     100.0
2021-06-02      81.0     100.0
```

## pynto built-in vocabulary

### Words for adding columns
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
c|value| -- c|Adds a constant-_value_ column.
csv|csv_file, index_col=0, header='infer'| -- c (c)|Adds columns from _csv_file_.
pandas|frame_or_series| -- c (c)|Adds columns from a pandas data structure.
c_range|value| -- c (c)|Add constant int columns from 0 to _value_.

### Combinators
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
call|depth=None, copy=False| a q -- c| Apply quotation to stack, up to _depth_ if specified.  Optionally leaves stack in place with _copy_.
each|start=0, stop=None, every=1, copy=False| a b q -- c d| Apply quotation stack elements from _start_ to _end_ in groups of _every_.  Optionally leaves stack in place with _copy_.
repeat|times| a b q -- c d| Apply quotation to the stack _times_ times.
cleave|num_quotations, depth=None, copy=False| a q q -- c d| Apply _num_quotations_ quotations to copies of stack elements up to _depth_.  Optionally leaves stack in place with _copy_.

### Words to manipulate columns
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
dup|| a -- a a| Duplicate top column.
roll|| a b c -- c a b| Permute columns.
swap|| a b -- b a| Swap top two columns.
drop|| a b c -- a b| Drop top column.
clear|| a b c -- | Clear columns.
interleave|count=None, split_into=2|a b c d -- a c b d|Divide columns into _split into_ groups and interleave group elements.
pull|start,end=None,clear=False|a b c -- b c a|Bring columns _start_ (to _end_) to the top.
hpull|\*headers, clear=False|a b c -- b c a|Bring columns with headers matching regex _headers_ to the top.  Optionally clear remainder of stack
hfilter|\*headers, clear=False|a b c -- a|Shortcut for hpull with _clear_=True

### Words to manipulate headers
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
hset|\*headers| a b -- a b|Set top columns' headers to _headers_.
hformat|format_string| a -- a|Apply _format_string_ to existing headers.
happly|header_function| a -- a|Apply _header_function_ to existing header.

### Words for arithmetic or logical operators
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
add||a b -- c|a + b
sub||a b -- c|a - b
mul||a b -- c|a * b
div||a b -- c|a / b
mod||a b -- c|a % b
pow||a b -- c|a ** b
eq||a b -- c|a == b
ne||a b -- c|a != b
ge||a b -- c|a >= b
gt||a b -- c|a > b
le||a b -- c|a <= b
lt||a b -- c|a < b
neg||a -- c|a * -1
abs||a -- c|abs(a)
sqrt||a -- c|a ** 0.5
zeroToNa|| a -- c|Replaces zeros with np.nan

### Words for creating window columns
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
rolling|window=2, exclude_nans=True, lookback_multiplier=2|a -- w|Create window column with values from most recent _window_ rows.  Exclude nan-valued rows from count unless _exclude_nans_.  Extend history up to _lookback_multiplier_ to look for non-nan rows.  
crossing||a b c -- w|Create window column with cross-sectional values from the same rows of all columns. 

### Words for calculating statistics on window columns
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
sum||w -- c|Sums of windows.
mean||w -- c|Means of windows.
var||w -- c|Variances of windows.
std||w -- c|Standard deviations of windows.
change||w -- c|Changes between first and last rows of windows.
pct_change||w -- c|Percent changes between first and last rows of windows.
log_change||w -- c|Differences of logs of first and last rows of windows.
first||w -- c|First rows of windows.
last||w -- c|Last rows of windows.
zscore||w -- c|Z-score of most recent rows within windows.

### Words for cleaning up data
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
fill|value|a -- a|Fill nans with _value_.
ffill||a -- a|Last observation carry-forward.
join|date|a b -- c|Join top two columns, switching from second to first on _date_ index.

### Other words
Name | Parameters |Stack effect<br>_before_&nbsp;--&nbsp;_after_|Description
:---|:---|:---|:---
ewma|window, fill_nans=True|a -- c|Calculates exponentially-weighted moving average with half-life _window_. 
wlag|number|w -- c|Lag _number_ rows.
