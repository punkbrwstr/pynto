## pynto: Data analysis in Python using the concatenative paradigm

pynto is a Python package that lets you manipulate tabular data with the expressiveness and code reusability of [concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language) programming.  With pynto you define an _expression_ that formally specifies how to calculate the data in your table.  Expressions are made by stringing together a sequence of functions called _words_.  It works like a pipeline: the output from one word becomes the input for the following word.  The table of data is treated like a stack of independent columns.  The rightmost column in the table is the top of the stack.  Words can add, remove or modify columns, but they are row-agnostic--expressions can be evaluated over any range of rows.  
## What does it look like?

```
>>> from pynto import * 
>>> stocks = csv('stocks.csv')                   # add columns to the stack
>>> ma_diff = dup | rolling(20) | wmean | sub    # define an operation
>>> stocks_ma = stocks | ~ma_diff | each         # operate on columns using quotation/combinator pattern
>>> stocks_ma['2019-01-01':]                     # evaluate your expression over certain rows
```
## Why pynto?

 - Expressive: Intuitive syntax with parameters that expand word functionality and combinators tame the stack  
 - Batteries included:  Datetime-based row ranges; moving window statistics
 - Performant: Efficient numpy internals
 - Interoperable: Seemlessly integration with pandas/numpy 

## Get pynto
```
pip install pynto
```
## pynto reference

### The Basics

Create expressions by _composing_ words together with `|`.  Words operate in left-to-right order, with operators following their operands in _postfix_ style.  When you assign an expression to a Python variable the variable name can be used as word in other expressions.
```
>>> square = dup | mul         # adds duplicate of top column to the stack, then multiplies top two columns 
```
The word `c` that adds a constant-value column to the stack.  Like many pynto words, `c` takes a _parameter_ in parentheses to specify the constant value `c(10.0)`. pynto can handle any NumPy data type, but all rows in a column have to have the same type.

```
>>> expr = c(10.0) | square    # apply square expression to a columns of 10s
```
To evaluate your expression specify the range of rows you want using standard Python `[start:stop:step]` indexing and slicing.  Indices can be ints or datetimes.  For a datetime index the step is the periodicity.  
```
>>> expr[:2]                   # evaluate first two rows                                                     
   constant
0     100.0
1     100.0
```
Each column has a str header that can be changed with header words.  `hset` sets the header to a new value.
```
>>> (expr | hset('ten squared'))[:2]
   ten squared
0        100.0
1        100.0
```
_Combinators_ are higher-order functions that allow pynto to do more complicated things like branching and looping.  Combinators operate on _quotations_, expressions that are pushed to the stack instead of operating on the stack.  To create a quotation use `~` before a word `~square` or before an expression in parentheses `~(dup | mul)` for an anonymous quotation.
```
>>> expr = c(9.) | c(10.) | ~square | each
>>> expr[0]
   constant  constant
0      81.0     100.0
```

## pynto vocabulary

### Words for adding columns
Name | Parameters |Stack effect<br> _before_ -- _after_|Description
:---|:---|:---|:---
c|value| -- c|Adds a constant-_value_ column
csv|csv_file, index_col=0, header='infer'| -- c (c)|Adds columns from _csv_file_
pandas|frame_or_series| -- c (c)|Adds columns from a pandas data structure
c_range|value| -- c (c)|Add constant int columns from 0 to _value_

### Words for arithmetic or logical operators
Name | Parameters |Stack effect<br> _before_ -- _after_|Description
:---|:---|:---|:---
add|| a b -- c| a + b
sub|| a b -- c| a - b
mul|| a b -- c| a * b
div|| a b -- c| a / b
mod|| a b -- c| a % b
exp|| a b -- c| a ** b
eq|| a b -- c| a == b
ne|| a b -- c| a != b
ge|| a b -- c| a >= b
gt|| a b -- c| a > b
le|| a b -- c| a <= b
lt|| a b -- c| a < b
neg|| a -- c| a * -1
absv|| a -- c| abs(a)
sqrt|| a -- c| a ** 0.5
zeroToNa|| a -- c| Replaces zeros with np.nan

### Words to manipulate columns
Name | Parameters |Stack effect<br> _before_ -- _after_|Description
:---|:---|:---|:---
dup|| a -- a a| Duplicate top column
roll|| a b c -- c a b| Permute columns
swap|| a b -- b a| Swap top two columns
drop|| a b c -- a b| Drop top column
clear|| a b c -- | Clear columns
interleave|count=None, split_into=2|a b c d -- a c b d|Divide columns into groups and interleave group elements
pull|start,end=None,clear=False|a b c -- b c a|Bring columns _start_ (to _end_) to the top


### Window words
Name | Parameters |Stack effect|Description
:---|:---|:---|:---
rolling| | | 

