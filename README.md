## pynto: Data analysis in Python using the concatenative paradigm

pynto lets you use Python to analyze tabular data with the expressiveness and code reusability of concatenative programming.  In pynto you manipulate data by stringing together functions ("words") into combinations ("expressions").  An expression works like a pipeline: the output from one word becomes the input for the following word.  All data in pynto is held in a table that is treated like a stack of independent columns.  The rightmost column is at the top of the stack.  pynto expressions define operations on the columns but are row-agnostic--they can be evaluated over any range of rows.  

## Why pynto?

 - Expressive: Simple syntax, parameters to modify word behavior, combinators to tame the stack  
 - Batteries included:  Time-based row ranges, moving window statistics
 - Performant: Efficient numpy-based internals
 - Interoperable: Integrates seemlessly with pandas 

### A simple example

```
>>> from pynto import *
>>> ma_diff = dup + rolling(20) + wmean + sub
>>> stocks_ma = csv('stocks.csv') + ~ma_diff + each
>>> stocks_ma['2019-01-01':]

```

## pynto reference

### Word parameters

### Quotations/combinators

## pynto vocabulary

### Data acquisition words
Name | Parameters |Stack effect|Description
:---:|:---|:---|:---
c|value| -- c|Constant


### Window words
Name | Parameters |Stack effect|Description
:---:|:---|:---|:---
rolling| | | 

