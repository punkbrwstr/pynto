## pynto: Data analysis in Python using the concatenative paradigm

pynto lets you manipulate tabular data in Python with the expressiveness and code reusability of [concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language) programming.  In pynto you create a formal definition for your data, an _expression_, by stringing together functions called _words_ .  The expression works like a pipeline: the output from one word becomes the input for the following word.  The data is kept in a table that is treated like a stack of independent columns.  The rightmost column in the table is the top of the stack.  pynto words operate on the columns and are row-agnostic--an expression can be evaluated over any range of rows.  

## Why pynto?

 - Expressive: Intuitive syntax; parameters expand word functionality; combinators tame the stack  
 - Batteries included:  Datetime-based row ranges; moving window statistics
 - Performant: No loops; efficient numpy-based internals
 - Interoperable: Integrates seemlessly with pandas/numpy 

### What does it look like?

```
>>> from pynto import * 
>>> stocks = csv('stocks.csv')                   # a word that adds some columns to the stack
>>> ma_diff = dup | rolling(20) | wmean | sub    # a word that defines an operation on one column
>>> stocks_ma = stocks | ~ma_diff | each         # apply operation to all columns using quotation/combinator pattern
>>> stocks_ma['2019-01-01':]                     # evaluate your expression over certain rows

```

## pynto reference

### Basics

#### Creating expressions
pynto expressions are created by _concatenating_ words together with the `|` operator.  When you assign an expression to a Python variable that variable name can be used as word.

#### Literals and types
Actually there are no literals, but `c` adds a constant-valued column to the stack.  Column values can be any NumPy data type, but the type will be consistent for all rows in the column.

### Word parameters

#### Row indexing





### Quotations/combinators

## pynto vocabulary

### Data acquisition words
Name | Parameters |Stack effect|Description
:---:|:---|:---|:---
c|value| -- c|Constant

### Arithmetic words

### Window words
Name | Parameters |Stack effect|Description
:---:|:---|:---|:---
rolling| | | 

