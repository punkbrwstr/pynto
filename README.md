## pynto: Data analysis in Python using the concatenative paradigm

pynto is a Python package that lets you manipulate tabular data using the expressiveness and code reusability of [concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language) programming.  With pynto you create a formal definition for your data, an _expression_, by stringing together functions called _words_ .  The expression works like a pipeline: the output from one word becomes the input for the following word.  Data is kept in a table that is treated like a stack of independent columns.  The rightmost column in the table is the top of the stack.  Words can add, remove or modify columns, but they are row-agnostic--expressions can be evaluated over any range of rows.  

## Why pynto?

 - Expressive: Intuitive postfix syntax; parameters expand word functionality; combinators tame the stack  
 - Batteries included:  Datetime-based row ranges; moving window statistics
 - Performant: Efficient numpy-based internals--no Python loops
 - Interoperable: Seemlessly integration with pandas/numpy 

### What does it look like?

```
>>> from pynto import * 
>>> stocks = csv('stocks.csv')                   # add columns to the stack
>>> ma_diff = dup | rolling(20) | wmean | sub    # define an operation
>>> stocks_ma = stocks | ~ma_diff | each         # operate on columns using quotation/combinator pattern
>>> stocks_ma['2019-01-01':]                     # evaluate your expression over certain rows

```

## pynto reference

### The Basics

pynto expressions are created by _concatenating_ pynto words together with the `|` operator.  When you assign an expression to a Python variable the variable name can be used as word in other expressions.
```
>>> square = dup | mul      # adds duplicate of top column to the stack, then multiplies top two columns 
```
There are no literals, but the word `c` adds a constant-value column to the stack.  Like many pynto words, `c` takes a _parameter_ in parentheses to specify the constant value `c(10.0)`. pynto can handle any NumPy data type.  All rows in a column will have the same type.  

```
>>> expr = c(10.0) | square  # concatenates square expression with constant columns of 10s
```


```
>>> expr[:2]                     # evaluate two rows                                                                                                                                      
   constant
0     100.0
1     100.0
```

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

