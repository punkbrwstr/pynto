## pynto: Data analysis in Python using the concatenative paradigm

## Introduction

pynto lets you use Python to analyze tabular data with the expressiveness and code reusability of concatenative programming.  In pynto, you work with data by stringing together functions called "words" into combinations called "expressions".  The words in an expression form a pipeline: the table outputted from one word becomes the input for the following word.  Words treat the table of data like a stack of independent columns.  The rightmost column is at the top of the stack.  pynto expressions define operations on the columns of a table and are row-agnostic--they can be evaluated over any range of rows.  







## Pynto vocabulary 

### Data acquisition words
Name | Parameters |Stack effect|Description
:---:|:---|:---|:---
c|value| -- c|Constant


### Window words
Name | Parameters |Stack effect|Description
:---:|:---|:---|:---
rolling| | | 

