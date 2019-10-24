import unittest
import pandas as pd
import numpy as np
from pynto import *

def get_test_data():
    return pd.DataFrame(np.arange(12).reshape(3,4),columns=['a','b','c','d'],
                            index=pd.date_range('1/1/2019',periods=3,freq='B'))

class TestRanges(unittest.TestCase):
    def test_int_range(self):
        expr = c(5)
        result = expr[:10]
        self.assertEqual(result.shape[0], 10)

    def test_date_range(self):
        expr = c(5)
        result = expr['2019-01-01':'2019-01-15']
        self.assertEqual(result.shape[0], 10)

class TestOperators(unittest.TestCase):
    def test_add(self):
        expr = c(5) | c(6) | add
        result = expr[0].iloc[0,0]
        self.assertEqual(result, 11.)

    def test_sub(self):
        expr = c(5) | c(6) | sub
        result = expr[0].iloc[0,0]
        self.assertEqual(result, -1.)

    def test_le(self):
        expr = c(5) | c(6) | le
        result = expr[0].iloc[0,0]
        self.assertEqual(result, 1.)

    def test_ge(self):
        expr = c(5) | c(6) | ge
        result = expr[0].iloc[0,0]
        self.assertEqual(result, 0.)

    def test_neg(self):
        expr = c(5) | neg
        result = expr[0].iloc[0,0]
        self.assertEqual(result, -5.)

class TestPandas(unittest.TestCase):
    def test_int_index(self):
        expr = pandas(get_test_data())
        result = expr[0].iloc[0,0]
        self.assertEqual(result, 0)

    def test_int_range(self):
        expr = pandas(get_test_data())
        result = expr[0:2]
        self.assertEqual(result.shape[0], 2)

    def test_date_index(self):
        expr = pandas(get_test_data())
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, 0)

    def test_date_range(self):
        expr = pandas(get_test_data())
        result = expr['2019-01-01':'2019-01-03']
        self.assertEqual(result.shape[0], 2)

class TestStackManipulation(unittest.TestCase):
    def test_pull(self):
        expr = pandas(get_test_data()) | pull(2)
        result = expr[0].iloc[0,-1]
        self.assertEqual(result, 1)

    def test_interleave(self):
        expr = pandas(get_test_data()) | interleave
        result = expr[0]
        self.assertEqual(result.iloc[0,-2], 1)
        
    def test_hpull(self):
        expr = pandas(get_test_data()) | hpull('b')
        result = expr[0]
        self.assertEqual(result.iloc[0,-1], 1)

class TestCombinators(unittest.TestCase):
    def test_single_quoted_word(self):
        expr = c(5) | c(6) | ~add | call
        result = expr[0].iloc[0,-1]
        self.assertEqual(result, 11)

    def test_multiple_quoted_words(self):
        expr = c(5) | c(6) | ~(c(1) | add) | call
        result = expr[0].iloc[0,-1]
        self.assertEqual(result, 7)

    def test_each(self):
        expr = c(5) | c(6) | ~(c(1) | add) | each
        result = expr[0]
        self.assertEqual(result.iloc[0,-1], 7)
        self.assertEqual(result.iloc[0,-2], 6)

    def test_each_with_copy(self):
        expr = c(5) | c(6) | ~(c(1) | add) | each(copy=True)
        result = expr[0]
        self.assertEqual(result.shape[1], 4)

    def test_cleave(self):
        expr = c(4) | ~neg | ~sqrt | cleave(2)
        result = expr[0]
        self.assertEqual(result.iloc[0,-1], 2)
        self.assertEqual(result.iloc[0,-2], -4)

class TestHeaders(unittest.TestCase):
    def test_hset(self):
        expr = pandas(get_test_data()) | hset('q','w','e','r')
        result = expr[0]
        self.assertEqual(result.columns[1], 'w')

    def test_hformat(self):
        expr = pandas(get_test_data()) | ~hformat('{0}x') | each
        result = expr[0]
        self.assertEqual(result.columns[1], 'bx')

    def test_happly(self):
        expr = pandas(get_test_data()) | ~happly(lambda h: h.upper()) | each
        result = expr[0]

class TestRolling(unittest.TestCase):
    def test_sum(self):
        expr = pandas(get_test_data()) | rolling | wsum
        result = expr[:]
        self.assertEqual(result.iloc[-1,-1], 18)

class TestCrossing(unittest.TestCase):
    def test_sum(self):
        expr = pandas(get_test_data()) | crossing | wsum
        result = expr[:]
        self.assertEqual(result.iloc[-1,-1], 38)

class TestDataCleanup(unittest.TestCase):
    def test_fill(self):
        data = get_test_data().astype('d')
        data.values[1,2] = np.nan
        expr = pandas(data) | ~fill(0) | each
        result = expr[:]
        self.assertEqual(result.iloc[1,2], 0)

    def test_ffill(self):
        data = get_test_data().astype('d')
        data.values[1,2] = np.nan
        expr = pandas(data) | ~ffill | each
        result = expr[:]
        self.assertEqual(result.iloc[1,2], 2.)

    def test_join_int(self):
        expr = c(5) | c(6) | join(3) 
        result = expr[:10]
        self.assertEqual(result.iloc[2,-1], 5)
        self.assertEqual(result.iloc[3,-1], 6)

    def test_join_dates(self):
        expr = c(5) | c(6) | join('2019-01-05') 
        result = expr['2019-01-01':'2019-01-15']
        self.assertEqual(result.iloc[3,-1], 5)
        self.assertEqual(result.iloc[4,-1], 6)

if __name__ == '__main__':
    unittest.main()
