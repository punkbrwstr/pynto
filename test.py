import unittest
import pandas as pd
import numpy as np
import pynto as pt

def get_test_data():
    return pd.DataFrame(np.arange(12).astype('int64').reshape(3,4),columns=['a','b','c','d'],
                            index=pd.date_range('1/1/2019',periods=3,freq='B'))

class TestRanges(unittest.TestCase):
    def test_int_range(self):
        expr = pt.c(5)
        result = expr[-10]
        self.assertEqual(result.shape[0], 10)

    def test_date_range(self):
        expr = pt.c(5)
        result = expr['2019-01-01':'2019-01-15']
        self.assertEqual(result.shape[0], 10)

class TestOperators(unittest.TestCase):
    def test_add(self):
        expr = pt.c(5).c(6).add
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, 11.)

    def test_sub(self):
        expr = pt.c(5).c(6).sub
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, -1.)

    def test_le(self):
        expr = pt.c(5).c(6).le
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, 1.)

    def test_ge(self):
        expr = pt.c(5).c(6).ge
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, 0.)

    def test_neg(self):
        expr = pt.c(5).neg
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, -5.)

class TestPandas(unittest.TestCase):
    def test_int_index(self):
        expr = pt.pandas(get_test_data())
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, 0)

    def test_date_index(self):
        expr = pt.pandas(get_test_data())
        result = expr['2019-01-01'].iloc[0,0]
        self.assertEqual(result, 0)

    def test_date_range(self):
        expr = pt.pandas(get_test_data())
        result = expr['2019-01-01':'2019-01-03']
        self.assertEqual(result.shape[0], 2)

class TestStackManipulation(unittest.TestCase):
    def test_pull(self):
        expr = pt.pandas(get_test_data()).pull(2)
        result = expr['2019-01-01'].iloc[0,-1]
        self.assertEqual(result, 1)

    def test_interleave(self):
        expr = pt.pandas(get_test_data()).interleave
        result = expr['2019-01-01']
        self.assertEqual(result.iloc[0,-2], 1)
        
    def test_hpull(self):
        expr = pt.pandas(get_test_data()).hpull('b')
        result = expr['2019-01-01']
        self.assertEqual(result.iloc[0,-1], 1)

class TestCombinators(unittest.TestCase):
    def test_single_quoted_word(self):
        expr = pt.c(5).c(6).quote(pt.add).call
        result = expr['2019-01-01'].iloc[0,-1]
        self.assertEqual(result, 11)

    def test_inline_quoted_word(self):
        expr = pt.c(5).c(6).q.add.p.call
        result = expr['2019-01-01'].iloc[0,-1]
        self.assertEqual(result, 11)

    def test_nested_quotes(self):
        expr = pt.r9.q.q.sub(100).p.call(2).p.each(every=3)
        result = expr[-1]
        self.assertTrue(np.array_equal(result.values[-1],[0,1.,-98,3,4,-95,6,7,-92]))

    def test_nested_mix(self):
        expr = pt.r9.q.q(pt.sub(100)).call(2).p.each(every=3)
        result = expr[-1]
        self.assertTrue(np.array_equal(result.values[-1],[0,1.,-98,3,4,-95,6,7,-92]))

    def test_multiple_quoted_words(self):
        expr = pt.c(5).c(6).quote(pt.c(1).add).call
        result = expr['2019-01-01'].iloc[0,-1]
        self.assertEqual(result, 7)

    def test_each(self):
        expr = pt.c(5).c(6).quote(pt.c(1).add).each
        result = expr['2019-01-01']
        self.assertEqual(result.iloc[0,-1], 7)
        self.assertEqual(result.iloc[0,-2], 6)

    def test_heach(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b').q.crossing.sum.p.heach.hsort
        result = expr[-1]
        self.assertTrue(np.array_equal(result.values[-1],[26.,18,1]))

    def test_ifexists(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b').q.q.mul(100).p.ifexists(3).crossing.sum.p.heach.hsort
        result = expr[-1]
        self.assertTrue(np.array_equal(result.values[-1],[818.,909,1]))

    def test_if(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b').q.q.mul(100).p.if_(lambda l: len(l) >= 3).crossing.sum.p.heach.hsort
        result = expr[-1]
        self.assertTrue(np.array_equal(result.values[-1],[818.,909,1]))

    def test_ifelse(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b').q.q.div(100).p.q.mul(100).p.ifelse(lambda l: len(l) >= 3).crossing.sum.p.heach.hsort
        result = expr[-1]
        self.assertTrue(np.array_equal(result.values[-1],[818.,909,.01]))

    def test_each_with_copy(self):
        expr = pt.c(5).c(6).quote(pt.c(1).add).each(copy=True)
        result = expr['2019-01-01']
        self.assertEqual(result.shape[1], 4)

    def test_cleave(self):
        expr = pt.c(4).quote(pt.neg).quote(pt.sqrt).cleave(2)
        result = expr['2019-01-01']
        self.assertEqual(result.iloc[0,-1], 2)
        self.assertEqual(result.iloc[0,-2], -4)

class TestHeaders(unittest.TestCase):
    def test_hset(self):
        expr = pt.pandas(get_test_data()).hset('q','w','e','r')
        result = expr['2019-01-01']
        self.assertEqual(result.columns[1], 'w')

    def test_hformat(self):
        expr = pt.pandas(get_test_data()).quote(pt.hformat('{0}x')).each
        result = expr['2019-01-01']
        self.assertEqual(result.columns[1], 'bx')

    def test_happly(self):
        expr = pt.pandas(get_test_data()).quote(pt.happly(lambda h: h.upper())).each
        result = expr['2019-01-01']

class TestRolling(unittest.TestCase):
    def test_sum(self):
        expr = pt.pandas(get_test_data()).rolling.sum
        result = expr['2019-01-01':'2019-01-04']
        self.assertEqual(result.iloc[-1,-1], 18)

class TestCrossing(unittest.TestCase):
    def test_sum(self):
        expr = pt.pandas(get_test_data()).crossing.sum
        result = expr['2019-01-01':'2019-01-04']
        self.assertEqual(result.iloc[-1,-1], 38)

class TestDataCleanup(unittest.TestCase):
    def test_fill(self):
        data = get_test_data().astype('d')
        data.values[1,2] = np.nan
        expr = pt.pandas(data).quote(pt.fill(0)).each
        result = expr[data.index[0]:data.index[-1]]
        self.assertEqual(result.iloc[1,2], 0)

    def test_ffill(self):
        data = get_test_data().astype('d')
        data.values[1,2] = np.nan
        expr = pt.pandas(data).quote(pt.ffill).each
        result = expr['2019-01-01':'2019-01-04']
        self.assertEqual(result.iloc[1,2], 2.)

    def test_join_dates(self):
        expr = pt.c(5).c(6).join('2019-01-05') 
        result = expr['2019-01-01':'2019-01-15']
        self.assertEqual(result.iloc[3,-1], 5)
        self.assertEqual(result.iloc[4,-1], 6)

class DbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            del(pt.db['test'])
        except KeyError:
            pass

    @classmethod
    def tearDownClass(cls):
        pass

class FrameTest(DbTest):
    def setUp(self):
        pt.db['test'] = pd.DataFrame(np.arange(12).astype('int64').reshape(3,4),columns=['a','b','c','d'],
                            index=pd.date_range('1/1/2019',periods=3,freq='B'))

    def tearDown(self):
        del(pt.db['test'])
        
    def test_read(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 4)
        self.assertEqual(f.shape[0], 3)
        self.assertEqual(f.sum().sum(), 66)

    def test_append_row(self):
        pt.db['test'] = pd.DataFrame(np.arange(4).astype('int64').reshape(1,4),columns=['a','b','c','d'],
                            index=pd.date_range('1/4/2019',periods=1,freq='B'))
        f = pt.db['test']
        self.assertEqual(f.shape[1], 4)
        self.assertEqual(f.shape[0], 4)
        self.assertEqual(f.sum().sum(), 72)

    def test_append_col(self):
        pt.db['test'] = pd.DataFrame(np.arange(3).astype('int64').reshape(3,1),columns=['e'],
                            index=pd.date_range('1/1/2019',periods=3,freq='B'))
        f = pt.db['test']
        self.assertEqual(f.shape[1], 5)
        self.assertEqual(f.shape[0], 3)
        self.assertEqual(f.sum().sum(), 69)

class SeriesTest(DbTest):
    def setUp(self):
        pt.db['test'] = pd.Series(np.arange(3).astype('int64'), index=pd.date_range('1/1/2019',periods=3,freq='B'))

    def tearDown(self):
        del(pt.db['test'])
        
    def test_read(self):
        s = pt.db['test']
        self.assertEqual(len(s), 3)
        self.assertEqual(s.sum(), 3)

    def test_read_before(self):
        s = pt.db.read('test', '2018-12-31')
        self.assertEqual(len(s), 4)
        self.assertEqual(s.iloc[0], 0)

    def test_read_after(self):
        s = pt.db.read('test', stop='2019-01-05')
        self.assertEqual(len(s), 4)
        self.assertEqual(s.iloc[-1], 0)

    def test_append(self):
        pt.db['test'] = pd.Series(np.arange(3).astype('int64'), index=pd.date_range('1/4/2019',periods=3,freq='B'))
        s = pt.db['test']
        self.assertEqual(len(s), 6)
        self.assertEqual(s.sum(), 6)

    def test_append_with_pad(self):
        pt.db['test'] = pd.Series(np.arange(3).astype('int64'), index=pd.date_range('1/5/2019',periods=3,freq='B'))
        s = pt.db['test']
        self.assertEqual(len(s), 7)
        self.assertEqual(s.sum(), 6)

    def test_replace(self):
        pt.db['test'] = pd.Series(10, index=pd.date_range('1/2/2019',periods=1,freq='B'))
        s = pt.db['test']
        self.assertEqual(len(s), 3)
        self.assertEqual(s.sum(), 12)

    def test_prepend(self):
        pt.db['test'] = pd.Series(np.arange(3).astype('int64'), index=pd.date_range('12/31/2018',periods=3,freq='B'))
        s = pt.db['test']
        self.assertEqual(len(s), 3)
        self.assertEqual(s.sum(), 3)

class MultiIndexFrameTest(DbTest):

    def tearDown(self):
        del(pt.db['test'])

    def test_square(self):
        index = pd.MultiIndex.from_product([pd.date_range('1/1/2019',periods=5,freq='B'),['x','y','z']])
        pt.db['test'] = pd.DataFrame(np.arange(45).astype('int64').reshape(15,3),columns=['a','b','c'], index=index)
        f = pt.db['test']
        f.loc['2019-01-07'].shape == (3,3)
        f.loc['2019-01-07'].iloc[2,2] == 44
        f.loc['2019-01-07','x']['a'] == 36

    def test_rect(self):
        index = pd.MultiIndex.from_product([pd.date_range('1/1/2019',periods=5,freq='B'),['x','y','z','aa','bb']])
        pt.db['test'] = pd.DataFrame(np.arange(75).astype('int64').reshape(25,3),columns=['a','b','c'], index=index)
        f = pt.db['test']
        f.loc['2019-01-07'].shape == (5,3)

if __name__ == '__main__':
    unittest.main()
