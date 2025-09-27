import unittest
import datetime
import pandas as pd
import numpy as np
import pynto as pt

def get_test_data():
    return pd.DataFrame(np.arange(12).astype('int64').reshape(3,4),columns=['a','b','c','d'],
                            index=pd.date_range('1/1/2019',periods=3,freq='B'))

class TestRowIndexing(unittest.TestCase):
    def test_int_index(self):
        with self.subTest('Positive index'):
            self.assertEqual(pt.c(*range(5)).rows[0].shape[0],1)
        with self.subTest('Negative index'):
            self.assertEqual(pt.c(*range(5)).rows[-1].shape[0],1)
        with self.subTest('Date index'):
            self.assertEqual(pt.c(*range(5)).rows['2024-05-02'].shape[0],1)
        with self.subTest('Int range'):
            self.assertEqual(pt.c(*range(5)).rows[2:5].shape[0],3)
        with self.subTest('Int range neg'):
            self.assertEqual(pt.c(*range(5)).rows[-5:-2].shape[0],3)
        with self.subTest('Int range open start'):
            self.assertEqual(pt.c(*range(5)).rows[:5].shape[0],5)
        with self.subTest('Int range open stop'):
            self.assertEqual(pt.c(*range(5)).rows[-5:].shape[0],5)
        with self.subTest('Date range'):
            self.assertEqual(pt.c(*range(5)).rows['2023-04-01':'2023-04-05'].shape[0],2)
        with self.subTest('Date range pre epoque'):
            self.assertEqual(pt.c(*range(5)).rows['1969-12-15':'1970-01-08'].shape[0],18)
        with self.subTest('Date range open'):
            self.assertEqual(pt.c(*range(5)).rows[:'1970-01-08'].shape[0],5)
        with self.subTest('Specify per'):
            self.assertEqual(pt.c(*range(5)).rows[:1:'M'].index[0].date(),datetime.date(1970,1,30))

class TestColumnIndexing(unittest.TestCase):
    def test_int_index(self):
        a = pt.c(*range(5)).pull[0].values[0]
        self.assertTrue(np.array_equal(a, np.array([[1.,2., 3., 4., 0.]])))

    def test_int_slice(self):
        a = pt.c(*range(5)).pull[0:2].values[0]
        self.assertTrue(np.array_equal(a, np.array([[2., 3., 4., 0., 1.]])))

    def test_copy(self):
        a = pt.c(*range(5)).pull[2:4, True].halpha.neg.rows[0]
        self.assertTrue(np.array_equal(a.values, np.array([[0., 1., 2., 3., 4., 2., -3.]])))
        self.assertEqual(a.columns[-1], 'g')
        self.assertEqual(a.columns[-2], 'f')


    def test_discard(self):
        a = pt.c(*range(5)).pull[2:4, False, True].values[0]
        self.assertTrue(np.array_equal(a, np.array([[ 2., 3.]])))

    def test_copy_safety(self):
        a = pt.c(*range(5)).pull[2:4, True].neg.values[0]
        self.assertEqual( a[0,3], 3.)

    def test_header_filter(self):
        a = pt.c(*range(5)).hset('a,b,c,d,e').pull['d'].values[0]
        self.assertEqual( a[0,-1], 3.)

    def test_header_filter_duplicate(self):
        a = pt.c(*range(5)).hset('a,d,c,d,e').pull[['d'],False,True].rows[-1]
        self.assertTrue(np.array_equal(a, np.array([[ 1., 3.]])))

    def test_multiple_header_filter(self):
        a = pt.c(*range(5)).hset('a,b,c,d,e').pull[['c','d'], True].values[0]
        self.assertTrue(np.array_equal(a, np.array([[0., 1., 2., 3., 4., 2., 3.]])))

class TestOtherColumnIndexingWords(unittest.TestCase):
    def test_drop(self):
        a = pt.c(*range(5)).hset('a,d,c,d,e').drop['d'].rows[0]
        self.assertTrue(np.array_equal(a, np.array([[0., 2., 4.]])))

    def test_filter(self):
        a = pt.c(*range(5)).hset('a,d,c,d,e').filter['d'].rows[0]
        self.assertTrue(np.array_equal(a, np.array([[1., 3,]])))

    def test_dup(self):
        a = pt.c(*range(5)).hset('a,d,c,d,e').dup['d'].rows[0]
        self.assertTrue(np.array_equal(a, np.array([[0., 1., 2., 3, 4., 1., 3.]])))

class TestOperators(unittest.TestCase):
    def test_binary_add(self):
        a = pt.c(*range(5)).add.values[0]
        self.assertTrue(np.array_equal(a, np.array([[0., 1., 2., 7.]])))

    def test_cross_add(self):
        a = pt.c(*range(5)).add[:].values[0]
        self.assertTrue(np.array_equal(a, np.array([[10.]])))

    def test_rolling_add(self):
        a = pt.c(*range(5)).radd(3).values[0]
        self.assertTrue(np.array_equal(a, np.array([[0., 1., 2., 3., 12.]])))

    def test_rolling_std(self):
        a = pt.randn.rstd(50000).values[0]
        self.assertTrue(abs(a[0] - 1) < 0.01)

    def test_cumulative_std(self):
        a = pt.randn.cstd.values[:50000]
        self.assertTrue(abs(a[-1] - 1) < 0.01)

    def test_accumulate_add(self):
        a = pt.c(*range(5)).cadd.values[-5:]
        self.assertTrue(np.array_equal(a[-1], np.array([0., 1., 2., 3., 20.])))

    def test_unary(self):
        a = pt.r5.neg.values[0]
        self.assertTrue(np.array_equal(a[-1], np.array([0., 1., 2., 3., -4.])))

    def test_unary_multiple(self):
        a = pt.r5.neg[:].values[0]
        self.assertTrue(np.array_equal(a[-1], np.array([-0., -1., -2., -3., -4.])))

    def test_rank(self):
        df = pd.DataFrame(np.roll(np.arange(25),12).reshape((5,5)),
                        index=pt.periods.Periodicity.B[:5].to_index(),
                        columns=['a','b','c','d','e'])
        a = pt.pandas(df).rank.values[:]
        self.assertTrue(np.array_equal(a[2], np.array([3., 4., 0., 1., 2.])))
        self.assertTrue(np.array_equal(a[3], np.array([0., 1., 2., 3., 4.])))


class TestNullary(unittest.TestCase):
    df = pd.DataFrame(np.roll(np.arange(25),12).reshape((5,5)),
                        index=pt.periods.Periodicity.B[:5].to_index(),
                        columns=['a','b','c','d','e'])
    def test_pandas(self):
        a = pt.pandas(self.df).values[:]
        self.assertEqual(a.shape[0], 5)
        self.assertEqual(a.shape[1], 5)

    def test_dc(self):
        a = pt.dc.values[:5]
        self.assertTrue(np.array_equal(a.T, np.array([[1.,1.,3.,1.,1.]])))

class TestStackManipulation(unittest.TestCase):

    def test_interleave(self):
        a = pt.c(*range(6)).interleave.values[0]
        self.assertTrue(np.array_equal(a, np.array([[0.,3.,1.,4.,2.,5.]])))

class TestCombinators(unittest.TestCase):
    def test_locals_quote(self):
        result = pt.c5.c6.q(pt.add).call.rows['2019-01-01'].iloc[0,-1]
        self.assertEqual(result, 11)

    def test_inline_quote(self):
        result = pt.c(5).c(6).q.add.p.call.rows['2019-01-01'].iloc[0,-1]
        self.assertEqual(result, 11)

    def test_nested_quotes(self):
        result = pt.c(*range(9)).q.q.c100.sub.p.call.p.map(every=3).rows[-1]
        self.assertTrue(np.array_equal(result.values[-1],[0,1.,-98,3,4,-95,6,7,-92]))

    def test_nested_mix(self):
        result = pt.c(*range(9)).q.q(pt.c100.sub).call.p.map(every=3).rows[-1]
        self.assertTrue(np.array_equal(result.values[-1],[0,1.,-98,3,4,-95,6,7,-92]))

    def test_map(self):
        result = pt.c(5).c(6).q(pt.c(1).add).map.rows['2019-01-01']
        self.assertEqual(result.iloc[0,-1], 7)
        self.assertEqual(result.iloc[0,-2], 6)

    def test_hmap(self):
        result = pt.c(*range(10)).hset('a,b,a,a,b,a,a,b').q.add[:].p.hmap.hsort.values[-1]
        self.assertTrue(np.array_equal(result[-1],[26.,18,1]))

    def test_ifexists(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b').q.q.c100.mul.p.ifexists(3).add[:].p.hmap.hsort
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1],[818.,909,1]))

    def test_ifexistselse(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b') \
            .q.q.c100.div.p.q.c100.mul.p.ifexistselse(3).add[:].p.hmap.hsort
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1],[818.,909,.01]))

    def test_if(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b') \
            .q.q.c100.mul.p.ifheaders(lambda l: len(l) >= 3).add[:].p.hmap.hsort
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1],[818.,909,1]))

    def test_ifelse(self):
        expr = pt.r10.hset('a,b,a,a,b,a,a,b') \
            .q.q.c100.div.p.q.c100.mul.p.ifheaderselse(lambda l: len(l) >= 3).add[:].p.hmap.hsort
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1],[818.,909,.01]))

    def test_cleave(self):
        result = pt.c(4).q.neg.p.q.sqrt.p.cleave.rows[0]
        self.assertEqual(result.iloc[0,-1], 2)
        self.assertEqual(result.iloc[0,-2], -4)

    def test_partial(self):
        result = pt.r5.q.mul.p.partial.map.last.values
        self.assertTrue(np.array_equal(result[-1],[0.,4.,8.,12.]))

    def test_compose(self):
        result = pt.r5.q.c1.add.p.q.inv.p.compose.map.last.values
        self.assertTrue(np.array_equal(result[-1],[1,0.5,1/3,1/4,1/5]))


class TestHeaders(unittest.TestCase):
    def test_hset(self):
        self.assertEqual(pt.r4.hset('q','w','e','r').columns[1], 'w')
        self.assertEqual(pt.r6.hset('q','w','e','r').columns[1], 'c')

    def test_halpha(self):
        self.assertEqual(pt.r10.halpha.columns[1], 'b')

    def test_hformat(self):
        self.assertEqual(pt.r5.halpha.q.hformat('{0}x').p.map.columns[1], 'bx')

    def test_happly(self):
        self.assertEqual(pt.r5.halpha.q.happly(lambda h: h.upper()).p.map.columns[1], 'B')

class TestDataCleanup(unittest.TestCase):
    def test_join(self):
        result = pt.c1.nan.join('1970-01-10').c3.join('1970-01-15').values[6:11]
        self.assertTrue(np.array_equal(result.T[0],
               [1.,np.nan,np.nan,np.nan,3.], equal_nan=True))

    def test_fill(self):
        result = pt.c1.nan.join('1970-01-10').c3.join('1970-01-15').fill(2).values[6:11]
        self.assertTrue(np.array_equal(result.T[0],
               [1.,2.,2.,2.,3.], equal_nan=True))

    def test_ffill(self):
        result = pt.c1.nan.join('1970-01-10').c3.join('1970-01-15').ffill.values[6:11]
        self.assertTrue(np.array_equal(result.T[0],
               [1.,1.,1.,1.,3.], equal_nan=True))

    def test_ffill_leaveend(self):
        result = pt.c1.nan.join('1970-01-10').c3.join('1970-01-13') \
            .nan.join('1970-01-15').ffill(leave_end=True).values[6:11]
        self.assertTrue(np.array_equal(result.T[0],
               [1.,1.,3.,3.,np.nan], equal_nan=True))

    def test_ffill_lookback(self):
        df = pt.c1.nan.join('1970-01-10').c3.join('1970-01-13') \
            .nan.join('1970-01-15').ffill.rows[6:11]
        result = pt.pandas(df).ffill(leave_end=False).values[7:11]
        self.assertTrue(np.array_equal(result.T[0], [1.,3.,3.,3.]))

class TestCornerCases(unittest.TestCase):
    def test_siblings_different_ranges(self):
        result = pt.c1.c2.neg[:].hset('a,b').radd.roll.cadd.start('2025-01-01').values['2025-05-14']
        self.assertTrue(np.array_equal(result[-1], [-4.,-96.], equal_nan=True))

    def test_siblings_no_drop(self):
        result = pt.c1.c2.neg[:].hset('a,b').radd.roll.cadd.start('2025-01-01') \
                .rank.drop.values['2025-05-14']
        self.assertEqual(result[0,0], 1.)


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
        pt.db['test'] = pt.r10.halpha.rows[:5]

    def tearDown(self):
        del(pt.db['test'])

    def test_read_frame(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 10)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.sum().sum(), 225)

    def test_read_range(self):
        f = pt.db['test','1970-01-05':'1970-01-13']
        self.assertEqual(f.shape[0], 6)

    def test_read_column(self):
        f = pt.db['test#b']
        self.assertEqual(f.shape[1], 1)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.sum().sum(), 5)

    def test_write_column(self):
        pt.db['test#e'] = pt.c99.rows[:5]
        f = pt.db['test']
        self.assertTrue(np.array_equal(f.values[-1], np.array([0.,1.,2.,3.,99.,5.,6.,7.,8.,9.])))

    def test_overwrite(self):
        pt.db['test'] = pt.r10.rev.halpha.rows[:5]
        f = pt.db['test']
        self.assertEqual(f.values[0,0], 9.)

    def test_append_rows(self):
        pt.db['test'] = pt.r10.halpha.rows[5:10]
        f = pt.db['test']
        self.assertEqual(f.shape[1], 10)
        self.assertEqual(f.shape[0], 10)

    def test_append_cols(self):
        pt.db['test'] = pt.r2.hset('k,l').rows[5:10]
        f = pt.db['test']
        self.assertEqual(f.shape[1], 12)
        self.assertEqual(f.shape[0], 10)

class IntTest(DbTest):
    def setUp(self):
        pt.db['test'] = pt.r10.halpha.rows[:5].astype(np.int64)

    def tearDown(self):
        del(pt.db['test'])

    def test_read_frame(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 10)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.values.dtype, np.int64)

class BoolTest(DbTest):
    def setUp(self):
        pt.db['test'] = pt.r10.halpha.rows[:5].astype(np.bool)

    def tearDown(self):
        del(pt.db['test'])

    def test_read_frame(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 10)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.values.dtype, np.bool)

class SeriesTest(DbTest):
    def setUp(self):
        pt.db['test'] = pt.r10.halpha.rows[:5].iloc[:,0]

    def tearDown(self):
        del(pt.db['test'])

    def test_read(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 1)
        self.assertEqual(f.shape[0], 5)


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
