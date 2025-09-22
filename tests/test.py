import unittest
import datetime
import os
import pandas as pd
import numpy as np
import pynto as pt
from pynto.database import SQLiteConnection, Db

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


class SQLiteDbTest(unittest.TestCase):
    """Test SQLite3 database implementation."""
    
    def setUp(self):
        """Set up in-memory SQLite database for testing."""
        # Create a fresh SQLite connection for each test
        self.sqlite_conn = SQLiteConnection()
        self.db = Db(connection=self.sqlite_conn)
    
    def tearDown(self):
        """Clean up after each test."""
        # Close the SQLite connection
        if hasattr(self.sqlite_conn, '_conn'):
            self.sqlite_conn._conn.close()

    def test_save_and_read_dataframe(self):
        """Test saving and reading a DataFrame to/from SQLite."""
        # Create test data
        df = get_test_data()
        
        # Save to SQLite database
        self.db['test_df'] = df
        
        # Read back from database
        result = self.db['test_df']
        
        # Verify data integrity
        self.assertEqual(result.shape, df.shape)
        self.assertTrue(np.array_equal(result.values, df.values))
        self.assertTrue(all(result.columns == df.columns))
    
    def test_save_and_read_series(self):
        """Test saving and reading a Series to/from SQLite."""
        # Create test series
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], 
                          index=pd.date_range('2020-01-01', periods=5, freq='B'),
                          name='test_series')
        
        # Save to SQLite database
        self.db['test_series'] = series
        
        # Read back from database
        result = self.db['test_series']
        
        # Verify data integrity
        self.assertEqual(result.shape, (5, 1))
        self.assertTrue(np.array_equal(result.values.flatten(), series.values))
    
    def test_save_multiple_dataframes(self):
        """Test saving multiple DataFrames with different keys."""
        # Create test data
        df1 = get_test_data()
        df2 = pd.DataFrame(np.random.randn(4, 3), 
                          columns=['x', 'y', 'z'],
                          index=pd.date_range('2021-01-01', periods=4, freq='B'))
        
        # Save both DataFrames
        self.db['df1'] = df1
        self.db['df2'] = df2
        
        # Read back both DataFrames
        result1 = self.db['df1']
        result2 = self.db['df2']
        
        # Verify data integrity
        self.assertEqual(result1.shape, df1.shape)
        self.assertEqual(result2.shape, df2.shape)
        self.assertTrue(np.array_equal(result1.values, df1.values))
        self.assertTrue(np.allclose(result2.values, df2.values))
    
    def test_delete_dataframe(self):
        """Test deleting DataFrames from SQLite database."""
        # Create and save test data
        df = get_test_data()
        self.db['test_delete'] = df
        
        # Verify it was saved
        result = self.db['test_delete']
        self.assertEqual(result.shape, df.shape)
        
        # Delete the DataFrame
        del self.db['test_delete']
        
        # Verify it was deleted (should raise KeyError)
        with self.assertRaises(KeyError):
            self.db['test_delete']
    
    def test_overwrite_dataframe(self):
        """Test overwriting an existing DataFrame."""
        # Save initial data
        df1 = get_test_data()
        self.db['test_overwrite'] = df1
        
        # Overwrite with new data
        df2 = pd.DataFrame(np.ones((2, 2)), 
                          columns=['col1', 'col2'],
                          index=pd.date_range('2022-01-01', periods=2, freq='B'))
        self.db['test_overwrite'] = df2
        
        # Read back and verify it's the new data
        result = self.db['test_overwrite']
        self.assertEqual(result.shape, (786,6))
    
    def test_range_query(self):
        """Test querying data with date ranges."""
        # Create data with specific date range
        dates = pd.date_range('2020-01-01', periods=10, freq='B')
        df = pd.DataFrame(np.arange(20).reshape(10, 2), 
                         columns=['A', 'B'], 
                         index=dates)
        
        # Save to database
        self.db['test_range'] = df
        
        # Query a specific range
        start_date = '2020-01-03'
        end_date = '2020-01-10'
        result = self.db['test_range', start_date:end_date]
        
        # Verify the range query worked
        expected_rows = 5  # 2020-01-03 to 2020-01-07 inclusive
        self.assertEqual(result.shape[0], expected_rows)
        self.assertEqual(result.shape[1], 2)
    
    def test_column_specific_save_and_read(self):
        """Test saving and reading specific columns."""
        # Create test data
        df = get_test_data()
        self.db['test_col'] = df
        
        # Save a specific column with column key
        series = pd.Series([100, 200, 300], 
                          index=pd.date_range('2019-01-01', periods=3, freq='B'),
                          name='new_col')
        self.db['test_col#new_col'] = series
        
        # Read back the specific column
        result_col = self.db['test_col#new_col']
        self.assertEqual(result_col.shape, (3, 1))
        
        # Read back the full DataFrame (should include new column)
        result_full = self.db['test_col']
        self.assertEqual(result_full.shape[1], 5)  # Original 4 + 1 new column
    
    def test_integer_and_boolean_datatypes(self):
        """Test handling of different data types."""
        # Test integer data
        int_df = pd.DataFrame(np.arange(6, dtype=np.int64).reshape(2, 3),
                             columns=['int_a', 'int_b', 'int_c'],
                             index=pd.date_range('2020-01-01', periods=2, freq='B'))
        self.db['test_int'] = int_df
        
        # Test boolean data
        bool_df = pd.DataFrame(np.array([True, False, True, False]).reshape(2, 2),
                              columns=['bool_a', 'bool_b'],
                              index=pd.date_range('2020-01-01', periods=2, freq='B'))
        self.db['test_bool'] = bool_df
        
        # Read back and verify data types
        int_result = self.db['test_int']
        bool_result = self.db['test_bool']
        
        self.assertEqual(int_result.values.dtype, np.int64)
        self.assertEqual(bool_result.values.dtype, np.bool_)
        self.assertTrue(np.array_equal(int_result.values, int_df.values))
        self.assertTrue(np.array_equal(bool_result.values, bool_df.values))
    
    def test_all_keys_listing(self):
        """Test listing all keys in the database."""
        # Save multiple DataFrames
        df1 = get_test_data()
        df2 = pd.DataFrame(np.random.randn(2, 2), 
                          columns=['col1', 'col2'],
                          index=pd.date_range('2020-02-01', periods=2, freq='B'))
        
        self.db['key1'] = df1
        self.db['key2'] = df2
        
        # Get all keys
        all_keys = self.db.all_keys()
        
        # Verify keys are present
        self.assertIn('key1', all_keys)
        self.assertIn('key2', all_keys)
    
    def test_delete_all(self):
        """Test deleting all data from database."""
        # Save some test data
        df1 = get_test_data()
        df2 = pd.DataFrame(np.random.randn(2, 2), 
                          columns=['col1', 'col2'],
                          index=pd.date_range('2020-02-01', periods=2, freq='B'))
        
        self.db['key1'] = df1
        self.db['key2'] = df2
        
        # Verify data exists
        self.assertEqual(len(self.db.all_keys()), 2)
        
        # Delete all data
        self.db.delete_all()
        
        # Verify all data is gone
        self.assertEqual(len(self.db.all_keys()), 0)


class SQLiteDefaultConnectionTest(unittest.TestCase):
    """Test that SQLite is used by default when Redis env vars are not set."""
    
    def setUp(self):
        """Save original environment and clear Redis env vars."""
        self.original_env = {}
        redis_env_vars = ['PYNTO_REDIS_HOST', 'PYNTO_REDIS_PATH', 'PYNTO_REDIS_PORT', 'PYNTO_REDIS_PASSWORD']
        
        for var in redis_env_vars:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]
        
        # Clear the global client to force recreation
        if hasattr(pt.database, '_CLIENT'):
            pt.database._CLIENT = None
    
    def tearDown(self):
        """Restore original environment."""
        # Restore original environment variables
        for var, value in self.original_env.items():
            os.environ[var] = value
        
        # Clear the global client
        if hasattr(pt.database, '_CLIENT'):
            pt.database._CLIENT = None
    
    def test_default_to_sqlite(self):
        """Test that SQLite is used by default when Redis env vars are not set."""
        # Get a client (should default to SQLite)
        client = pt.get_client()
        
        # Verify it's using SQLite connection
        self.assertIsInstance(client.connection, SQLiteConnection)
        
        # Test basic functionality
        test_data = get_test_data()
        client['test_default_sqlite'] = test_data
        
        result = client['test_default_sqlite']
        self.assertEqual(result.shape, test_data.shape)
        self.assertTrue(np.array_equal(result.values, test_data.values))
        
        # Clean up
        del client['test_default_sqlite']


if __name__ == '__main__':
    unittest.main()
