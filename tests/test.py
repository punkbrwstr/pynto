import unittest
import datetime
import pandas as pd
import numpy as np
import pynto as pt


def get_test_data():
    return pd.DataFrame(
        np.arange(12).astype('int64').reshape(3, 4),
        columns=['a', 'b', 'c', 'd'],
        index=pd.date_range('1/1/2019', periods=3, freq='B'),
    )


class TestRowIndexing(unittest.TestCase):
    def test_int_index(self):
        with self.subTest('Positive index'):
            self.assertEqual(pt.r5.rows[0].shape[0], 1)
        with self.subTest('Negative index'):
            self.assertEqual(pt.r5.rows[-1].shape[0], 1)
        with self.subTest('Date index'):
            self.assertEqual(pt.r5.rows['2024-05-02'].shape[0], 1)
        with self.subTest('Int range'):
            self.assertEqual(pt.r5.rows[2:5].shape[0], 3)
        with self.subTest('Int range neg'):
            self.assertEqual(pt.r5.rows[-5:-2].shape[0], 3)
        with self.subTest('Int range open start'):
            self.assertEqual(pt.r5.rows[:5].shape[0], 5)
        with self.subTest('Int range open stop'):
            self.assertEqual(pt.r5.rows[-5:].shape[0], 5)
        with self.subTest('Date range'):
            self.assertEqual(
                pt.r5.rows['2023-04-01':'2023-04-05'].shape[0], 2
            )
        with self.subTest('Date range pre epoque'):
            self.assertEqual(
                pt.r5.rows['1969-12-15':'1970-01-08'].shape[0], 18
            )
        with self.subTest('Date range open'):
            self.assertEqual(pt.r5.rows[:'1970-01-08'].shape[0], 5)
        with self.subTest('Specify per'):
            self.assertEqual(
                pt.r5.rows[:1:'M'].index[0].date(), datetime.date(1970, 1, 30)
            )


class TestColumnIndexing(unittest.TestCase):
    def test_int_index(self):
        a = pt.r5.pull[0].values[0]
        self.assertTrue(np.array_equal(a, np.array([[1.0, 2.0, 3.0, 4.0, 0.0]])))

    def test_int_slice(self):
        a = pt.r5.pull[0:2].values[0]
        self.assertTrue(np.array_equal(a, np.array([[2.0, 3.0, 4.0, 0.0, 1.0]])))

    def test_copy(self):
        a = pt.r5.pull[2:4, True].halpha.neg.rows[0]
        self.assertTrue(
            np.array_equal(a.values, np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 2.0, -3.0]]))
        )
        self.assertEqual(a.columns[-1], 'g')
        self.assertEqual(a.columns[-2], 'f')

    def test_discard(self):
        a = pt.r5.pull[2:4, False, True].values[0]
        self.assertTrue(np.array_equal(a, np.array([[2.0, 3.0]])))

    def test_discard_original(self):
        a = pt.c1.dup.keep[1].last
        self.assertEqual(a.values[0],1.)

    def test_discard_copy(self):
        a = pt.c1.dup.keep[0].last
        self.assertEqual(a.values[0],1.)

    def test_copy_safety(self):
        a = pt.r5.pull[2:4, True].neg.values[0]
        self.assertEqual(a[0, 3], 3.0)

    def test_header_filter(self):
        a = pt.r5.hset('a,b,c,d,e').pull['d'].values[0]
        self.assertEqual(a[0, -1], 3.0)

    def test_header_filter_duplicate(self):
        a = pt.r5.hset('a,d,c,d,e').pull[['d'], False, True].rows[-1]
        self.assertTrue(np.array_equal(a, np.array([[1.0, 3.0]])))

    def test_multiple_header_filter(self):
        a = pt.r5.hset('a,b,c,d,e').pull[['c', 'd'], True].values[0]
        self.assertTrue(
            np.array_equal(a, np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0]]))
        )

    def test_multiple_header_with_groups(self):
        a = pt.r4.inc[:].halpha.div[['a', 'c'], True].radd.last.values
        self.assertTrue(np.array_equal(a, np.array([1.0, 2.0, 3.0, 4.0, 2 / 3.0])))

    def test_multiple_header_with_two_groups(self):
        a = pt.r4.inc[:].halpha.div[['a', 'c'], True].radd[:].last.values
        self.assertTrue(np.array_equal(a, np.array([2.0, 4.0, 6.0, 8.0, 2 / 3.0])))


class TestOtherColumnIndexingWords(unittest.TestCase):
    def test_drop(self):
        a = pt.r5.hset('a,d,c,d,e').drop['d'].rows[0]
        self.assertTrue(np.array_equal(a, np.array([[0.0, 2.0, 4.0]])))

    def test_keep(self):
        a = pt.r5.hset('a,d,c,d,e').keep['d'].rows[0]
        self.assertTrue(
            np.array_equal(
                a,
                np.array(
                    [
                        [
                            1.0,
                            3,
                        ]
                    ]
                ),
            )
        )

    def test_dup(self):
        a = pt.r5.hset('a,d,c,d,e').dup['d'].rows[0]
        self.assertTrue(
            np.array_equal(a, np.array([[0.0, 1.0, 2.0, 3, 4.0, 1.0, 3.0]]))
        )


class TestOperators(unittest.TestCase):
    def test_binary_add(self):
        a = pt.r5.add.values[0]
        self.assertTrue(np.array_equal(a, np.array([[0.0, 1.0, 2.0, 7.0]])))

    def test_cross_add(self):
        a = pt.r5.add[:].values[0]
        self.assertTrue(np.array_equal(a, np.array([[10.0]])))

    def test_rolling_add(self):
        a = pt.r5.radd(3).values[0]
        self.assertTrue(np.array_equal(a, np.array([[0.0, 1.0, 2.0, 3.0, 12.0]])))

    def test_rolling_std(self):
        a = pt.randn.rstd(50000).values[0]
        self.assertTrue(abs(a[0] - 1) < 0.01)

    def test_cumulative_std(self):
        a = pt.randn.cstd.values[:50000]
        self.assertTrue(abs(a[-1] - 1) < 0.01)

    def test_accumulate_add(self):
        a = pt.r5.cadd.values[-5:]
        self.assertTrue(np.array_equal(a[-1], np.array([0.0, 1.0, 2.0, 3.0, 20.0])))

    def test_unary(self):
        a = pt.r5.neg.values[0]
        self.assertTrue(np.array_equal(a[-1], np.array([0.0, 1.0, 2.0, 3.0, -4.0])))

    def test_unary_multiple(self):
        a = pt.r5.neg[:].values[0]
        self.assertTrue(np.array_equal(a[-1], np.array([-0.0, -1.0, -2.0, -3.0, -4.0])))

    def test_rank(self):
        df = pd.DataFrame(
            np.roll(np.arange(25), 12).reshape((5, 5)),
            index=pt.periods.Periodicity.B[:5].to_index(),
            columns=['a', 'b', 'c', 'd', 'e'],
        )
        a = pt.from_pandas(df).rank.values[:]
        self.assertTrue(np.array_equal(a[2], np.array([3.0, 4.0, 0.0, 1.0, 2.0])))
        self.assertTrue(np.array_equal(a[3], np.array([0.0, 1.0, 2.0, 3.0, 4.0])))


class TestNullary(unittest.TestCase):
    df = pd.DataFrame(
        np.roll(np.arange(25), 12).reshape((5, 5)),
        index=pt.periods.Periodicity.B[:5].to_index(),
        columns=['a', 'b', 'c', 'd', 'e'],
    )

    def test_from_pandas(self):
        a = pt.from_pandas(self.df).values[:]
        self.assertEqual(a.shape[0], 5)
        self.assertEqual(a.shape[1], 5)

    def test_day_count(self):
        a = pt.day_count.values[:5]
        self.assertTrue(np.array_equal(a.T, np.array([[1.0, 1.0, 3.0, 1.0, 1.0]])))


class TestStackManipulation(unittest.TestCase):
    def test_interleave(self):
        a = pt.r6.interleave.values[0]
        self.assertTrue(np.array_equal(a, np.array([[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]])))

    def test_id(self):
        a = pt.r5.id.values[0]
        self.assertTrue(np.array_equal(a, np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])))

    def test_nip(self):
        # nip keeps only the top column
        a = pt.r5.nip.values[0]
        self.assertTrue(np.array_equal(a, np.array([[4.0]])))

    def test_swap(self):
        # swap swaps the top two columns
        a = pt.r5.swap.values[0]
        self.assertTrue(np.array_equal(a, np.array([[0.0, 1.0, 2.0, 4.0, 3.0]])))


class TestCombinators(unittest.TestCase):
    def test_locals_quote(self):
        result = pt.c5.c6.q(pt.add).call.rows['2019-01-01'].iloc[0, -1]
        self.assertEqual(result, 11)

    def test_inline_quote(self):
        result = pt.c(5).c(6).q.add.p.call.rows['2019-01-01'].iloc[0, -1]
        self.assertEqual(result, 11)

    def test_nested_quotes(self):
        result = pt.r9.q.q.c100.sub.p.call.p.map(every=3).rows[-1]
        self.assertTrue(
            np.array_equal(result.values[-1], [0, 1.0, -98, 3, 4, -95, 6, 7, -92])
        )

    def test_nested_mix(self):
        result = pt.r9.q.q(pt.c100.sub).call.p.map(every=3).rows[-1]
        self.assertTrue(
            np.array_equal(result.values[-1], [0, 1.0, -98, 3, 4, -95, 6, 7, -92])
        )

    def test_map(self):
        result = pt.c(5).c(6).q(pt.c(1).add).map.rows['2019-01-01']
        self.assertEqual(result.iloc[0, -1], 7)
        self.assertEqual(result.iloc[0, -2], 6)

    def test_hmap(self):
        result = pt.r10.hset('c,c,a,b,a,a,b,a,a,b').q.add[:].p.hmap.hsort.values[-1]
        self.assertTrue(np.array_equal(result[-1], [26.0, 18, 1]))

    def test_ifexists(self):
        expr = (
            pt.r10.hset('c,c,a,b,a,a,b,a,a,b')
            .q.q.c100.mul.p.ifexists(3)
            .add[:]
            .p.hmap.hsort
        )
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1], [818.0, 909, 1]))

    def test_ifexistselse(self):
        expr = (
            pt.r10.hset('c,c,a,b,a,a,b,a,a,b')
            .q.q.c100.div.p.q.c100.mul.p.ifexistselse(3)
            .add[:]
            .p.hmap.hsort
        )
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1], [818.0, 909, 0.01]))

    def test_if(self):
        expr = (
            pt.r10.hset('c,c,a,b,a,a,b,a,a,b')
            .q.q.c100.mul.p.ifheaders(lambda length: len(length) >= 3)
            .add[:]
            .p.hmap.hsort
        )
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1], [818.0, 909, 1]))

    def test_ifelse(self):
        expr = (
            pt.r10.hset('c,c,a,b,a,a,b,a,a,b')
            .q.q.c100.div.p.q.c100.mul.p.ifheaderselse(lambda length: len(length) >= 3)
            .add[:]
            .p.hmap.hsort
        )
        result = expr.values[-1]
        self.assertTrue(np.array_equal(result[-1], [818.0, 909, 0.01]))

    def test_cleave(self):
        result = pt.c(4).q.neg.p.q.sqrt.p.cleave.rows[0]
        self.assertEqual(result.iloc[0, -1], 2)
        self.assertEqual(result.iloc[0, -2], -4)

    def test_partial(self):
        result = pt.r5.q.mul.p.partial.map.last.values
        self.assertTrue(np.array_equal(result, [0.0, 4.0, 8.0, 12.0]))

    def test_compose(self):
        result = pt.r5.q.c1.add.p.q.inv.p.compose.map.last.values
        self.assertTrue(np.array_equal(result, [1, 0.5, 1 / 3, 1 / 4, 1 / 5]))

    def test_repeat(self):
        # Applies q(c1.add) three times: 0 + 1 + 1 + 1 = 3
        result = pt.c0.q(pt.c1.add).repeat(3).rows[0]
        self.assertEqual(result.iloc[0, -1], 3.0)


class TestHeaders(unittest.TestCase):
    def test_hset(self):
        self.assertEqual(pt.r4.hset('q', 'w', 'e', 'r').columns[1], 'w')
        self.assertEqual(pt.r6.hset('q', 'w', 'e', 'r').columns[1], 'c1')

    def test_halpha(self):
        self.assertEqual(pt.r10.halpha.columns[1], 'b')

    def test_hformat(self):
        self.assertEqual(pt.r5.halpha.q.hformat('{0}x').p.map.columns[1], 'bx')

    def test_happly(self):
        self.assertEqual(
            pt.r5.halpha.q.happly(lambda h: h.upper()).p.map.columns[1], 'B'
        )

    def test_hsetall(self):
        # hsetall repeats the given headers cyclically
        cols = pt.r6.hsetall('x', 'y').columns
        self.assertEqual(cols[0], 'x')
        self.assertEqual(cols[1], 'y')
        self.assertEqual(cols[2], 'x')
        self.assertEqual(cols[4], 'x')


class TestDataCleanup(unittest.TestCase):
    def test_join(self):
        result = pt.c1.nan.join('1970-01-10').c3.join('1970-01-15').values[6:11]
        self.assertTrue(
            np.array_equal(
                result.T[0], [1.0, np.nan, np.nan, np.nan, 3.0], equal_nan=True
            )
        )

    def test_fill(self):
        result = pt.c1.nan.join('1970-01-10').c3.join('1970-01-15').fill(2).values[6:11]
        self.assertTrue(
            np.array_equal(result.T[0], [1.0, 2.0, 2.0, 2.0, 3.0], equal_nan=True)
        )

    def test_ffill(self):
        result = pt.c1.nan.join('1970-01-10').c3.join('1970-01-15').ffill.values[6:11]
        self.assertTrue(
            np.array_equal(result.T[0], [1.0, 1.0, 1.0, 1.0, 3.0], equal_nan=True)
        )

    def test_ffill_leaveend(self):
        result = (
            pt.c1.nan.join('1970-01-10')
            .c3.join('1970-01-13')
            .nan.join('1970-01-15')
            .ffill(leave_end=True)
            .values[6:11]
        )
        self.assertTrue(
            np.array_equal(result.T[0], [1.0, 1.0, 3.0, 3.0, np.nan], equal_nan=True)
        )

    def test_ffill_lookback(self):
        df = (
            pt.c1.nan.join('1970-01-10')
            .c3.join('1970-01-13')
            .nan.join('1970-01-15')
            .ffill.rows[6:11]
        )
        result = pt.from_pandas(df).ffill(leave_end=False).values['1970-01-10':'1970-01-16']
        self.assertTrue(np.array_equal(result.T[0], [1.0, 3.0, 3.0, 3.0]))

    def test_fillfirst(self):
        # fillfirst fills the first row of the evaluated range with the most recent
        # non-NaN value from the lookback window before the range start.
        # rows[7:8] starts at Jan 12 (first B day after Jan 10 join boundary).
        # fillfirst(5) looks back 5 B days and finds Jan 9 = 1.0.
        result = pt.c1.nan.join('1970-01-10').fillfirst(5).rows[7:8]
        self.assertEqual(result.iloc[0, 0], 1.0)
        # Without fillfirst, row 7 (Jan 12) is NaN (past the join point)
        result_raw = pt.c1.nan.join('1970-01-10').rows[7:8]
        self.assertTrue(np.isnan(result_raw.iloc[0, 0]))

    def test_sync(self):
        # sync sets the entire row to NaN when any column has NaN
        result = pt.c1.nan.join('1970-01-10').c3.sync.values[6:8]
        # Row 6 (Jan 9): col1=1.0, col2=3.0 — both non-NaN, unchanged
        self.assertEqual(result[0, 0], 1.0)
        self.assertEqual(result[0, 1], 3.0)
        # Row 7 (Jan 12): col1=NaN — entire row becomes NaN
        self.assertTrue(np.isnan(result[1, 0]))
        self.assertTrue(np.isnan(result[1, 1]))

    def test_zero_first(self):
        # zero_first replaces the first (oldest) value with 0, rest unchanged
        result = pt.c5.zero_first.values[:2]
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 0], 5.0)

    def test_zero_to_na(self):
        # zero_to_na replaces zeros with NaN
        result = pt.c0.zero_to_na.values[0]
        self.assertTrue(np.isnan(result[0, 0]))


class TestUnaryOps(unittest.TestCase):
    def test_lnot(self):
        self.assertTrue(bool(pt.c0.lnot.values[0][0, 0]))   # not 0 = True
        self.assertFalse(bool(pt.c1.lnot.values[0][0, 0]))  # not 1 = False

    def test_sign(self):
        result = pt.c(5).c(-3).c0.sign[:].values[0]
        self.assertTrue(np.array_equal(result, [[1.0, -1.0, 0.0]]))

    def test_expm1(self):
        # exp(0) - 1 = 0
        result = pt.c0.expm1.values[0]
        self.assertAlmostEqual(result[0, 0], 0.0)

    def test_log1p(self):
        # ln(1 + 0) = 0
        result = pt.c0.log1p.values[0]
        self.assertAlmostEqual(result[0, 0], 0.0)

    def test_inc(self):
        result = pt.c5.inc.values[0]
        self.assertEqual(result[0, 0], 6.0)

    def test_dec(self):
        result = pt.c5.dec.values[0]
        self.assertEqual(result[0, 0], 4.0)


class TestNanIgnoring(unittest.TestCase):
    def test_nadd(self):
        # add[:] with any NaN in row → NaN output
        add_result = pt.c1.nan.add[:].values[0]
        self.assertTrue(np.isnan(add_result[0, 0]))
        # nadd[:] ignores NaN — only outputs NaN when ALL inputs are NaN
        nadd_result = pt.c1.nan.nadd[:].values[0]
        self.assertEqual(nadd_result[0, 0], 1.0)


class TestReverseCumulative(unittest.TestCase):
    def test_rcadd(self):
        # Reverse cumulative sum of constant 1: each row = number of remaining periods
        result = pt.c1.rcadd.values[-5:]
        self.assertEqual(result[0, 0], 5.0)   # oldest of last 5 has 5 remaining
        self.assertEqual(result[-1, 0], 1.0)  # most recent has only itself


class TestRollingSpecial(unittest.TestCase):
    # 20 rows of hardcoded random data (two columns x, y).
    # Expected values were derived by running rolling_cov / rolling_cor / rolling_ewma
    # on the same 7-row window that pynto uses (lookback = max(5, int(5*1.25)) = 6).
    DATA = np.array([
        [ 0.304717, -1.039984], [ 0.750451,  0.940565], [-1.951035, -1.302180],
        [ 0.127840, -0.316243], [-0.016801, -0.853044], [ 0.879398,  0.777792],
        [ 0.066031,  1.127241], [ 0.467509, -0.859292], [ 0.368751, -0.958883],
        [ 0.878450, -0.049926], [-0.184862, -0.680930], [ 1.222541, -0.154529],
        [-0.428328, -0.352134], [ 0.532309,  0.365444], [ 0.412733,  0.430821],
        [ 2.141648, -0.406415], [-0.512243, -0.813773], [ 0.615979,  1.128972],
        [-0.113947, -0.840156], [-0.824481,  0.650593],
    ])
    WINDOW = 5
    # Expected outputs at the last row, computed with rolling_cov/cor/ewma on the
    # 7-row lookback window (rows 13..19):
    COV_EXPECTED = -0.025080091449840018
    COR_EXPECTED = -0.034949284471091864
    EWM_EXPECTED = -0.05362102194787367

    def _df(self, cols=None):
        idx = pt.periods.Periodicity.B[:20].to_index()
        d = self.DATA if cols is None else self.DATA[:, cols]
        c = ['x', 'y'] if cols is None else [['x', 'y'][i] for i in cols]
        return pd.DataFrame(d, columns=c, index=idx)

    def test_rcov(self):
        last = self._df().index[-1]
        result = pt.from_pandas(self._df()).rcov(self.WINDOW).values[last]
        self.assertAlmostEqual(result[0, 0], self.COV_EXPECTED, places=10)

    def test_rcor(self):
        last = self._df().index[-1]
        result = pt.from_pandas(self._df()).rcor(self.WINDOW).values[last]
        self.assertAlmostEqual(result[0, 0], self.COR_EXPECTED, places=10)

    def test_ewm_mean(self):
        last = self._df([0]).index[-1]
        result = pt.from_pandas(self._df([0])).ewm_mean(self.WINDOW).values[last]
        self.assertAlmostEqual(result[0, 0], self.EWM_EXPECTED, places=10)


class TestCornerCases(unittest.TestCase):
    def test_siblings_different_ranges(self):
        result = (
            pt.c1.c2.neg[:]
            .hset('a,b')
            .radd.roll.cadd.set_start('2025-01-01')
            .values['2025-05-14']
        )
        self.assertTrue(np.array_equal(result[-1], [-4.0, -96.0], equal_nan=True))

    def test_siblings_no_drop(self):
        result = (
            pt.c1.c2.neg[:]
            .hset('a,b')
            .radd.roll.cadd.set_start('2025-01-01')
            .rank.drop.values['2025-05-14']
        )
        self.assertEqual(result[0, 0], 1.0)


class DbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            del pt.db['test']
        except KeyError:
            pass

    @classmethod
    def tearDownClass(cls):
        pass


class FrameTest(DbTest):
    def setUp(self):
        pt.db['test'] = pt.r10.halpha.rows[:5]

    def tearDown(self):
        del pt.db['test']

    def test_read_frame(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 10)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.sum().sum(), 225)

    def test_read_column(self):
        f = pt.db['test#b']
        self.assertEqual(f.shape[1], 1)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.sum().sum(), 5)

    def test_write_column(self):
        pt.db['test#e'] = pt.c99.rows[:5]
        f = pt.db['test']
        self.assertTrue(
            np.array_equal(
                f.values[-1],
                np.array([0.0, 1.0, 2.0, 3.0, 99.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            )
        )

    def test_overwrite(self):
        pt.db['test'] = pt.r10.rev.halpha.rows[:5]
        f = pt.db['test']
        self.assertEqual(f.values[0, 0], 9.0)

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
        del pt.db['test']

    def test_read_frame(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 10)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.values.dtype, np.int64)


class BoolTest(DbTest):
    def setUp(self):
        pt.db['test'] = pt.r10.halpha.rows[:5].astype(np.bool)

    def tearDown(self):
        del pt.db['test']

    def test_read_frame(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 10)
        self.assertEqual(f.shape[0], 5)
        self.assertEqual(f.values.dtype, np.bool)


class SeriesTest(DbTest):
    def setUp(self):
        pt.db['test'] = pt.r10.halpha.rows[:5].iloc[:, 0]

    def tearDown(self):
        del pt.db['test']

    def test_read(self):
        f = pt.db['test']
        self.assertEqual(f.shape[1], 1)
        self.assertEqual(f.shape[0], 5)


class MultiIndexFrameTest(DbTest):
    def tearDown(self):
        del pt.db['test']

    def test_square(self):
        index = pd.MultiIndex.from_product(
            [pd.date_range('1/1/2019', periods=5, freq='B'), ['x', 'y', 'z']]
        )
        pt.db['test'] = pd.DataFrame(
            np.arange(45).astype('int64').reshape(15, 3),
            columns=['a', 'b', 'c'],
            index=index,
        )
        f = pt.db['test']
        f.loc['2019-01-07'].shape == (3, 3)
        f.loc['2019-01-07'].iloc[2, 2] == 44
        f.loc['2019-01-07', 'x']['a'] == 36

    def test_rect(self):
        index = pd.MultiIndex.from_product(
            [
                pd.date_range('1/1/2019', periods=5, freq='B'),
                ['x', 'y', 'z', 'aa', 'bb'],
            ]
        )
        pt.db['test'] = pd.DataFrame(
            np.arange(75).astype('int64').reshape(25, 3),
            columns=['a', 'b', 'c'],
            index=index,
        )
        f = pt.db['test']
        f.loc['2019-01-07'].shape == (5, 3)


class TestResample(DbTest):
    # resample_* methods act on saved (Redis) columns when the stored periodicity
    # differs from the requested output periodicity.
    #
    # Dataset: Jan + Feb 1970, B-frequency.
    #   Jan (22 B days): values 1..21 for Jan1..Jan29, NaN for Jan30 (last B day).
    #   Feb (20 B days): values 1..20 for all Feb days (no NaNs).
    KEY = 'resample_test'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        jan_idx = pt.periods.Periodicity.B[:22].to_index()   # Jan 1 – Jan 30
        feb_idx = pt.periods.Periodicity.B[22:42].to_index() # Feb 2 – Feb 27
        jan_vals = list(range(1, 22)) + [np.nan]  # 1..21, then NaN on Jan 30
        feb_vals = list(range(1, 21))              # 1..20
        idx = jan_idx.append(feb_idx)
        vals = jan_vals + feb_vals
        s = pd.Series(vals, index=idx, name='v', dtype='float64')
        pt.db[cls.KEY] = s

    @classmethod
    def tearDownClass(cls):
        del pt.db[cls.KEY]

    def test_per_last(self):
        # LAST fills forward: Jan 30 is NaN so the filled value is Jan 29 = 21.
        result = pt.load(self.KEY).set_periodicity('M').rows[0:1:'M']
        self.assertEqual(result.iloc[0, 0], 21.0)

    def test_per_lastnofill(self):
        # LAST_NOFILL: takes the raw value at Jan 30 = NaN (no forward fill).
        result = pt.load(self.KEY).resample_lastnofill.set_periodicity('M').rows[0:1:'M']
        self.assertTrue(np.isnan(result.iloc[0, 0]))

    def test_per_sum(self):
        # Feb values 1..20: sum = 210.
        # (First-period SUM uses from_values[last_B_of_Jan] = NaN, so test Feb.)
        result = pt.load(self.KEY).resample_sum.set_periodicity('M').rows[1:2:'M']
        self.assertEqual(result.iloc[0, 0], 210.0)

    def test_per_avg(self):
        # Feb mean of 1..20 = 10.5.
        result = pt.load(self.KEY).resample_avg.set_periodicity('M').rows[1:2:'M']
        self.assertAlmostEqual(result.iloc[0, 0], 10.5)

    def test_per_first(self):
        # First B day of Feb = 1.
        result = pt.load(self.KEY).resample_first.set_periodicity('M').rows[1:2:'M']
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_per_min(self):
        # Feb min of 1..20 = 1.
        result = pt.load(self.KEY).resample_min.set_periodicity('M').rows[1:2:'M']
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_per_max(self):
        # Feb max of 1..20 = 20. Request both months so the B range covers
        # only the correct data for each month.
        result = pt.load(self.KEY).resample_max.set_periodicity('M').rows[0:2:'M']
        self.assertEqual(result.iloc[1, 0], 20.0)


class TestSaved(DbTest):
    KEY = 'saved_word_test'

    def setUp(self):
        pt.db[self.KEY] = pt.r5.halpha.rows[:5]

    def tearDown(self):
        del pt.db[self.KEY]

    def test_load(self):
        # load() pushes columns from the DB onto the stack
        result = pt.load(self.KEY).rows[:5]
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.columns[0], 'a')
        self.assertEqual(result.iloc[-1, -1], 4.0)


if __name__ == '__main__':
    unittest.main()
