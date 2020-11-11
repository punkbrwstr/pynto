import calendar
import datetime
from collections import namedtuple
from dateutil.relativedelta import relativedelta
import pandas as pd

class Range(object):

    @classmethod
    def from_indexer(cls, indexer):
        if isinstance(indexer, slice):
            if ((indexer.start and isinstance(indexer.start, int)) or
            (indexer.stop and isinstance(indexer.stop, int)) or (indexer.start
            is None and indexer.stop is None and indexer.step is None)):
                return cls(indexer.start, indexer.stop, indexer.step, 'int')
            else:
                return cls.from_dates(indexer.start, indexer.stop, indexer.step)
        else:
            if isinstance(indexer, int):
                return cls(indexer, indexer + 1, 1, 'int')
            else:
                start = get_index('B', indexer)
                return cls(start, start + 1, 'B', 'datetime')

    @classmethod
    def all(cls):
        return cls(None, None, 1, 'int')

    @classmethod
    def from_dates(cls, start=None, stop=None, periodicity='B'):
        if not periodicity:
            periodicity = 'B'
        if start:
            start = get_index(periodicity, start)
        if stop:
            stop = get_index(periodicity, stop)
        return cls(start, stop, periodicity, 'datetime')
    
    @classmethod
    def from_date_range(cls, date_range):
        periodicity = date_range.freq.name
        start = get_index(periodicity, date_range[0])
        stop = get_index(periodicity, date_range[-1]) + 1
        return cls(start, stop, periodicity, 'datetime')

    @classmethod
    def change_periodicity(cls, date_range, periodicity):
        start = get_index(periodicity, date_range.start_date())
        stop = get_index(periodicity, get_date(date_range.step, date_range.stop - 1)) + 1
        return cls(start, stop, periodicity, 'datetime')

    @classmethod
    def from_indicies(cls, start=0, stop=None, step=1):
        range_type = 'int' if isinstance(step, int) else 'datetime'
        return cls(start, stop, step, range_type)

    def __init__(self, start, stop, step, range_type):
        self.start = start
        self.stop = stop
        self.step = step
        self.range_type = range_type

    def __len__(self):
        self._fill_blanks()
        return self.stop - self.start

                
    def __repr__(self):
        if self.range_type == 'datetime':
            r = '['
            r += ':' if self.start is None else f"'{self.start_date().strftime('%Y-%m-%d')}':"
            r += '' if self.stop is None else f"'{self.end_date().strftime('%Y-%m-%d')}':"
            r +=  f"'{self.step}']"
            return r
        else:
            return f'[{str(self.start)}:{str(self.stop)}:{str(self.step)}]'

    def _fill_blanks(self):
        if self.range_type == 'int':
            assert not self.stop is None, 'Range is missing stop'
            if not self.start:
                self.start = 0
            if not self.step:
                self.step = 1

    def expand(self, by):   
        expanded = self.__class__(self.start, self.stop, self.step, self.range_type)
        if by > 0:
            expanded.stop += by
        elif by < 0:
            expanded.start += by
        return expanded

    def offset(self, by):   
        return self.__class__(self.start + by, self.stop + by, self.step, self.range_type)

    def start_date(self):
        assert not isinstance(self.step, int), 'Ordinal range'
        return get_date(self.step,self.start) if not self.start is None else None

    def end_date(self, exclusive=True):
        assert not isinstance(self.step, int), 'Ordinal range'
        if self.stop is None:
            return None
        return get_date(self.step,self.stop - (0 if exclusive else 1))

    def to_index(self):
        self._fill_blanks()
        if self.range_type == 'int':
            return range(self.start, self.stop, self.step)
        else:
            return pd.date_range(get_date(self.step,self.start),
                get_date(self.step,self.stop), freq=self.step)[:-1]

def now():
    d = datetime.datetime.now()
    if d.weekday() < 5 and datetime.datetime.now().hour >= 17:
        return offset('B', d, 1)
    else:
        return d.date()

def parse_date(date):
    if isinstance(date, datetime.datetime):
        return date.date()
    if isinstance(date, pd._libs.tslibs.timestamps.Timestamp):
        return date.date()
    if isinstance(date, datetime.date):
        return date       
    if isinstance(date, str):
        return datetime.datetime.strptime(date, "%Y-%m-%d").date()

def round_b(date):
    date = parse_date(date)
    weekday = date.weekday()
    if weekday < 5:
        return date
    else:
        return date + datetime.timedelta(days=7 - weekday)

def round_w_fri(date):
    date = parse_date(date)
    weekday = date.weekday()
    if weekday == 4:
        return date
    else:
        return date + datetime.timedelta(days=4 - weekday)

def round_w_tue(date):
    date = parse_date(date)
    weekday = date.weekday()
    if weekday == 1:
        return date
    elif weekday == 0:
        return date + datetime.timedelta(days=1)
    else:
        return date + datetime.timedelta(days=8 - weekday)

def round_bm(date):
    date = parse_date(date)
    next_month = date.replace(day=28) + datetime.timedelta(days=4) 
    last_day = next_month - datetime.timedelta(days=next_month.day)
    return last_day - datetime.timedelta(days=max(0,last_day.weekday() - 4))

def round_ba_dec(date):
    date = parse_date(date)
    date = date +  relativedelta(months=12-date.month)
    return round_bm(date)

def round_bq_dec(date):
    date = parse_date(date)
    remainder = date.month % 3
    plusmonths = 0 if remainder == 0 else 2 // remainder
    date = date +  relativedelta(months=plusmonths)
    return round_bm(date)

def add_bm(date, months):
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, calendar.monthrange(year,month)[1])
    day = datetime.date(year, month, day)
    return round_bm(day)

def add_b(date, b):
    daysSinceMonday = date.weekday()
    prevMonday = date - datetime.timedelta(days=daysSinceMonday)
    weekendsInDistance = (b + daysSinceMonday) // 5
    return prevMonday + datetime.timedelta(days=b + weekendsInDistance * 2 + daysSinceMonday)

def count_b(d1, d2):
    daysSinceMonday1 = d1.weekday()
    prevMonday1 = d1 - datetime.timedelta(days=daysSinceMonday1)
    daysSinceMonday2 = d2.weekday()
    prevMonday2 = d2 - datetime.timedelta(days=daysSinceMonday2)
    days = (prevMonday2 - prevMonday1).days
    days -= days * 2 // 7
    return days - daysSinceMonday1 + daysSinceMonday2
    

Periodicity = namedtuple('Periodicity',['epoque','round_function',
                'count_function','offset_function', 'id', 'annualization_factor'])

PERIODICITIES = {
    'B' : Periodicity(
            datetime.date(1970,1,1),
            round_b,
            count_b,
            add_b,
            0,
            250),
    'W-FRI' : Periodicity(
            datetime.date(1970,1,2),
            round_w_fri,
            lambda d1, d2: (d2 - d1).days // 7,
            lambda d, i: d + datetime.timedelta(days=i * 7), 
            1,
            52),
    'W-TUE' : Periodicity(
            datetime.date(1969,12,30),
            round_w_tue,
            lambda d1, d2: (d2 - d1).days // 7,
            lambda d, i: d + datetime.timedelta(days=i * 7), 
            1,
            52),
    'BM' : Periodicity(
            datetime.date(1970,1,30),
            round_bm,
            lambda d2, d1: (d1.year - d2.year) * 12 + d1.month - d2.month,
            lambda d, i: add_bm(d,i), 
            2,
            12),
    'BQ-DEC' : Periodicity(
            datetime.date(1970,3,31),
            round_bq_dec,
            lambda d2, d1: ((d1.year - d2.year) * 12 + d1.month - d2.month) // 3,
            lambda d, i: add_bm(d, i * 3), 
            3,
            4),
    'BA-DEC' : Periodicity(
            datetime.date(1970,12,31),
            round_ba_dec,
            lambda d2, d1: d1.year - d2.year,
            lambda d, i: add_bm(d, i * 12), 
            3,
            1)
}


def get_index(periodicity, date):
    if isinstance(date, int):
        return date
    p = PERIODICITIES[periodicity]
    return p.count_function(p.epoque, p.round_function(date))

def get_date(periodicity, index):
    p = PERIODICITIES[periodicity]
    return p.offset_function(p.epoque, index)

def offset(periodicity, date, offset):
    return get_date(periodicity, get_index(periodicity, date) + offset)

def get_datetimeindex(periodicity, start, end):
    return pd.date_range(get_date(periodicity,start),
                                get_date(periodicity,end),
                                freq=periodicity)[:-1]
def get_id(periodicity):
    return PERIODICITIES[periodicity].id

def get_annualization_factor(periodicity):
    return PERIODICITIES[periodicity].annualization_factor
