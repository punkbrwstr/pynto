from __future__ import annotations
import calendar
import datetime
from typing import Union
import pandas as pd
from .dates import *
from . import periodicities

class Range:

    def __init__(self, start: Union[datelike,int],
                        stop: Union[datelike,int] = None, 
                        periodicity: Union[periodicities.Periodicity, str] = None):
        if periodicity is None:
            periodicity = periodicities.B
        elif isinstance(periodicity, str):
            periodicity = getattr(periodicities, periodicity)
        if stop is None:
            stop = periodicity.get_index(periodicities.now(periodicity))
            if isinstance(start, int):
                start = stop - abs(start)
        elif not isinstance(stop, int):
            stop = periodicity.get_index(stop)
        if not isinstance(start, int):
            start = periodicity.get_index(start)
        self.start = start
        self.stop = stop
        self.periodicity = periodicity

    @classmethod
    def from_indexer(cls, indexer: Union[slice,int,datelike]):
        if isinstance(indexer, slice):
            return cls(indexer.start, indexer.stop, indexer.step)
        else:
            if not isinstance(indexer, int):
                indexer = periodicities.B.get_index(indexer)
            return cls(indexer, indexer + 1)

    @classmethod
    def from_index(cls, date_range: pd.DatetimeIndex):
        range_ =  cls(date_range[0], date_range[-1],
                    periodicities.from_pandas(date_range.freq.name))
        range_.stop += 1
        return range_

    @classmethod
    def change_periodicity(cls, range_: Range,
                            periodicity: Union[str,periodicities.Periodicity]):
        if isinstance(periodicity, str):
            periodicity = getattr(periodicities, periodicity)
        return cls(range_[0], periodicity.offset(range_[-1], 1), periodicity)

    def __iter__(self):
        i = 0
        while i + self.start < self.stop:
            yield self[i]
            i += 1

    def __getitem__(self, index):
        if index >= 0:
            if index >= self.stop - self.start:
                raise IndexError
            return self.periodicity.get_date(index + self.start)
        else:
            if abs(index) > self.stop - self.start:
                raise IndexError
            return self.periodicity.get_date(index + self.stop)

    def __len__(self):
        assert self.stop >= self.start, f'Negative range length {self.start}-{self.stop}'
        return self.stop - self.start
                
    def __repr__(self):
        if len(self) == 0:
            return '[]'
        end_exclusive = self.periodicity.get_date(self.stop)
        return f'[{self[0].strftime("%Y-%m-%d")}:{end_exclusive.strftime("%Y-%m-%d")}:{str(self.periodicity)}]'

    def __hash__(self):
        return hash((self.start,self.stop,self.periodicity))

    def __eq__(self, other):
        return self.start == other.start and self.stop == other.stop and self.periodicity == other.periodicity

    def expand(self, by):   
        expanded = self.__class__(self.start, self.stop, self.periodicity)
        if by > 0:
            expanded.stop += by
        elif by < 0:
            expanded.start += by
        return expanded

    def offset(self, by):   
        return self.__class__(self.start + by, self.stop + by, self.periodicity)

    def to_index(self):
        return pd.date_range(self[0], self[-1], freq=self.periodicity.pandas_offset_code)

    def day_counts(self):
        return [(self.periodicity.get_date(i) - self.periodicity.get_date(i-1)).days \
                    for i in range(self.start, self.stop)]
