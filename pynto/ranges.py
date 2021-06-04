from __future__ import annotations
import calendar
import datetime
from typing import Union
import pandas as pd
from .dates import *
from . import periodicities

class Range:

    def __init__(self, start: Union[datelike,int],
                        stop: Union[datelike,int], 
                        periodicity: Union[periodicities.Periodicity, str] = None):
        if periodicity is None:
            periodicity = periodicities.B
        elif isinstance(periodicity, str):
            periodicity = getattr(periodicities, periodicity)
        if start is None:
            start = 0
        elif not isinstance(start, int):
            start = periodicity.get_index(start)
        if stop is None:
            stop = now()
        if not isinstance(stop, int):
            stop = periodicity.get_index(stop)
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
        return cls(range_[0], range_[-1], periodicity)

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
            if abs(index) > self.stop - self.start + 1:
                raise IndexError
            return self.periodicity.get_date(index + self.stop + 1)

    def __len__(self):
        return self.stop - self.start

                
    def __repr__(self):
        return f'[{self[0].strftime("%Y-%m-%d")}:{self[-1].strftime("%Y-%m-%d")}:{str(self.periodicity)}]'

    def expand(self, by):   
        expanded = self.__class__(self.start, self.stop, self.step)
        if by > 0:
            expanded.stop += by
        elif by < 0:
            expanded.start += by
        return expanded

    def offset(self, by):   
        return self.__class__(self.start + by, self.stop + by, self.step)

    def to_index(self):
        return pd.date_range(self[0], self[-2], freq=self.periodicity.pandas_offset_code)


