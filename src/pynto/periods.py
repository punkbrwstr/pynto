from __future__ import annotations

import calendar
import datetime
import zoneinfo
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import pandas as pd

datelike = str | datetime.date | datetime.datetime | pd.Timestamp

def parse_date(date: datelike) -> datetime.date:
    if isinstance(date, pd._libs.tslibs.timestamps.Timestamp):
        return date.date() # type: ignore[no-any-return]
    elif isinstance(date, datetime.date):
        return date
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, str):
        return datetime.datetime.strptime(date[:10], "%Y-%m-%d").date()
    else:
        raise TypeError(f'{type(date)} is not datelike')

def _round_b(date: datetime.date) -> datetime.date:
    weekday = date.weekday()
    if weekday < 5:
        return date
    else:
        return date + datetime.timedelta(days=7 - weekday)

def _count_b(d1: datetime.date, d2: datetime.date) -> int:
    daysSinceMonday1 = d1.weekday()
    prevMonday1 = d1 - datetime.timedelta(days=daysSinceMonday1)
    daysSinceMonday2 = d2.weekday()
    prevMonday2 = d2 - datetime.timedelta(days=daysSinceMonday2)
    days = (prevMonday2 - prevMonday1).days
    days -= days * 2 // 7
    return days - daysSinceMonday1 + daysSinceMonday2

def _offset_b(date: datetime.date, b: int) -> datetime.date:
    daysSinceMonday = date.weekday()
    prevMonday = date - datetime.timedelta(days=daysSinceMonday)
    weekendsInDistance = (b + daysSinceMonday) // 5
    return prevMonday + datetime.timedelta(days=b + weekendsInDistance * 2 + daysSinceMonday)


def _count_w(d1: datetime.date, d2: datetime.date) -> int:
    return (d2 - d1).days // 7

def _offset_w(date: datetime.date, by: int) -> datetime.date:
    return date + datetime.timedelta(days=by * 7) 

def _get_round_w(weekday: int) -> Callable[[datetime.date], datetime.date]:
    def _round_w(date: datetime.date) -> datetime.date:
        days_ahead = weekday - date.weekday()
        if days_ahead < 0:  # Target day already happened this week
            days_ahead += 7
        return date + datetime.timedelta(days_ahead)
    return _round_w

def _round_m(date: datetime.date) -> datetime.date:
    next_month = date.replace(day=28) + datetime.timedelta(days=4) 
    last_day = next_month - datetime.timedelta(days=next_month.day)
    return last_day - datetime.timedelta(days=max(0,last_day.weekday() - 4))

def _offset_m(date: datetime.date, months: int) -> datetime.date:
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, calendar.monthrange(year,month)[1])
    date = datetime.date(year, month, day)
    return _round_m(date)

def _count_m(d1: datetime.date, d2: datetime.date) -> int:
    return (d2.year - d1.year) * 12 + d2.month - d1.month

def _round_q(date: datetime.date) -> datetime.date:
    remainder = date.month % 3
    plusmonths = 0 if remainder == 0 else 2 // remainder
    month = date.month + plusmonths
    year = date.year
    if  month > 12:
        month -= 12
        year += 1
    date = datetime.date(year, month, 28)
    return _round_m(date)

def _count_q(d1: datetime.date, d2: datetime.date) -> int:
    return ((d2.year - d1.year) * 12 + d2.month - d1.month) // 3

def _offset_q(d: datetime.date, i: int) -> datetime.date:
    return _offset_m(d, i * 3)


def _round_y(date: datetime.date) -> datetime.date:
    return _round_m(datetime.date(date.year, 12, 31))

def _count_y(d1: datetime.date, d2: datetime.date) -> int:
    return d2.year - d1.year

def _offset_y(date: datetime.date, by: int) -> datetime.date:
    return _offset_m(date, by * 12)

def _str_b(per: Period) -> str:
    return str(per[-1])

def _str_w(per: Period) -> str:
    return f'{per[-1].year}-{per[-1].isocalendar().week}'

def _str_m(per: Period) -> str:
    return str(per[-1])[:7]

def _str_y(per: Period) -> str:
    return str(per[-1])[:4]

def _str_q(per: Period) -> str:
    return f'{per[-1].year}-{(per[-1].month - 1) // 3 + 1}'



@dataclass(repr=False)
class PeriodicityMixin:
    code: str
    epoque: datetime.date
    annualization_factor: int
    offset_code: str
    bbg_code: str | None
    _round: Callable[[datetime.date], datetime.date]
    _count: Callable[[datetime.date, datetime.date], int]
    _offset: Callable[[datetime.date, int], datetime.date]
    _to_str: Callable[[Period], str]


class Periodicity(PeriodicityMixin,Enum):
    B = 'B',datetime.date(1970, 1,1), 260, 'B',           'DAILY', \
        _round_b, _count_b, _offset_b, _str_b
    N = 'N',datetime.date(1969,12,29), 52, 'W-MON',          None, \
        _get_round_w(0), _count_w, _offset_w, _str_w
    T = 'T',datetime.date(1969,12,30), 52, 'W-TUE',          None, \
        _get_round_w(1), _count_w, _offset_w, _str_w
    W = 'W',datetime.date(1969,12,31), 52, 'W-WED',          None, \
        _get_round_w(2), _count_w, _offset_w, _str_w
    H = 'H',datetime.date(1969, 1, 1), 52, 'W-THU',          None, \
        _get_round_w(3), _count_w, _offset_w, _str_w
    F = 'F',datetime.date(1970, 1, 2), 52, 'W-FRI',      'WEEKLY', \
        _get_round_w(4), _count_w, _offset_w, _str_w
    M = 'M',datetime.date(1970, 1,30), 12, 'BME',       'MONTHLY', \
        _round_m, _count_m, _offset_m, _str_m
    Q = 'Q',datetime.date(1970, 3,31),  4, 'BQE-DEC', 'QUARTERLY', \
        _round_q, _count_q, _offset_q, _str_q
    Y = 'Y',datetime.date(1970,12,31),  1, 'BYE-DEC',    'YEARLY', \
        _round_y, _count_y, _offset_y, _str_y

    def __getitem__(self, index: datelike | Period | int | slice) -> Period | Range:
        if isinstance(index, datelike):
            ordinal = self._count(self.epoque, self._round(parse_date(index)))
            return Period(ordinal, self)
        elif isinstance(index, int):
            if index < 0:
                index += self.next().ordinal
            return Period(index, self)
        elif isinstance(index, Period):
            ordinal = self._count(self.epoque, self._round(index[-1]))
            return Period(ordinal, self)
        elif isinstance(index, slice):
            if isinstance(index.stop, datelike):
                date = parse_date(index.stop)
                stop = self._count(self.epoque, self._round(date))
            elif isinstance(index.stop, int):
                stop = index.stop
                if stop < 0:
                    stop += self.next().ordinal
            elif isinstance(index.stop, Period):
                stop = self._count(self.epoque, self._round(index.stop[-1]))
            elif index.stop is None:
                stop = self.next().ordinal
            else: 
                raise TypeError(f'Unsupported indexer')
            if isinstance(index.start, datelike):
                date = parse_date(index.start)
                start = self._count(self.epoque, self._round(date))
            elif isinstance(index.start, int):
                start = index.start
                if start < 0:
                    start += self.next().ordinal
            elif isinstance(index.start, Period):
                start = self._count(self.epoque, self._round(index.start[-1]))
            elif index.start is None:
                start = 0
            else: 
                raise TypeError(f'Unsupported indexer')
            return Range(start, stop, self)
        else:
            raise TypeError(f'Unsupported indexer')

    def next(self) -> Period:
        ny = datetime.datetime.now(datetime.UTC).astimezone(zoneinfo.ZoneInfo(key='US/Eastern'))
        ny_date = ny.date()
        if ny.hour >= 17:
            ny_date += datetime.timedelta(days=1)
        return self[ny_date]

    def current(self) -> Period:
        return self.next() - 1

    def __hash__(self):
        return hash(self.code)

    def __repr__(self):
        return f'Periodicity.{self.code}' 

    @classmethod
    def from_offset_code(cls, code: str) -> Periodicity:
        for p in cls:
            if p.offset_code == code:
                return p
        raise ValueError('Incompatible index freq: ' + code)

    @classmethod
    def from_ordinal(cls, ordinal: int) -> Periodicity:
        for p in cls:
            if p.ordinal == ordinal:
                return p
        raise ValueError()

    def __str__(self):
        return self.name

@dataclass
class Period:
    ordinal: int
    periodicity: Periodicity

    def __len__(self) -> int:
        return (self.periodicity._offset(self.periodicity.epoque, self.ordinal) \
                   - self.periodicity._offset(self.periodicity.epoque, self.ordinal - 1 )).days

    def __getitem__(self, index: int) -> datetime.date:
        if index >= 0:
            first = self.periodicity._offset(self.periodicity.epoque, self.ordinal - 1 ) \
                + datetime.timedelta(days=1)
            d = first + datetime.timedelta(days=index)
            if index == 0:
                return d
            last = self.periodicity._offset(self.periodicity.epoque, self.ordinal)
        else:
            last = self.periodicity._offset(self.periodicity.epoque, self.ordinal)
            d = last + datetime.timedelta(days=index + 1)
            if index == -1:
                return d
            first = self.periodicity._offset(self.periodicity.epoque, self.ordinal - 1 ) \
                + datetime.timedelta(days=1)
        if d < first or d > last:
            raise IndexError(f'Period index {index} out of range for {self}')
        return d

    def __add__(self, by: int) -> Period:
        return Period(self.ordinal + by, self.periodicity)

    def __sub__(self, by: int) -> Period:
        return Period(self.ordinal - by, self.periodicity)

    def expand(self, by: int = 1) -> Range:
        if by > 0:
            start, stop = self.ordinal, self.ordinal + by
        elif by == 0:
            start, stop = self.ordinal - 1, self.ordinal
        else:
            start, stop = self.ordinal + by, self.ordinal + 1
        return Range(start, stop, self.periodicity)


    def __str__(self) -> str:
        return f'{self.periodicity}{self.periodicity._to_str(self)}'

    def __repr__(self) -> str:
        return self.__str__()

@dataclass
class Range:
    start: int #inclusive
    stop: int #exclusive
    periodicity: Periodicity

    def __post_init__(self):
        assert self.start <= self.stop, \
            f'Range stop cannot be less than start ({self.start}-{self.stop})'

    @classmethod
    def from_index(cls, date_range: pd.DatetimeIndex) -> Range:
        assert date_range.freq is not None, 'Index must have freq.' 
        return  Periodicity.from_offset_code(date_range.freq.name) \
            [date_range[0]:date_range[-1]].expand(1) # type: ignore

    def change_periodicity(self, periodicity: Periodicity | str):
        if isinstance(periodicity, str):
            periodicity = Periodicity[periodicity.upper()]
        return periodicity[self[0][0]:self[-1][-1]].expand() # type: ignore

    def resample_indicies(self, range_: Range, round_: bool = True) -> tuple[int,list[int]]:
        my_ordinals, target_ordinals = [], []
        for i, target_per in enumerate(range_):
            my_per = self.periodicity[target_per]
            if my_per.ordinal >= self.start and my_per.ordinal < self.stop \
                and (round_ or my_per[-1] == target_per[-1]) :
                my_ordinals.append(my_per.ordinal - self.start)
                target_ordinals.append(i)
        return target_ordinals, my_ordinals

    def __iter__(self) -> Iterator[Period]:
        i = 0
        while i + self.start < self.stop:
            yield self[i]
            i += 1

    def __getitem__(self, index: int) -> Period:
        if index >= 0:
            if index >= self.stop - self.start:
                raise IndexError
            per = Period(index + self.start, self.periodicity)
        else:
            if abs(index) > self.stop - self.start:
                raise IndexError
            per = Period(index + self.stop, self.periodicity)
        return per

    def __len__(self) -> int:
        assert self.stop >= self.start, f'Negative range length {self.start}-{self.stop}'
        return self.stop - self.start

    def __add__(self, by: int) -> Period:
        return Range(self.start + by, self.stop + by, self.periodicity)

    def __sub__(self, by: int) -> Period:
        return Range(self.start - by, self.stop - by, self.periodicity)
                
    def __repr__(self) -> str:
        if len(self) == 0:
            return '[]'
        return f'{str(self.periodicity)}[{self[0][-1]}:{(self[-1] + 1)[-1]}]'

    def __hash__(self) -> int:
        return hash((self.start,self.stop,self.periodicity))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self.start == other.start and self.stop == other.stop and self.periodicity == other.periodicity

    def expand(self, by: int = 1) -> Range:   
        expanded = Range(self.start, self.stop,self.periodicity)
        if by > 0:
            expanded.stop += by
        elif by < 0:
            expanded.start += by
        return expanded

    def offset(self, by) -> Range:   
        return Range(self.start + by, self.stop + by,self.periodicity)

    def to_index(self) -> pd.DatetimeIndex:
        return pd.date_range(self[0][-1], self[-1][-1],
                    freq=self.periodicity.offset_code)

    def day_counts(self) -> list[int]:
        dates = [p[-1] for p in self.expand(-1)]
        return [(d1 - d0).days for d1,d0 in zip(dates[1:], dates[:-1])]



