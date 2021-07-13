import datetime
from dataclasses import dataclass
from typing import Callable
from .dates import *

@dataclass
class Periodicity:
    code: str
    epoque: datetime.date
    id_: int
    annualization_factor: int
    pandas_offset_code: str
    round_function: Callable[[datelike], datetime.date]
    count_function: Callable[[datetime.date, datetime.date], int]
    offset_function: Callable[[datetime.date, int], datetime.date]

    def get_index(self, date: Union[int,datetime.date]):
        if isinstance(date, int):
            return date
        return self.count_function(self.epoque, self.round_function(date))

    def get_date(self, index):
        return self.offset_function(self.epoque, index)

    def offset(self, date, offset):
        return self.get_date(self.get_index(date) + offset)

    def __str__(self):
        return self.code

    def __repr__(self):
        return self.code



B =  Periodicity('B', datetime.date(1970,1,1), 0, 260, 'B', round_b, count_b, add_b)
W_F = Periodicity('W_F', datetime.date(1970,1,2), 1, 52, 'W-FRI',
        round_w_fri, count_w, offset_w) 
W_T = Periodicity('W_T', datetime.date(1969,12,30), 1, 52, 'W-TUE',
        round_w_tue, count_w, offset_w)
M = Periodicity('M', datetime.date(1970,1,30), 2, 12, 'BM',
        round_bm, count_m, add_bm)
Q = Periodicity('Q', datetime.date(1970,3,31), 3, 4, 'BQ-DEC',
        round_bq_dec, count_q, offset_q)
Y = Periodicity('Y', datetime.date(1970,12,31), 3, 1, 'BA-DEC',
        round_ba_dec, count_y, offset_y)

_pandas_map = {'B': B, 'W-TUE': W_T, 'W-FRI': W_F, 'BM': M, 'BQ-DEC': Q, 'BA-DEC': Y}
from_pandas = _pandas_map.get
