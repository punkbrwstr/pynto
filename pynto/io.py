import numpy as np
import pandas as pd
from pynto.main import Column, to_pandas_range
from lima import read_series, read_frame_headers


def lima_series(series_key):
    def lima_series(stack):
        def lima_col(date_range,series_key=series_key):
            return read_series(series_key, date_range.start, date_range.end, date_range.periodicity, as_series=False)[3]
        stack.append(Column(series_key, f'lima series:{series_key}', lima_col))
    return lima_series

def lima_frame(frame_key):
    def lima_frame(stack):
        for header in read_frame_headers(frame_key):
            def lima_col(date_range,series_key=f'{frame_key}.{header}'):
                return read_series(series_key, date_range.start, date_range.end, date_range.periodicity, as_series=False)[3]
            stack.append(Column(header, f'{frame_key}:{header}', lima_col))
    return lima_frame

def pandas_frame(data_frame, header=0):
    def read_frame(stack):
        for header, col in data_frame.iteritems():
            def frame_col(date_range,col=col):
                return col.reindex(to_pandas_range(date_range), method='ffill').values
            stack.append(Column(header,f'read_frame {header}',frame_col))
    return read_frame

def csv(csv_file, header=0):
    frame = pd.read_csv(csv_file, index_col=0, header=header, parse_dates=True)
    return pandas_frame(frame)
