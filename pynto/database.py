from __future__ import annotations
import os
import redis
import struct
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import namedtuple
from typing import Union, Dict
from redis.connection import UnixDomainSocketConnection
from .dates import datelike
from .ranges import Range
from . import periodicities

def get_client() -> Db:
    global _CLIENT
    if not '_CLIENT' in globals():
        args = {}
        if 'PYNTO_REDIS_PASSWORD' in os.environ:
            args['password'] = os.environ['PYNTO_REDIS_PASSWORD']
        if 'PYNTO_REDIS_PATH' in os.environ:
            args['path'] = os.environ['PYNTO_REDIS_PATH']
        else:
            if 'PYNTO_REDIS_HOST' in os.environ:
                args['host'] = os.environ['PYNTO_REDIS_HOST']
            if 'PYNTO_REDIS_PORT' in os.environ:
                args['port'] = os.environ['PYNTO_REDIS_PORT']
        _CLIENT = Db(**args)
    return _CLIENT

Type = namedtuple('Type', ['code', 'pad_value', 'length'])

TYPES: Dict[str, Type]  = {
    '<f8': Type('<f8',np.nan,8),
    '|b1': Type('|b1',False,1),
    '<i8': Type('<i8',0,8)
}

METADATA_FORMAT = '<6s6sll'
METADATA_SIZE = struct.calcsize(METADATA_FORMAT)

@dataclass
class Metadata:
    dtype: str
    periodicity_code: str
    start: int
    stop: int

    @property
    def is_frame(self):
        return self.dtype.startswith('<U')

    @property
    def periodicity(self):
        return periodicities.from_pandas(self.periodicity_code)

class Db:

    def __init__(self, **kwargs):
        if 'path' in kwargs:
           kwargs['connection_class'] = UnixDomainSocketConnection
        self._pool = redis.ConnectionPool(**kwargs)

    @property
    def connection(self):
        return redis.Redis(connection_pool=self._pool)

    def __setitem__(self, key: str, pandas: Union[pd.Series, pd.DataFrame]):
        if not isinstance(pandas.index, pd.MultiIndex) and pandas.index.freq is None:
            raise Exception('Missing index freq.')
        series_to_save = []
        if isinstance(pandas, pd.Series):
            series_to_save.append((key, pandas))
        else:
            self._check_dups(pandas.columns)

            if isinstance(pandas.index, pd.MultiIndex):
                assert len(pandas.index.levels) == 2
                rows = pandas.loc[pandas.index[0][0]].index
                self._check_dups(rows, 'row')
                rows = np.char.add('$',rows.values.astype('U'))
                columns = np.char.add('$',pandas.columns.values.astype('U'))
                columns = np.char.add(np.repeat(rows,len(columns)), np.tile(columns,len(rows)))
                index=pandas.index.levels[0]
                pandas = pd.DataFrame(pandas.values.reshape(len(index), len(columns)),
                            index=index, columns=columns) 
            periodicity = periodicities.from_pandas(pandas.index.freq.name)
            end = periodicity.get_index(pandas.index[-1].date()) + 1
            try:
                md = self._read_metadata(key)
                if md.periodicity != periodicity:
                    raise Exception('Incompatible periodicity.')
                columns = set(self._get_data_range(key, 0, -1).decode().split('\t'))
                if end > md.stop:
                    self._update_end(key, end)
                first_save = False
            except KeyError:
                start = periodicity.get_index(pandas.index[0].date())
                md = Metadata('<U', pandas.index.freq.name, start, end)
                columns = set()
                first_save = True
            new_columns = []
            for column,series in pandas.iteritems():
                series_code = f'{key}:{column}'
                if not column in columns:
                    columns.add(column)
                    new_columns.append(column)
                series = series[series.first_valid_index():series.last_valid_index()]
                series_to_save.append((series_code, series))
            if first_save:
                self._write(key, md, '\t'.join(new_columns).encode()) 
            elif len(new_columns) > 0:
                self.connection.append(key, ('\t' + '\t'.join(new_columns)).encode()) 
        for key, series in series_to_save:
            periodicity = periodicities.from_pandas(series.index.freq.name)
            start = periodicity.get_index(series.index[0].date())
            series_md = Metadata(series.dtype.str, series.index.freq.name,
                                    start, start + len(series.index)) 
            try:
                saved_md = self._read_metadata(key)
                if saved_md.periodicity != periodicity:
                    raise Exception(f'Incompatible periodicity.')   
            except:
                saved_md = None
            data = series.values
            if saved_md is None or series_md.start < saved_md.start:
                self._write(key, series_md, data.tobytes())
                continue
            if series_md.start > saved_md.stop:
                pad = np.full(series_md.start - saved_md.stop, TYPES[saved_md.dtype].pad_value)
                data = np.hstack([pad, data])
                start = saved_md.stop 
            else:
                start = series_md.start
            start_offset = (start - saved_md.start) * np.dtype(saved_md.dtype).itemsize
            self._set_data_range(key, start_offset, data.tobytes())
            if series_md.stop > saved_md.stop:
                self._update_end(key, series_md.stop)

    def __delitem__(self, key: str):
        try:
            md = self._read_metadata(key)
        except KeyError:
            return
        if md.is_frame:
            for series_key in self.read_frame_series_keys(key):
                self.connection.delete(series_key)
        self.connection.delete(key)

    def __getitem__(self, key: str):
        return self.read(key)

    def read(self, key: str,
                start: Union[int,datelike] = None,
                stop: Union[int,datelike] = None,
                periodicity: Union[str, periodicities.Periodicity] = None,
                resample_method: str = 'last') -> Union[pd.Series, pd.DataFrame]:
        md = self._read_metadata(key)
        if isinstance(periodicity, str):
            periodicity = getattr(periodicities, periodicity)
        if not md.is_frame:
            data = self.read_series_data(key, start, stop, periodicity, md, resample_method)
            return pd.Series(data[3], index=Range(*data[:3]).to_index(), name=key)
        else:
            data = self._read_frame_data(key, start, stop, periodicity, md, resample_method)
            index = Range(*data[:3]).to_index()
            c = np.array(data[3])
            array = data[4]
            if c[0].startswith('$') and c[0].count('$') == 2:
                columns, rows = {}, {}
                for a,b in np.char.split(np.char.lstrip(c,'$'),'$'):
                    rows[a] = None
                    columns[b] = None
                index = pd.MultiIndex.from_product([index, rows])
                array = array.reshape(array.shape[0]*len(rows),len(columns))
            else:
                columns = c
            return pd.DataFrame(array, columns=columns, index=index)

    def read_frame_diagonal(self, key, start=None, end=None, periodicity=None, resample_method='last'):
        md = self._read_metadata(key)
        start = md.start if start is None else get_index(md.periodicity, start)
        end = md.stop if end is None else get_index(md.periodicity, end)
        periodicity = md.periodicity if periodicity is None else periodicity
        index = Range(start, end, periodicity).to_index()
        c = np.array(self.read_frame_headers(key))
        if not (c[0].startswith('$') and c[0].count('$') == 2):
            raise ValueError(f'Frame {key} is not two-dimensional')
        columns = []
        for a,b in np.char.split(np.char.lstrip(c,'$'),'$'):
            if a == b:
                columns.append(a)
        array = np.column_stack([self.read_series_data(f'{key}:${c}${c}', start, end, periodicity,
                    resample_method)[3] for c in columns])
        return pd.DataFrame(array, columns=columns, index=index)

    def read_range(self, key: str) -> Range:
        md = self._read_metadata(key)
        return Range(md.start, md.stop, md.periodicity)

    def read_series_data(self, key: str,
                            start: Union[int,datelike] = None,
                            stop: Union[int,datelike] = None,
                            periodicity: periodicities.Periodicity = None,
                            md: Metadata = None,
                            resample_method: str = 'last'):
        if md is None:
            md = self._read_metadata(key)
        needs_resample = periodicity is not None and periodicity != md.periodicity
        if needs_resample:
            if not start is None:
                start_index = md.periodicity.get_index(
                        periodicity.get_date(periodicity.get_index(start)))
            else:
                start_index = md.start
            if not stop is None:
                stop_index = md.periodicity.get_index(
                        periodicity.get_date(periodicity.get_index(stop))) 
            else:
                stop_index = md.stop
        else:
            if start is None:
                start_index = md.start
            else:
                start_index = md.periodicity.get_index(start)
            if stop is None:
                stop_index = md.stop
            else:
                stop_index = md.periodicity.get_index(stop)
        periodicity = periodicity if periodicity else md.periodicity
        if start_index < md.stop and stop_index >= md.start:
            itemsize = np.dtype(md.dtype).itemsize 
            selected_start = max(0, start_index - md.start)
            selected_end = min(stop_index, md.stop + 2) - md.start
            buff = self._get_data_range(key, selected_start * itemsize, selected_end * itemsize)
            data = np.frombuffer(buff, md.dtype)
            if len(data) != stop_index - start_index:
                output_start = max(0, md.start - start_index)
                output = np.full(stop_index - start_index,TYPES[md.dtype].pad_value)
                output[output_start:output_start+len(data)] = data
            else:
                output = data
        else:
            output = np.full(stop_index - start_index,TYPES[md.dtype].pad_value)
        if needs_resample:
            s = pd.Series(output, index=Range(
                            start_index,stop_index,md.periodicity).to_index(), name=key)
            s = getattr(s.resample(periodicity.pandas_offset_code),
                            resample_method)().reindex(Range(start, stop, periodicity).to_index())
            return (periodicity.get_index(s.index[0].date()),
                    periodicity.get_index(s.index[-1].date()),
                    periodicity, s.values)
        else:
            return (start_index, stop_index, md.periodicity, output)
        
    def read_frame_headers(self, key):
        return self._get_data_range(key, 0, -1).decode().split('\t')

    def read_frame_series_keys(self, key):
        return [f'{key}:{c}' for c in self.read_frame_headers(key)]

    def _read_metadata(self, key: str) -> Metadata:
        data = self.connection.getrange(key,0,METADATA_SIZE-1)
        if len(data) == 0:
            raise KeyError(f'No metadata for key {key}')
        s = struct.unpack(METADATA_FORMAT, data)
        return Metadata(s[0].decode().strip(), s[1].decode().strip(), s[2], s[3])

    def _read_frame_data(self, key: str,
                            start: Union[int,datelike] = None,
                            stop: Union[int,datelike] = None,
                            periodicity: periodicities.Periodicity = None,
                            md: Metadata = None,
                            resample_method: str = 'last'):
        if md is None:
            md = self._read_metadata(key)
        start = md.start if start is None else md.periodicity.get_index(start)
        stop = md.stop if stop is None else md.periodicity.get_index(stop)
        periodicity = md.periodicity if periodicity is None else periodicity
        columns = self.read_frame_headers(key)
        data = np.column_stack([self.read_series_data(f'{key}:{c}', start, stop, periodicity,
                    None, resample_method)[3] for c in columns])
        return (start, stop, periodicity, columns, data)


    def _update_end(self, key: str, stop: int):
        self.connection.setrange(key, METADATA_SIZE - struct.calcsize('<l'), struct.pack('<l',int(stop)))

    def _write(self, key: str, metadata: Metadata, data: np.ndarray):
        packed_md = struct.pack(METADATA_FORMAT,
                                '{0: <6}'.format(metadata.dtype).encode(),
                                '{0: <6}'.format(metadata.periodicity.pandas_offset_code).encode(),
                                metadata.start,
                                metadata.stop) 
        self.connection.set(key, packed_md + data)

    def _get_data_range(self, key: str, start: int, stop: int):
        stop = -1 if stop == -1 else METADATA_SIZE + stop - 1
        return self.connection.getrange(key, str(METADATA_SIZE + start), str(stop))
        
    def _set_data_range(self, key: str, start: int, data: np.ndarray):
        self.connection.setrange(key, str(METADATA_SIZE + start), data)

    def _check_dups(self, index: pd.Index, type_='column'):
        unq, unq_cnt = np.unique(index, return_counts=True)
        if len(unq) != len(index):
            dups = unq[unq_cnt > 1].tolist()
            raise ValueError(f'Duplicate {type_} name{"s: " + str(dups)  if len(dups) > 1 else ": " + str(dups[0])}')


