from __future__ import annotations
import os
import redis
import struct
import warnings
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from collections import namedtuple
from typing import Union, Dict, NamedTuple, Iterable, Any
from redis.connection import UnixDomainSocketConnection
from .main import Word, Column
from .periods import Periodicity, Period, Range, datelike

DELIMITER = '/'
METADATA_PREFIX = '/p2m/'
DATA_PREFIX = '/p2d/'
INDEX_PREFIX = '/p2i'

_CLIENT: Db | None = None

def get_client() -> Db:
    global _CLIENT
    #if not '_CLIENT' in globals():
    if _CLIENT is None:
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

@dataclass
class DataTypeMixin:
    dtype: str | None
    pad_value: Any
    length: int

class DataType(DataTypeMixin, Enum):
    F =  '<f8', np.nan, 8
    I = '<i8', 0, 8
    B = '|b1', False, 1
    A = None, None, 0

    @classmethod
    def from_dtype(cls, code: str) -> DataType:
        for p in cls:
            if p.dtype == code:
                return p
        raise ValueError()




METADATA_FORMAT = '<cllB'
METADATA_SIZE = struct.calcsize(METADATA_FORMAT)

@dataclass
class Metadata:
    type_: DataType
    range_: Range
    columns: list[str]
    column_metadata: list[Metadata] = field(default_factory=list)

    def pack(self) -> bytes:
        return struct.pack(METADATA_FORMAT, self.type_.name.encode(), 
                    self.range_.start, self.range_.stop,
                    self.range_.periodicity.ordinal) \
                    + '\t'.join(self.columns).encode()

    def save(self, connection: redis.Redis,  key: str) -> None:
        p = connection.pipeline()
        p.set(f'{METADATA_PREFIX}{key}', self.pack())
        for column, md in zip(column, column_metadata):
            p.set(f'{METADATA_PREFIX}{key}:{column}', md.pack())
        p.execute()            

    @classmethod
    def unpack(cls, bytes_) -> Metadata:
        t, start, stop, p = struct.unpack(METADATA_FORMAT, bytes_[:METADATA_SIZE)])
        if len(bytes_) > METADATA_SIZE:
            cols = bytes_[METADATA_SIZE:].decode().split('\t')
        else:
            cols = [] 
        return cls(DataType[t.decode()], Range(start, stop, Periodicity.from_ordinal(p)))
    @classmethod
    def load(cls, connection: redis.Redis,  key: str) -> Metadata:
        md = Metadata.unpack(connection.get(f'{METADATA_PREFIX}{key}')
        p.set(f'{METADATA_PREFIX}{key}', self.pack())

class Db:

    def __init__(self, **kwargs):
        if 'path' in kwargs:
           kwargs['connection_class'] = UnixDomainSocketConnection
        self._pool = redis.ConnectionPool(**kwargs)

    @property
    def connection(self):
        return redis.Redis(connection_pool=self._pool)
    
    def __call__(self, key):
        return Saved()(key)

    def get_metadata(self, keys: str | list[str]) -> list[Metadata]:
        keys = keys if isinstance(keys, list) else [keys] 
        p = self.connection.pipeline()
        for key in keys:
            p.getrange(f'{METADATA_PREFIX}{key}', 0, METADATA_SIZE-1)
        mds = []
        for data in p.execute():
            if len(data) == 0:
                mds.append(None)
            else:
                mds.append(Metadata.unpack(data))
        return mds

    def __setitem__(self, key: str, pandas: pd.Series | pd.DataFrame):
        if isinstance(pandas, pd.Series):
            series_to_save = [(key, pandas)]
        else:
            self._check_dups(pandas.columns)
            if isinstance(pandas.index, pd.MultiIndex):
                assert len(pandas.index.levels) == 2
                rows = pandas.loc[pandas.index[0][0]].index
                self._check_dups(rows, 'row')
                rows = np.char.add('$',rows.values.astype('U'))
                columns = np.char.add('$',pandas.columns.values.astype('U'))
                columns = np.char.add(np.repeat(rows,len(columns)), np.tile(columns,len(rows)))
                index=pandas.index.remove_unused_levels().levels[0]
                pandas = pd.DataFrame(pandas.values.reshape(len(index), len(columns)),
                            index=index, columns=columns) 
            frame_md = self.get_metadata(key)[0]
            range_ = Range.from_index(pandas.index)
            if frame_md:
                assert frame_md.periodicity != range_.periodicity, 'Incompatible periodicity.'
                existing_columns = set(frame_md.columns)
                new_columns = set(pandas.columns).difference(existing_columns)
                if len(new_columns) > 0:
                    self.connection.append(f'{METADATA_PREFIX}{key}',
                            ('\t' + '\t'.join(new_columns)).encode()) 
            else:
                self.connection.set(f'{METADATA_PREFIX}{key}',
                    Metadata(DataType.A, range_, pandas.columns).pack())
            series_to_save = [(f'{key}:{c}',s) for c,s in pandas.items()]
        metadatas = self.get_metadata([s[0] for s in series_to_save])
        for saved_md, (k, series) in zip(metadatas, series_to_save):
            # don't save leading or trailing nans
            if series.values.dtype.kind == 'f':
                nz = (~np.isnan(series.to_numpy())).nonzero()[0]
                if len(nz) == 0: # just save one nan if all nans
                    series = series.iloc[0:1]
                else:
                    series = series.iloc[nz.min():nz.max()+1]
            if isinstance(series.dtype,  pd.api.extensions.ExtensionDtype):
                raise Exception('Unsupported type.')
            else:
                type_str = series.dtype.str
            range_ = Range.from_index(series.index)
            series_md = Metadata(DataType.from_dtype(type_str), range_, [])
            assert not saved_md or saved_md.range_.periodicity == series.periodicity
            data = series.to_numpy()
            if saved_md is None or series_md.range_.start < saved_md.range_.start:
                self.connection.set(METADATA_PREFIX + k, series_md.pack())
                self.connection.set(DATA_PREFIX + k, data.tobytes())
            else:
                if series_md.range_.start > saved_md.range_.stop:
                    pad = np.full(series_md.start - saved_md.stop, saved_md.datatype.pad_value)
                    data = np.hstack([pad, data])
                    #STARTHERE
                    start = saved_md.stop 
                else:
                    start = series_md.start
                start_offset = (start - saved_md.start) * saved_md.datatype.length
                self._set_data_range(k, saved_md.datatype.length, start_offset, data)
                if series_md.stop > saved_md.stop:
                    self._update_end(k, series_md.stop)
        self.add_to_index(key)

    def __delitem__(self, key: str):
        md = self._read_metadata(key)
        if md:
            if md.is_frame:
                for series_key in self.read_frame_series_keys(key):
                    self.connection.delete(f'{PREFIX}{DELIMITER}{series_key}')
            self.connection.delete(f'{PREFIX}{DELIMITER}{key}')
        self.remove_from_index(key)

    def __getitem__(self, key: str):
        md = self._read_metadata(key)
        if not md.is_frame:
            data = self.read_series_data(key, start, stop, p, md, resample_method)
            return pd.Series(data[3], index=Range(*data[:3]).to_index(), name=key)
        else:
            data = self._read_frame_data(key, start, stop, p, md, resample_method)
            index = Range(*data[:3]).to_index()
            c = np.array(data[3])
            array = data[4]
            if c[0].startswith('$') and c[0].count('$') == 2:
                columns: dict[str,None] = {} #dict not set preserves order
                rows: dict[str,None] = {}
                for a,b in np.char.split(np.char.lstrip(c,'$'),'$'):
                    rows[a] = None
                    columns[b] = None
                index = pd.MultiIndex.from_product([index, list(rows.keys())])
                array = array.reshape(array.shape[0]*len(rows),len(columns))
                c = np.array(columns.keys())
            return pd.DataFrame(array, columns=c, index=index)

    def read_frame_diagonal(self, key, start=None, stop=None, periodicity=None, resample_method='last'):
        md = self._read_metadata(key)
        if not md:
            raise KeyError(f'PyntoDB key {key} not found')
        start = md.start if start is None else md.periodicity.get_index(start)
        stop = md.stop if stop is None else md.periodicity.get_index(stop)
        periodicity = md.periodicity if periodicity is None else periodicity
        index = Range(start, stop, periodicity).to_index()
        c = np.array(self.read_frame_headers(key))
        if not (c[0].startswith('$') and c[0].count('$') == 2):
            raise ValueError(f'Frame {key} is not two-dimensional')
        columns = []
        for a,b in np.char.split(np.char.lstrip(c,'$'),'$'):
            if a == b:
                columns.append(a)
        array = np.column_stack([self.read_series_data(f'{key}:${c}${c}',
            start, stop, periodicity, resample_method=resample_method)[3] for c in columns])
        return pd.DataFrame(array, columns=columns, index=index)

    def read_range(self, key: str) -> Range:
        md = self._read_metadata(key)
        if not md:
            return None
        elif md.is_frame:
            return Range(*self._read_frame_range(key, md=md)[:3])
        return Range(md.start, md.stop, md.periodicity)

    def read_series_data(self, key: str,
                            start: int | datelike | None = None,
                            stop: int | datelike | None = None,
                            periodicity: Periodicity | None = None,
                            md: Metadata | None = None,
                            resample_method: str = 'last'):
        if md is None:
            md = self._read_metadata(key)
            if not md:
                raise KeyError(f'PyntoDB key {key} not found')
        if periodicity is not None and periodicity != md.periodicity:
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
            if start:
                start_index = md.periodicity.get_index(start)
            else:
                start_index = md.start
            if stop:
                stop_index = md.periodicity.get_index(stop)
            else:
                stop_index = md.stop
        if start_index < md.stop and stop_index >= md.start:
            selected_start = max(0, start_index - md.start)
            selected_end = min(stop_index, md.stop + 2) - md.start
            buff = self._get_data_range(key, md.datatype.length, selected_start * md.datatype.length,
                                        selected_end * md.datatype.length)
            data = np.frombuffer(buff, md.datatype.dtype_str)
            if len(data) != stop_index - start_index:
                output_start = max(0, md.start - start_index)
                output = np.full(stop_index - start_index,md.datatype.pad_value)
                output[output_start:output_start+len(data)] = data
            else:
                output = data
        else:
            output = np.full(stop_index - start_index, md.datatype.pad_value)
        if periodicity is not None and periodicity != md.periodicity:
            s = pd.Series(output, index=Range(
                            start_index,stop_index,md.periodicity).to_index(), name=key)
            s = getattr(s.resample(periodicity.pandas_offset_code), resample_method)() \
                            .reindex(Range(start, stop, periodicity).to_index())
            return (periodicity.get_index(s.index[0].date()),
                    periodicity.get_index(s.index[-1].date()),
                    periodicity, s.values)
        else:
            return (start_index, stop_index, md.periodicity, output)
        
    def read_frame_headers(self, key):
        return self._get_data_range(key).decode().split('\t')

    def read_frame_series_keys(self, key):
        return [f'{key}:{c}' for c in self.read_frame_headers(key)]

    
    def _read_frame_range(self, key: str,
                            start: int | datelike | None = None,
                            stop: int | datelike | None= None,
                            periodicity: Periodicity | None = None,
                            md: Metadata | None = None):
        if md is None:
            md = self._read_metadata(key)
        columns = self.read_frame_headers(key)
        keys = [f'{key}:{c}' for c in columns]
        mds = self._read_metadatas(keys)
        if periodicity is None:
            periodicity = md.periodicity
        if start is None:
            start = min([md.start for md in mds])
        if stop is None:
            stop = max([md.stop for md in mds])
        return (start, stop, periodicity, keys, mds, columns)

    def _read_frame_data(self, key: str,
                            start: int | datelike | None = None,
                            stop: int | datelike | None = None,
                            periodicity: Periodicity | None = None,
                            md: Metadata | None = None,
                            resample_method: str = 'last'):
        start, stop, periodicity, keys, mds, columns = self._read_frame_range(key,
                                                            start, stop, periodicity, md)
        data = []
        for md, key in zip(mds,keys):
            data.append(self.read_series_data(key, start, stop, periodicity, md, resample_method)[3])
        return (start, stop, periodicity, columns, np.column_stack(data))


    def _update_end(self, key: str, stop: int):
        self.connection.setrange(f'{PREFIX}{DELIMITER}{key}', METADATA_SIZE - struct.calcsize('<l'), struct.pack('<l',int(stop)))

    def _write(self, key: str, metadata: Metadata, data: np.ndarray):
        self.connection.set(f'{PREFIX}{DELIMITER}{key}', metadata.pack() +
                            bytes(metadata.datatype.length) + data.tobytes())

    def _get_data_range(self, key: str, data_length: int = 0,
                        start: int = 0, stop: int = -1):
        skip = METADATA_SIZE + data_length
        if stop != -1:
            stop += skip - 1
        return self.connection.getrange(f'{PREFIX}{DELIMITER}{key}', str(skip + start), str(stop))
        
    def _set_data_range(self, key: str, data_length: int, start: int, data: np.ndarray):
        skip = METADATA_SIZE + data_length
        self.connection.setrange(f'{PREFIX}{DELIMITER}{key}', str(skip + start), data.tobytes())

    def _check_dups(self, index: pd.Index, type_='column'):
        unq, unq_cnt = np.unique(index, return_counts=True)
        if len(unq) != len(index):
            dups = unq[unq_cnt > 1].tolist()
            raise ValueError(f'Duplicate {type_} name{"s: " + str(dups)  if len(dups) > 1 else ": " + str(dups[0])}')

    def add_to_index(self, key: str):
        for key,value in self._get_index_parts(key):
            self.connection.zadd(key, {value: 0.0})

    def remove_from_index(self, key: str):
        for key,value in self._get_index_parts(key):
            self.connection.zrem(key, value)
            if self.connection.zcard(key) == 0:
               self.connection.delete(key)

    def _get_index_parts(self, key: str):
        parts = key.split(DELIMITER)
        indicies = []
        for i in range(len(parts)):
            #self.connection.zadd(f'{INDEX_PREFIX}{"/".join(parts[0:i])}',{parts[i]:0})
            indicies.append((f'{"/".join([INDEX_PREFIX] + parts[0:i])}', parts[i]))
        return indicies


    def reindex(self):
        for key in self.connection.keys(f'{INDEX_PREFIX}*'):
            self.connection.delete(key)
        for key in self.connection.keys(f'{PREFIX}{DELIMITER}*'):
            self.add_to_index(key.decode()[len(PREFIX + DELIMITER):])


def saved_col(range_, args, _):
    if range_.start is not None and range_.stop is not None and range_.periodicity is not None:
        return get_client().read_series_data(args['key'], range_.start,
                        range_.stop, range_.periodicity)[3]
    else:
        data = get_client().read_series_data(args['key'])
        values = data[3][range_.start: range_.stop: range_.periodicity]
        if range_.start is None:
            range_.start = data[0]
        elif range_.start < 0:
            range_.start = data[1] + range_.start
        else:
            range_.start = data[0] + range_.start

        if range_.stop is None:
            range_.stop = data[1]
        elif range_.stop < 0:
            range_.stop = data[1] + range_.stop
        else:
            range_.stop = data[0] + range_.stop
        range_.periodicity_code = data[2]
        return values

@dataclass(repr=False)
class Saved(Word):
    name: str = 'saved'
    def __call__(self, key): return super().__call__(locals())
    def operate(self, stack):
        md = get_client()._read_metadata(self.args['key'])
        if not md.is_frame:
            stack.append(Column(self.args['key'], self.name, saved_col, self.args, []))
        else:   
            for header in get_client().read_frame_headers(self.args['key']):
                col_args = self.args.copy()
                col_args.update({'key': f'{self.args["key"]}:{header}'})
                stack.append(Column(header, self.name, saved_col, col_args, []))

    

