from __future__ import annotations

import datetime
import math
import os
import re
import struct
import uuid
import warnings
from collections import namedtuple
from dataclasses import asdict, astuple, dataclass, field
from enum import Enum
from operator import attrgetter
from typing import Any, Dict, Iterable, NamedTuple, Union

import numpy as np
import pandas as pd
import redis
from redis.connection import UnixDomainSocketConnection

from .periods import Period, Periodicity, Range, datelike

INDEX = 'p2m'
DATA_PREFIX = 'p2d:'.encode()

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

def _trim_values(series: pd.Series) -> pd.Series:
    if series.values.dtype.kind == 'f':
        nz = (~np.isnan(series.to_numpy())).nonzero()[0]
        if len(nz) == 0: # don't save if all nans
            series = None
        else:
            series = series.iloc[nz.min():nz.max()+1]
    return series

def _check_dups(index: np.ndarray, type_='column') -> np.ndarray:
    unq, unq_cnt = np.unique(index, return_counts=True)
    if len(unq) != len(index):
        dups = unq[unq_cnt > 1].tolist()
        raise ValueError(f'Duplicate {type_} name{"s: " + str(dups)  if len(dups) > 1 else ": " + str(dups[0])}')
    return index


@dataclass
class DataTypeMixin:
    dtype: str
    pad_value: Any
    length: int

class DataType(DataTypeMixin, Enum):
    F =  '<f8', np.nan, 8
    I = '<i8', 0, 8
    B = '|b1', False, 1

    def __str__(self):
        return self.name

    @classmethod
    def from_dtype(cls, code: str) -> DataType:
        for p in cls:
            if p.dtype == code:
                return p
        raise ValueError(f'Unsupported dtype: "{code}"')

METADATA_FORMAT = f'<256s128s128sLllccdd16s'

@dataclass
class Metadata(Range):
    type_: DataType
    key: str
    ordinal: int
    col_header: str
    row_header: str
    id_: uuid.UUID = field(default_factory = uuid.uuid4)
    create_timestamp: datetime.datetime = field(default_factory= \
        lambda: datetime.datetime.now(datetime.UTC))
    update_timestamp: datetime.datetime = field(default_factory= \
        lambda: datetime.datetime.now(datetime.UTC))

    @property
    def data_key(self):
        return DATA_PREFIX + self.id_.bytes

    def pack(self, keep_timestamp: bool = False) -> bytes:
        update = self.update_timestamp.timestamp() if keep_timestamp else \
                    datetime.datetime.now(datetime.UTC).timestamp()
        return struct.pack(METADATA_FORMAT,
                    self.key.encode(),
                    self.col_header.encode(),
                    self.row_header.encode(),
                    self.ordinal,
                    self.start,
                    self.stop,
                    self.periodicity.code.encode(),
                    self.type_.name.encode(),
                    self.create_timestamp.timestamp(),
                    update,
                    self.id_.bytes)

    @classmethod
    def unpack(cls, bytes_: bytes) -> Metadata:
        key, col_header, row_header, ordinal, start, stop, per, \
        typ,create, update, id_ = struct.unpack(METADATA_FORMAT, bytes_)
        return cls(start, stop, Periodicity[per.decode()],
                DataType[typ.decode()],
                key.decode().strip('\x00'),
                ordinal,
                col_header.decode().strip('\x00'),
                row_header.decode().strip('\x00'),
                uuid.UUID(bytes=id_),
                datetime.datetime.fromtimestamp(create),
                datetime.datetime.fromtimestamp(update))

class Db:

    def __init__(self, **kwargs):
        if 'path' in kwargs:
           kwargs['connection_class'] = UnixDomainSocketConnection
        self._pool = redis.ConnectionPool(**kwargs)

    @property
    def connection(self):
        return redis.Redis(connection_pool=self._pool)

    def split_key(self, key: str) -> str:
        pattern = r'([^#]+)(?:#([^$]*))?(?:\$(.*))?'
        return  re.match(pattern, key).groups()

    def make_safe(self, frame: str, column: str, row: str) -> str:
        key = struct.pack('<256s', frame.encode())
        if column:
            key += struct.pack('<128s', column.encode())
        if row:
            key += struct.pack('<128s', row.encode())
        return key.decode()

    def get_metadata(self, key: str) -> list[Metadata]:
        k = self.make_safe(*self.split_key(key))
        mds = [Metadata.unpack(p) for p in
                    self.connection.zrangebylex(INDEX, f'[{k}', f'[{k}\xff')]
        if not mds:
            raise KeyError(f'Db key \'{key}\' not found')
        mds.sort(key=attrgetter('ordinal'))
        return mds

    def columns(self, key: str) -> list[str]:
        return list(dict.fromkeys([md.col_header for md in self.get_metadata(key)]))

    def all_keys(self) -> list[str]:
        return list(set([Metadata.unpack(p).key for p in
                    self.connection.zrange(INDEX, 0, -1)]))

    def all_series(self) -> dict[str,list[tuple[str,str]]]:
        mds = [Metadata.unpack(p) for p in
                    self.connection.zrange(INDEX, 0, -1)]
        mds.sort(key=attrgetter('ordinal'))
        keys = {}
        for md in mds:
            if md.key not in keys:
                keys[md.key] = []
            keys[md.key].append((md.col_header, md.row_header))
        return keys

    def diags(self, key: str) -> pd.DataFrame:
        cols = []
        for header in self.columns(key):
            col = self[f'{key}#{header}${header}']
            col.index = col.index.droplevel(1)
            cols.append(col)
        return pd.concat(cols, axis=1)



    def delete_all(self) -> list[Metadata]:
        p = self.connection.pipeline()
        for packed in self.connection.zrange(INDEX, 0, -1):
            p.delete(Metadata.unpack(packed).data_key)
            p.zrem(INDEX, packed)
        p.execute()

    def __setitem__(self, key: str, pandas: pd.Series | pd.DataFrame):
        saved: dict[tuple[str,str],Metadata] = {}
        series: list[tuple[str,str,np.ndarray]] = []
        frame, column, row = self.split_key(key)
        assert column is None or isinstance(pandas, pd.Series) or pandas.shape[1] == 1, \
                'Can only assign one column to a specific column key'
        safe_key = self.make_safe(frame, column, row)
        for p in self.connection.zrangebylex(INDEX, f'[{safe_key}', f'[{safe_key}\xff'):
            md = Metadata.unpack(p)
            saved[(md.col_header, md.row_header)] = (md, p)
        if isinstance(pandas, pd.Series):
            series.append((column or pandas.name or '', row or '', pandas))
        else:
            columns = _check_dups(pandas.columns.values)
            if isinstance(pandas.index, pd.MultiIndex):
                assert len(pandas.index.levels) == 2, 'Too many axes.'
                rows = _check_dups(pandas.loc[pandas.index[0][0]].index.values, 'Rows')
                index=pandas.index.remove_unused_levels().levels[0]
                flat = pandas.values.reshape(len(index), len(columns) * len(rows))
                for i, (row, col) in enumerate(zip(np.repeat(rows,len(columns)),
                                                np.tile(columns,len(rows)))):
                    series.append((str(col), str(row),
                                    pd.Series(flat[:,i],index=index)))
            else:
                series.extend([(column or h,'',s) for h,s in pandas.items()])
        p = self.connection.pipeline()
        toadd, todel = [],[]
        for col, row, s in series:
            assert not isinstance(s.dtype,  pd.api.extensions.ExtensionDtype)
            type_ = DataType.from_dtype(s.dtype.str)
            md_tuple = saved.get((col,row))
            if not md_tuple:
                s = _trim_values(s)
                if s is None:
                    continue
            range_ = Range.from_index(s.index)
            data = s.to_numpy()
            if not md_tuple:
                data_offset = 0
                series_md = Metadata(range_.start, range_.stop, range_.periodicity,
                                        type_, frame, len(saved), col, row)
                saved[(col, row)] = series_md
                toadd.append(series_md)
            else:
                series_md, packed = md_tuple
                assert series_md.periodicity.code == range_.periodicity.code, \
                    f'Periodicity does match saved for {key}'
                assert series_md.type_ == type_, \
                    f'Datatype does match saved for {key}'
                assert series_md.start <= range_.stop and \
                        series_md.stop >= range_.start,  \
                    f'Data not contiguous with saved for {key}'
                if series_md.start > range_.start:
                    if range_.stop < series_md.stop:
                        existing_start = range_.stop - series_md.start
                        data = np.hstack([data,
                            self.connection.get(series_md.data_key)[existing_start:]])
                    data_offset = 0
                    series_md.start = range_.start
                else: #series_md.range_.stop <= range_.start
                    data_offset = range_.start - series_md.start
                series_md.stop = max(range_.stop, series_md.stop)
                toadd.append(series_md)
                todel.append(packed)
            p.setrange(series_md.data_key, data_offset * type_.length, data.tobytes())
        if todel:
            p.zrem(INDEX, *todel)
        if toadd:
            p.zadd(INDEX, {m.pack(): 0.0 for m in toadd})
        p.execute()

    def __delitem__(self, key: str):
        key = self.make_safe(*self.split_key(key))
        p = self.connection.pipeline()
        for packed in self.connection.zrangebylex(INDEX, f'[{key}', f'[{key}\xff'):
            p.delete(Metadata.unpack(packed).data_key)
            p.zrem(INDEX, packed)
        p.execute()

    def __getitem__(self, args: str | tuple[str,slice]) -> pd.Frame:
        if isinstance(args, str):
            key =  args
            start, stop = None, None
        else:
            key = args[0]
            start, stop = args[1].start, args[1].stop
        safe_key = self.make_safe(*self.split_key(key))
        mds = [Metadata.unpack(d) for d in \
                self.connection.zrangebylex(INDEX, f'[{safe_key}', f'[{safe_key}\xff')]
        if not mds:
            raise KeyError(f'Key "{key}" not found')
        mds.sort(key=attrgetter('ordinal'))
        per = mds[0].periodicity
        if start is None:
            start = min([md.start for md in mds])
        elif not isinstance(start, int):
            start = per[start].start
        if stop is None:
            stop = max([md.stop for md in mds])
        elif not isinstance(stop, int):
            stop = per[stop].start
        ary = np.full((stop - start,len(mds)), mds[0].type_.pad_value, order='F')
        p = self.connection.pipeline()
        offsets = [self._req(md, start, stop, p) for md in mds]
        for i, md, offset, bytes_ in zip(range(len(mds)),mds, offsets, p.execute()):
            if len(bytes_) > 0:
                data = np.frombuffer(bytes_, md.type_.dtype)
                ary[offset:offset + len(data),i] = data
        cols = np.array([md.col_header for md in mds])
        index = per[start:stop].to_index()
        if mds[0].row_header:
            columns: dict[str,None] = {} #dict not set preserves order
            rows: dict[str,None] = {}
            for md in mds:
                rows[md.row_header] = None
                columns[md.col_header] = None
            index = pd.MultiIndex.from_product([index, list(rows.keys())])
            ary = ary.reshape(ary.shape[0]*len(rows),len(columns))
            cols = np.array(columns.keys())
        df = pd.DataFrame(ary,columns=cols, index=index)
        return df

    def _req(self, saved: Metadata, start: int, stop: int, p: redis.client.Pipeline)  -> int:
        if start < saved.stop and stop >= saved.start:
            offset = max(0,saved.start - start)
            start = max(start, saved.start)
            stop = min(stop, saved.stop)
            p.getrange(saved.data_key,
                (start - saved.start) * saved.type_.length,
                (stop - saved.start) * saved.type_.length - 1)
        else: # no overlap with saved 
            offset = -1
            p.getrange(saved.data_key, -1, 0)
        return offset
