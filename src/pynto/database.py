from __future__ import annotations

import atexit
import datetime
import math
import os
import re
import sqlite3
import struct
import uuid
import warnings
from abc import ABC, abstractmethod
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
    if _CLIENT is None:
        # Check if Redis environment variables are set
        redis_configured = ('PYNTO_REDIS_HOST' in os.environ or 
                          'PYNTO_REDIS_PATH' in os.environ)
        
        if redis_configured:
            # Use Redis connection
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
            connection = RedisConnection(**args)
        else:
            # Default to SQLite connection
            db_file = os.environ.get('PYNTO_DB_FILE')
            connection = SQLiteConnection(db_file)
        
        _CLIENT = Db(connection=connection)
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

class DatabaseConnection(ABC):
    """Abstract base class for database connections used by pynto."""
    
    @abstractmethod
    def create_pipeline(self):
        """Create a pipeline for batching operations."""
        pass
    
    @abstractmethod
    def get_index_range_by_lex(self, key: str, min_lex: str, max_lex: str) -> list[bytes]:
        """Get items from sorted set within lexicographical range."""
        pass
    
    @abstractmethod
    def get_all_index_members(self, key: str) -> list[bytes]:
        """Get all members from sorted set."""
        pass
    
    @abstractmethod
    def delete_key(self, key: bytes) -> None:
        """Delete a key from the database."""
        pass
    
    @abstractmethod
    def remove_from_index(self, key: bytes, *members: bytes) -> None:
        """Remove members from a sorted set."""
        pass
    
    @abstractmethod
    def add_to_index(self, key: bytes, mapping: dict[bytes, float]) -> None:
        """Add members with scores to a sorted set."""
        pass
    
    @abstractmethod
    def set_data(self, key: bytes, offset: int, data: bytes) -> None:
        """Set bytes at a specific offset in a string key."""
        pass
    
    @abstractmethod
    def get_bytes(self, key: bytes) -> bytes:
        """Get the entire value of a string key as bytes."""
        pass
    
    @abstractmethod
    def get_range_bytes(self, key: bytes, start: int, end: int) -> bytes:
        """Get a range of bytes from a string key."""
        pass


class RedisConnection(DatabaseConnection):
    """Redis implementation of DatabaseConnection."""
    
    def __init__(self, **kwargs):
        if 'path' in kwargs:
           kwargs['connection_class'] = UnixDomainSocketConnection
        self._pool = redis.ConnectionPool(**kwargs)
    
    @property
    def _redis(self):
        return redis.Redis(connection_pool=self._pool)
    
    def create_pipeline(self):
        """Create a Redis pipeline for batching operations."""
        return RedisPipeline(self._redis.pipeline())
    
    def get_index_range_by_lex(self, key: str, min_lex: str, max_lex: str) -> list[bytes]:
        """Get items from sorted set within lexicographical range."""
        return self._redis.zrangebylex(key, min_lex, max_lex)
    
    def get_all_index_members(self, key: str) -> list[bytes]:
        """Get all members from sorted set."""
        return self._redis.zrange(key, 0, -1)
    
    def delete_key(self, key: bytes) -> None:
        """Delete a key from Redis."""
        self._redis.delete(key)
    
    def remove_from_index(self, key: bytes, *members: bytes) -> None:
        """Remove members from a Redis sorted set."""
        self._redis.zrem(key, *members)
    
    def add_to_index(self, key: bytes, mapping: dict[bytes, float]) -> None:
        """Add members with scores to a Redis sorted set."""
        self._redis.zadd(key, mapping)
    
    def set_data(self, key: bytes, offset: int, data: bytes) -> None:
        """Set bytes at a specific offset in a Redis string."""
        self._redis.setrange(key, offset, data)
    
    def get_bytes(self, key: bytes) -> bytes:
        """Get the entire value of a Redis string as bytes."""
        return self._redis.get(key) or b''
    
    def get_range_bytes(self, key: bytes, start: int, end: int) -> bytes:
        """Get a range of bytes from a Redis string."""
        return self._redis.getrange(key, start, end)


class SQLiteConnection(DatabaseConnection):
    """SQLite implementation of DatabaseConnection."""
    
    _open_connections = []
    
    def __init__(self, db_file: str = None):
        self.db_file = db_file or ":memory:"
        self._conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self._create_tables()
        
        # Track this connection for cleanup
        SQLiteConnection._open_connections.append(self._conn)
        
        # Register cleanup handler on first connection
        if len(SQLiteConnection._open_connections) == 1:
            atexit.register(SQLiteConnection._cleanup_connections)
    
    def _create_tables(self):
        """Create the required tables if they don't exist."""
        cursor = self._conn.cursor()
        
        # Table for sorted set (metadata p2m) - using BLOB for binary keys
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS p2m (
                member BLOB PRIMARY KEY
            )
        ''')
        
        # Table for p2d storage - using BLOB for binary keys
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS p2d (
                key BLOB PRIMARY KEY,
                value BLOB
            )
        ''')
        
        self._conn.commit()
    
    def close(self):
        """Close the SQLite connection."""
        if self._conn and self._conn in SQLiteConnection._open_connections:
            self._conn.close()
            SQLiteConnection._open_connections.remove(self._conn)
    
    @classmethod
    def _cleanup_connections(cls):
        """Close all open SQLite connections."""
        for conn in cls._open_connections[:]:  # Copy list to avoid modification during iteration
            try:
                conn.close()
            except Exception:
                pass  # Ignore errors during cleanup
        cls._open_connections.clear()
    
    def create_pipeline(self):
        """Create a SQLite pipeline for batching operations."""
        return SQLitePipeline(self)
    
    def get_index_range_by_lex(self, key: str, min_lex: str, max_lex: str) -> list[bytes]:
        """Get items from sorted set within lexicographical range."""
        cursor = self._conn.cursor()
        
        # Convert Redis-style range to bytes for direct comparison
        min_val = min_lex[1:] if min_lex.startswith('[') else min_lex
        max_val = max_lex[1:] + '\xff' if max_lex.startswith('[') else max_lex
        
        # Convert to bytes for direct BLOB comparison
        min_val_bytes = min_val.encode('utf-8')
        max_val_bytes = max_val.encode('utf-8')
        
        cursor.execute('''
            SELECT member FROM p2m 
            WHERE member >= ? AND member < ?
            ORDER BY member
        ''', (min_val_bytes, max_val_bytes))
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_index_members(self, key: str) -> list[bytes]:
        """Get all members from sorted set."""
        cursor = self._conn.cursor()
        cursor.execute('SELECT member FROM p2m ORDER BY member')
        return [row[0] for row in cursor.fetchall()]
    
    def delete_key(self, key: bytes) -> None:
        """Delete a key from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute('DELETE FROM p2d WHERE key = ?', (key,))
        self._conn.commit()
    
    def remove_from_index(self, key: bytes, *members: bytes) -> None:
        """Remove members from SQLite sorted set."""
        cursor = self._conn.cursor()
        for member in members:
            cursor.execute('DELETE FROM p2m WHERE member = ?', (member,))
        self._conn.commit()
    
    def add_to_index(self, key: bytes, mapping: dict[bytes, float]) -> None:
        """Add members to SQLite sorted set (scores ignored for now)."""
        cursor = self._conn.cursor()
        for member, score in mapping.items():
            cursor.execute('''
                INSERT OR REPLACE INTO p2m (member) VALUES (?)
            ''', (member,))
        self._conn.commit()
    
    def set_data(self, key: bytes, offset: int, data: bytes) -> None:
        """Set bytes at a specific offset in SQLite."""
        cursor = self._conn.cursor()
        # Get existing data
        cursor.execute('SELECT value FROM p2d WHERE key = ?', (key,))
        row = cursor.fetchone()
        
        if row is None:
            # Create new entry with zeros up to offset, then data
            new_data = b'\x00' * offset + data
        else:
            existing_data = row[0] or b''
            # Extend existing data if needed
            if len(existing_data) < offset:
                existing_data += b'\x00' * (offset - len(existing_data))
            
            # Replace bytes at offset
            new_data = existing_data[:offset] + data + existing_data[offset + len(data):]
        
        cursor.execute('''
            INSERT OR REPLACE INTO p2d (key, value) VALUES (?, ?)
        ''', (key, new_data))
        self._conn.commit()
    
    def get_bytes(self, key: bytes) -> bytes:
        """Get the entire value as bytes from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute('SELECT value FROM p2d WHERE key = ?', (key,))
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else b''
    
    def get_range_bytes(self, key: bytes, start: int, end: int) -> bytes:
        """Get a range of bytes from SQLite using substr."""
        cursor = self._conn.cursor()
        
        if start == -1 and end == 0:
            return b''  # Redis convention for empty range
        
        if end == -1:
            # Get from start to end of blob
            cursor.execute('SELECT substr(value, ?, length(value) - ? + 1) FROM p2d WHERE key = ?', 
                          (start + 1, start, key))  # SQLite substr is 1-indexed
        else:
            # Get specific range
            length = end - start + 1
            cursor.execute('SELECT substr(value, ?, ?) FROM p2d WHERE key = ?', 
                          (start + 1, length, key))  # SQLite substr is 1-indexed
        
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else b''


class SQLitePipeline:
    """Pipeline for SQLite operations."""
    
    def __init__(self, conn):
        self._conn = conn
        self._commands = []
        self._results = []
    
    def delete_key(self, key: bytes) -> None:
        """Add delete command to pipeline."""
        self._commands.append(('delete', key))
    
    def remove_from_index(self, key: bytes, *members: bytes) -> None:
        """Add remove from sorted set command to pipeline."""
        for member in members:
            self._commands.append(('remove_index', member))
    
    def add_to_index(self, key: bytes, mapping: dict[bytes, float]) -> None:
        """Add members to sorted set in pipeline."""
        for member, score in mapping.items():
            self._commands.append(('add_index', member))
    
    def set_data(self, key: bytes, offset: int, data: bytes) -> None:
        """Add set range bytes command to pipeline."""
        self._commands.append(('set_data', key, offset, data))
    
    def get_range_bytes(self, key: bytes, start: int, end: int) -> None:
        """Add get range bytes command to pipeline."""
        self._commands.append(('get_range_bytes', key, start, end))
    
    def execute(self):
        """Execute all commands in the pipeline."""
        results = []
        
        for cmd in self._commands:
            if cmd[0] == 'delete':
                self._conn.delete_key(cmd[1])
                results.append(None)
                
            elif cmd[0] == 'remove_index':
                self._conn.remove_from_index(None, cmd[1])  # key not used in SQLite implementation
                results.append(None)
                
            elif cmd[0] == 'add_index':
                self._conn.add_to_index(None, {cmd[1]: 0.0})  # key and score not used in SQLite implementation
                results.append(None)
                
            elif cmd[0] == 'set_data':
                key, offset, data = cmd[1], cmd[2], cmd[3]
                self._conn.set_data(key, offset, data)
                results.append(None)
                
            elif cmd[0] == 'get_range_bytes':
                key, start, end = cmd[1], cmd[2], cmd[3]
                result = self._conn.get_range_bytes(key, start, end)
                results.append(result)
        
        self._commands.clear()
        return results


class RedisPipeline:
    """Wrapper for Redis pipeline to abstract Redis-specific methods."""
    
    def __init__(self, pipeline):
        self._pipeline = pipeline
    
    def delete_key(self, key: bytes) -> None:
        """Add delete command to pipeline."""
        self._pipeline.delete(key)
    
    def remove_from_index(self, key: bytes, *members: bytes) -> None:
        """Add remove from sorted set command to pipeline."""
        self._pipeline.zrem(key, *members)
    
    def add_to_index(self, key: bytes, mapping: dict[bytes, float]) -> None:
        """Add members with scores to sorted set in pipeline."""
        self._pipeline.zadd(key, mapping)
    
    def set_data(self, key: bytes, offset: int, data: bytes) -> None:
        """Add set range bytes command to pipeline."""
        self._pipeline.setrange(key, offset, data)
    
    def get_range_bytes(self, key: bytes, start: int, end: int) -> None:
        """Add get range bytes command to pipeline."""
        self._pipeline.getrange(key, start, end)
    
    def execute(self):
        """Execute all commands in the pipeline."""
        return self._pipeline.execute()


class Db:

    def __init__(self, connection: DatabaseConnection = None, **kwargs):
        if connection is not None:
            self.connection = connection
        else:
            # Default to Redis connection for backward compatibility
            self.connection = RedisConnection(**kwargs)

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
        key = self.make_safe(*self.split_key(key))
        mds = [Metadata.unpack(p) for p in
                    self.connection.get_index_range_by_lex(INDEX, f'[{key}', f'[{key}\xff')]
        mds.sort(key=attrgetter('ordinal'))
        return mds

    def all_keys(self) -> list[Metadata]:
        mds = [Metadata.unpack(p) for p in
                    self.connection.get_all_index_members(INDEX)]
        mds.sort(key=attrgetter('ordinal'))
        keys = {}
        for md in mds:
            if md.key not in keys:
                keys[md.key] = []
            keys[md.key].append((md.col_header, md.row_header))
        return keys

    def delete_all(self) -> list[Metadata]:
        p = self.connection.create_pipeline()
        for packed in self.connection.get_all_index_members(INDEX):
            p.delete_key(Metadata.unpack(packed).data_key)
            p.remove_from_index(INDEX, packed)
        p.execute()

    def __setitem__(self, key: str, pandas: pd.Series | pd.DataFrame):
        saved: dict[tuple[str,str],Metadata] = {}
        series: list[tuple[str,str,np.ndarray]] = []
        frame, column, row = self.split_key(key)
        assert column is None or isinstance(pandas, pd.Series) or pandas.shape[1] == 1, \
                'Can only assign one column to a specific column key'
        safe_key = self.make_safe(frame, column, row)
        for p in self.connection.get_index_range_by_lex(INDEX, f'[{safe_key}', f'[{safe_key}\xff'):
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
        p = self.connection.create_pipeline()
        toadd, todel = [],[]
        for col, row, s in series:
            assert not isinstance(s.dtype,  pd.api.extensions.ExtensionDtype)
            type_ = DataType.from_dtype(s.dtype.str)
            s = _trim_values(s)
            if s is None:
                continue
            range_ = Range.from_index(s.index)
            data = s.to_numpy()
            md_tuple = saved.get((col,row))
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
                            self.connection.get_bytes(series_md.data_key)[existing_start:]])
                    data_offset = 0
                    series_md.start = range_.start
                else: #series_md.range_.stop <= range_.start
                    data_offset = range_.start - series_md.start
                series_md.stop = max(range_.stop, series_md.stop)
                toadd.append(series_md)
                todel.append(packed)
            p.set_data(series_md.data_key, data_offset * type_.length, data.tobytes())
        if todel:
            p.remove_from_index(INDEX, *todel)
        if toadd:
            p.add_to_index(INDEX, {m.pack(): 0.0 for m in toadd})
        p.execute()

    def __delitem__(self, key: str):
        key = self.make_safe(*self.split_key(key))
        p = self.connection.create_pipeline()
        for packed in self.connection.get_index_range_by_lex(INDEX, f'[{key}', f'[{key}\xff'):
            p.delete_key(Metadata.unpack(packed).data_key)
            p.remove_from_index(INDEX, packed)
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
                self.connection.get_index_range_by_lex(INDEX, f'[{safe_key}', f'[{safe_key}\xff')]
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
        p = self.connection.create_pipeline()
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

    def _req(self, saved: Metadata, start: int, stop: int, p)  -> int:
        if start < saved.stop and stop >= saved.start:
            offset = max(0,saved.start - start)
            start = max(start, saved.start)
            stop = min(stop, saved.stop)
            p.get_range_bytes(saved.data_key,
                (start - saved.start) * saved.type_.length,
                (stop - saved.start) * saved.type_.length - 1)
        else: # no overlap with saved 
            offset = -1
            p.get_range_bytes(saved.data_key, -1, 0)
        return offset
