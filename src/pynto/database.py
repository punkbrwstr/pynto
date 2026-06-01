from __future__ import annotations

import datetime
import logging
import os
import re
import struct
import uuid
from dataclasses import dataclass, field
from enum import Enum
from operator import attrgetter
from typing import Any

import numpy as np
import pandas as pd

from .connection import (
    Batch,
    Connection,
    DynamoDBConnection,
    RedisConnection,
    S3Connection,
    SQLiteConnection,
)
from .periods import Periodicity, Range

INDEX = 'p2m'
DATA_PREFIX = 'p2d:'.encode()
KEY_LENGTH = 96
COL_HEADER_LENGTH = 64
ROW_HEADER_LENGTH = 64

METADATA_FORMAT = f'<{KEY_LENGTH}s{COL_HEADER_LENGTH}s{ROW_HEADER_LENGTH}sLllccdd16s'
OLD_METADATA_FORMAT = '<256s128s128sLllccdd16s'
METADATA_SIZE = struct.calcsize(METADATA_FORMAT)
OLD_METADATA_SIZE = struct.calcsize(OLD_METADATA_FORMAT)
COPY_BATCH_SIZE = 100
logger = logging.getLogger(__name__)

_CLIENT: Db | None = None


def get_client() -> Db:
    global _CLIENT
    # if not '_CLIENT' in globals():
    if _CLIENT is None:
        database = os.environ.get('PYNTO_DATABASE', '').lower()
        if database == 'redis':
            _CLIENT = Db(**_redis_kwargs_from_env())
        elif database == 's3':
            _CLIENT = Db(connection=S3Connection())
        elif database == 'dynamodb':
            _CLIENT = Db(connection=DynamoDBConnection())
        elif database == 'sqlite':
            _CLIENT = Db(connection=SQLiteConnection())
        elif _has_redis_env():
            _CLIENT = Db(**_redis_kwargs_from_env())
        elif _has_s3_env():
            _CLIENT = Db(connection=S3Connection())
        elif _has_dynamodb_env():
            _CLIENT = Db(connection=DynamoDBConnection())
        elif _has_sqlite_env():
            _CLIENT = Db(connection=SQLiteConnection())
        else:
            _CLIENT = Db(**_redis_kwargs_from_env())
    return _CLIENT


def set_client(connection: Connection) -> Db:
    global _CLIENT
    _CLIENT = Db(connection=connection)
    return _CLIENT


def use_redis(**kwargs: Any) -> Db:
    args = _redis_kwargs_from_env()
    args.update(kwargs)
    return set_client(RedisConnection(**args))


def use_s3(**kwargs: Any) -> Db:
    return set_client(S3Connection(**kwargs))


def use_dynamodb(**kwargs: Any) -> Db:
    return set_client(DynamoDBConnection(**kwargs))


def use_sqlite(**kwargs: Any) -> Db:
    return set_client(SQLiteConnection(**kwargs))


def copy_data(
    source: Db | Connection,
    destination: Db | Connection,
    clear: bool = False,
    auxiliary_prefix: str | None = None,
) -> None:
    _copy_data(_as_db(source), _as_db(destination), clear, auxiliary_prefix)


def sync_data(
    source: Db | Connection,
    destination: Db | Connection,
    clear: bool = False,
    auxiliary_prefix: str | None = None,
) -> SyncResult:
    return _sync_data(_as_db(source), _as_db(destination), clear, auxiliary_prefix)


def copy_redis_to_s3(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> None:
    logger.info('Copying pynto data from Redis to S3')
    copy_data(
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        Db(connection=S3Connection()),
        clear,
        auxiliary_prefix,
    )


def sync_redis_to_s3(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> SyncResult:
    logger.info('Syncing pynto data from Redis to S3')
    return sync_data(
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        Db(connection=S3Connection()),
        clear,
        auxiliary_prefix,
    )


def copy_s3_to_redis(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> None:
    logger.info('Copying pynto data from S3 to Redis')
    copy_data(
        Db(connection=S3Connection()),
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        clear,
        auxiliary_prefix,
    )


def sync_s3_to_redis(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> SyncResult:
    logger.info('Syncing pynto data from S3 to Redis')
    return sync_data(
        Db(connection=S3Connection()),
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        clear,
        auxiliary_prefix,
    )


def copy_redis_to_dynamodb(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> None:
    logger.info('Copying pynto data from Redis to DynamoDB')
    copy_data(
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        Db(connection=DynamoDBConnection()),
        clear,
        auxiliary_prefix,
    )


def sync_redis_to_dynamodb(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> SyncResult:
    logger.info('Syncing pynto data from Redis to DynamoDB')
    return sync_data(
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        Db(connection=DynamoDBConnection()),
        clear,
        auxiliary_prefix,
    )


def copy_dynamodb_to_redis(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> None:
    logger.info('Copying pynto data from DynamoDB to Redis')
    copy_data(
        Db(connection=DynamoDBConnection()),
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        clear,
        auxiliary_prefix,
    )


def sync_dynamodb_to_redis(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> SyncResult:
    logger.info('Syncing pynto data from DynamoDB to Redis')
    return sync_data(
        Db(connection=DynamoDBConnection()),
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        clear,
        auxiliary_prefix,
    )


def copy_redis_to_sqlite(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> None:
    logger.info('Copying pynto data from Redis to SQLite')
    copy_data(
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        Db(connection=SQLiteConnection()),
        clear,
        auxiliary_prefix,
    )


def sync_redis_to_sqlite(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> SyncResult:
    logger.info('Syncing pynto data from Redis to SQLite')
    return sync_data(
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        Db(connection=SQLiteConnection()),
        clear,
        auxiliary_prefix,
    )


def copy_sqlite_to_redis(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> None:
    logger.info('Copying pynto data from SQLite to Redis')
    copy_data(
        Db(connection=SQLiteConnection()),
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        clear,
        auxiliary_prefix,
    )


def sync_sqlite_to_redis(
    clear: bool = False, auxiliary_prefix: str | None = None
) -> SyncResult:
    logger.info('Syncing pynto data from SQLite to Redis')
    return sync_data(
        Db(connection=SQLiteConnection()),
        Db(connection=RedisConnection(**_redis_kwargs_from_env())),
        clear,
        auxiliary_prefix,
    )


def migrate_redis_metadata() -> int:
    connection = RedisConnection(**_redis_kwargs_from_env())
    migrated = 0
    batch = connection.batch()
    for packed in connection.set_members(INDEX):
        if len(packed) == METADATA_SIZE:
            continue
        md = Metadata.unpack(packed)
        repacked = md.pack(keep_timestamp=True)
        batch.remove_set_members(INDEX, packed)
        batch.add_set_members(INDEX, [repacked])
        migrated += 1
        if migrated % COPY_BATCH_SIZE == 0:
            batch.execute()
            batch = connection.batch()
    if migrated:
        batch.execute()
    return migrated


@dataclass
class SyncResult:
    frames_seen: int = 0
    frames_copied: int = 0
    series_seen: int = 0
    series_copied: int = 0
    series_appended: int = 0
    series_current: int = 0
    series_skipped_older: int = 0


def _copy_data(
    source: Db,
    destination: Db,
    clear: bool = False,
    auxiliary_prefix: str | None = None,
) -> None:
    if clear:
        logger.info('Clearing destination pynto data')
        destination.delete_all()
        logger.info('Destination cleared')
    if auxiliary_prefix is not None:
        _copy_auxiliary_data(source.connection, destination.connection, auxiliary_prefix)
    keys = source.keys()
    total = len(keys)
    logger.info('Found %s source frames to copy', total)
    for count, key in enumerate(keys, start=1):
        frame = source[key]
        destination[key] = frame
        logger.info(
            'Copied %s/%s: %s (%s rows, %s columns)',
            count,
            total,
            key,
            frame.shape[0],
            frame.shape[1],
        )
    logger.info('Copy complete: %s source frames copied', total)


def _sync_data(
    source: Db,
    destination: Db,
    clear: bool = False,
    auxiliary_prefix: str | None = None,
) -> SyncResult:
    if clear:
        logger.info('Clearing destination pynto data')
        destination.delete_all()
        logger.info('Destination cleared')
    if auxiliary_prefix is not None:
        _copy_auxiliary_data(source.connection, destination.connection, auxiliary_prefix)
    keys = source.keys()
    total = len(keys)
    result = SyncResult(frames_seen=total)
    logger.info('Found %s source frames to sync', total)
    for count, key in enumerate(keys, start=1):
        _sync_frame(source, destination, key, count, total, result)
    logger.info(
        'Sync complete: %s frames seen, %s frames copied, %s series copied, '
        '%s series appended, %s series already current',
        result.frames_seen,
        result.frames_copied,
        result.series_copied,
        result.series_appended,
        result.series_current,
    )
    return result


def _sync_frame(
    source: Db,
    destination: Db,
    key: str,
    count: int,
    total: int,
    result: SyncResult,
) -> None:
    source_mds = source.get_metadata(key)
    result.series_seen += len(source_mds)
    try:
        destination_mds = destination.get_metadata(key)
    except KeyError:
        frame = source[key]
        destination[key] = frame
        result.frames_copied += 1
        result.series_copied += len(source_mds)
        logger.info(
            'Synced %s/%s: copied new frame %s (%s rows, %s columns)',
            count,
            total,
            key,
            frame.shape[0],
            frame.shape[1],
        )
        return

    destination_by_header = {
        (md.col_header, md.row_header): md for md in destination_mds
    }
    copied = 0
    appended = 0
    current = 0
    skipped_older = 0
    for source_md in source_mds:
        series_key = _metadata_key(source_md)
        destination_md = destination_by_header.get(
            (source_md.col_header, source_md.row_header)
        )
        if destination_md is None:
            _copy_series_data(source, destination, source_md)
            copied += 1
            result.series_copied += 1
            logger.info(
                'Copied new series %s (%s rows)',
                series_key,
                source_md.stop - source_md.start,
            )
            continue

        _check_sync_compatible(source_md, destination_md, series_key)
        if source_md.start < destination_md.start:
            skipped_older += 1
            result.series_skipped_older += 1
            logger.info(
                'Skipped older source data for %s: source start %s, '
                'destination start %s',
                series_key,
                source_md.start,
                destination_md.start,
            )
        if source_md.stop <= destination_md.stop:
            current += 1
            result.series_current += 1
            continue
        if source_md.start > destination_md.stop:
            raise ValueError(
                f'Cannot sync {series_key}: source starts at {source_md.start}, '
                f'destination stops at {destination_md.stop}'
            )
        frame = source[(series_key, slice(destination_md.stop, source_md.stop))]
        destination[series_key] = frame
        appended += 1
        result.series_appended += 1
        logger.info(
            'Appended %s rows to %s',
            source_md.stop - destination_md.stop,
            series_key,
        )

    logger.info(
        'Synced %s/%s: %s (%s copied, %s appended, %s current, %s older skipped)',
        count,
        total,
        key,
        copied,
        appended,
        current,
        skipped_older,
    )


def _copy_series_data(source: Db, destination: Db, md: Metadata) -> None:
    data = source.connection.get(md.data_key)
    batch = destination.connection.batch()
    batch.set_range(md.data_key, 0, data)
    batch.add_set_members(INDEX, [md.pack(keep_timestamp=True)])
    batch.execute()


def _metadata_key(md: Metadata) -> str:
    if not md.col_header:
        return md.key
    key = f'{md.key}#{md.col_header}'
    if md.row_header:
        key += f'${md.row_header}'
    return key


def _check_sync_compatible(
    source_md: Metadata, destination_md: Metadata, key: str
) -> None:
    if source_md.periodicity != destination_md.periodicity:
        raise ValueError(
            f'Cannot sync {key}: source periodicity {source_md.periodicity} '
            f'does not match destination periodicity {destination_md.periodicity}'
        )
    if source_md.type_ != destination_md.type_:
        raise ValueError(
            f'Cannot sync {key}: source type {source_md.type_} '
            f'does not match destination type {destination_md.type_}'
        )


def _copy_auxiliary_data(
    source: Connection, destination: Connection, prefix: str
) -> None:
    hash_keys = source.hash_keys(prefix)
    logger.info(
        'Found %s source hash keys with prefix %r to copy', len(hash_keys), prefix
    )
    for count, key in enumerate(hash_keys, start=1):
        mapping = {
            field.decode(): value
            for field, value in source.hgetall(key).items()
        }
        if mapping:
            destination.hmset(key, mapping)
        logger.info(
            'Copied auxiliary hash %s/%s: %s (%s fields)',
            count,
            len(hash_keys),
            key,
            len(mapping),
        )

    set_keys = source.lex_set_keys(prefix)
    logger.info(
        'Found %s source lexical set keys with prefix %r to copy',
        len(set_keys),
        prefix,
    )
    for count, key in enumerate(set_keys, start=1):
        members = source.set_members(key)
        if members:
            destination.zadd(key, {member.decode(): 0.0 for member in members})
        logger.info(
            'Copied auxiliary lexical set %s/%s: %s (%s members)',
            count,
            len(set_keys),
            key,
            len(members),
        )


def _as_db(database: Db | Connection) -> Db:
    if isinstance(database, Db):
        return database
    return Db(connection=database)


def _has_s3_env() -> bool:
    return 'PYNTO_S3_BUCKET' in os.environ


def _has_redis_env() -> bool:
    return any(
        name in os.environ
        for name in (
            'PYNTO_REDIS_PASSWORD',
            'PYNTO_REDIS_PATH',
            'PYNTO_REDIS_HOST',
            'PYNTO_REDIS_PORT',
        )
    )


def _has_dynamodb_env() -> bool:
    return any(
        name in os.environ
        for name in ('PYNTO_DYNAMODB_DATA_TABLE', 'PYNTO_DYNAMODB_SET_TABLE')
    )


def _has_sqlite_env() -> bool:
    return 'PYNTO_SQLITE_PATH' in os.environ


def _redis_kwargs_from_env() -> dict[str, Any]:
    args: dict[str, Any] = {}
    if 'PYNTO_REDIS_PASSWORD' in os.environ:
        args['password'] = os.environ['PYNTO_REDIS_PASSWORD']
    if 'PYNTO_REDIS_PATH' in os.environ:
        args['path'] = os.environ['PYNTO_REDIS_PATH']
    else:
        if 'PYNTO_REDIS_HOST' in os.environ:
            args['host'] = os.environ['PYNTO_REDIS_HOST']
        if 'PYNTO_REDIS_PORT' in os.environ:
            args['port'] = os.environ['PYNTO_REDIS_PORT']
    return args


def _trim_values(series: pd.Series) -> pd.Series | None:
    if series.values.dtype.kind == 'f':
        nz = (~np.isnan(series.to_numpy())).nonzero()[0]
        if len(nz) == 0:  # don't save if all nans
            return None
        else:
            series = series.iloc[nz.min() : nz.max() + 1]
    return series


def _check_dups(index: np.ndarray, type_: str = 'column') -> np.ndarray:
    unq, unq_cnt = np.unique(index, return_counts=True)
    if len(unq) != len(index):
        dups = unq[unq_cnt > 1].tolist()
        raise ValueError(
            f'Duplicate {type_} name{"s: " + str(dups) if len(dups) > 1 else ": " + str(dups[0])}'
        )
    return index


@dataclass
class DataTypeMixin:
    dtype: str
    pad_value: Any
    length: int


class DataType(DataTypeMixin, Enum):
    F = '<f8', np.nan, 8
    N = '<i8', 0, 8
    B = '|b1', False, 1

    def __str__(self):
        return self.name

    @classmethod
    def from_dtype(cls, code: str) -> DataType:
        for p in cls:
            if p.dtype == code:
                return p
        raise ValueError(f'Unsupported dtype: "{code}"')


@dataclass
class Metadata(Range):
    type_: DataType
    key: str
    ordinal: int
    col_header: str
    row_header: str
    id_: uuid.UUID = field(default_factory=uuid.uuid4)
    create_timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    update_timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    @property
    def data_key(self):
        return DATA_PREFIX + self.id_.bytes

    def pack(self, keep_timestamp: bool = False) -> bytes:
        key = self._pack_string(self.key, KEY_LENGTH, 'key')
        col_header = self._pack_string(
            self.col_header, COL_HEADER_LENGTH, 'col_header'
        )
        row_header = self._pack_string(
            self.row_header, ROW_HEADER_LENGTH, 'row_header'
        )
        update = (
            self.update_timestamp.timestamp()
            if keep_timestamp
            else datetime.datetime.now(datetime.UTC).timestamp()
        )
        return struct.pack(
            METADATA_FORMAT,
            key,
            col_header,
            row_header,
            self.ordinal,
            self.start,
            self.stop,
            self.periodicity.code.encode(),
            self.type_.name.encode(),
            self.create_timestamp.timestamp(),
            update,
            self.id_.bytes,
        )

    @staticmethod
    def _pack_string(value: str, length: int, name: str) -> bytes:
        encoded = value.encode()
        if len(encoded) > length:
            raise ValueError(
                f'Metadata {name} is too long: {len(encoded)} bytes > {length}'
            )
        return encoded

    @classmethod
    def unpack(cls, bytes_: bytes) -> Metadata:
        if len(bytes_) == METADATA_SIZE:
            metadata_format = METADATA_FORMAT
        elif len(bytes_) == OLD_METADATA_SIZE:
            metadata_format = OLD_METADATA_FORMAT
        else:
            raise ValueError(f'Unsupported metadata size: {len(bytes_)} bytes')
        (
            key,
            col_header,
            row_header,
            ordinal,
            start,
            stop,
            per,
            typ,
            create,
            update,
            id_,
        ) = struct.unpack(metadata_format, bytes_)
        return cls(
            start,
            stop,
            Periodicity[per.decode()],
            DataType[typ.decode()],
            key.decode().strip('\x00'),
            ordinal,
            col_header.decode().strip('\x00'),
            row_header.decode().strip('\x00'),
            uuid.UUID(bytes=id_),
            datetime.datetime.fromtimestamp(create),
            datetime.datetime.fromtimestamp(update),
        )


class Db:
    def __init__(
        self, *, connection: Connection | None = None, **kwargs: Any
    ) -> None:
        self.connection: Connection = connection or RedisConnection(**kwargs)

    def split_key(self, key: str) -> tuple[str, str | None, str | None]:
        pattern = r'([^#]+)(?:#([^$]*))?(?:\$(.*))?'
        m = re.match(pattern, key)
        assert m is not None
        frame: str = m.group(1)
        column: str | None = m.group(2)
        row: str | None = m.group(3)
        return frame, column, row

    def make_safe(self, frame: str, column: str | None, row: str | None) -> str:
        key = struct.pack(f'<{KEY_LENGTH}s', frame.encode())
        if column:
            if column.endswith('*'):
                key += column[:-1].encode()
            else:
                key += struct.pack(f'<{COL_HEADER_LENGTH}s', column.encode())
                if row:
                    if row.endswith('*'):
                        key += row[:-1].encode()
                    else:
                        key += struct.pack(f'<{ROW_HEADER_LENGTH}s', row.encode())
        return key.decode()

    def get_metadata(self, key: str) -> list[Metadata]:
        k = self.make_safe(*self.split_key(key))
        mds = [
            Metadata.unpack(p)
            for p in self.connection.set_members_by_prefix(INDEX, k)
        ]
        if not mds:
            raise KeyError(f"Db key '{key}' not found")
        mds.sort(key=attrgetter('ordinal'))
        return mds

    def columns(self, key: str) -> list[str]:
        return list(dict.fromkeys([md.col_header for md in self.get_metadata(key)]))

    def keys(self, prefix: str | None = None) -> list[str]:
        if prefix:
            all_keys = self.connection.set_members_by_prefix(INDEX, prefix)
        else:
            all_keys = self.connection.set_members(INDEX)
        keys: list[str] = []
        for packed in all_keys:
            k = packed[:KEY_LENGTH].decode().strip('\x00')
            if len(keys) == 0 or k != keys[-1]:
                keys.append(k)
        return keys

    def all_series(self) -> dict[str, list[tuple[str, str]]]:
        mds = [Metadata.unpack(p) for p in self.connection.set_members(INDEX)]
        mds.sort(key=attrgetter('ordinal'))
        keys: dict[str, list[tuple[str, str]]] = {}
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

    def delete_all(self) -> None:
        if self.connection.clear():
            return
        packed_members = self.connection.set_members(INDEX)
        total = len(packed_members)
        batch = self.connection.batch()
        for count, packed in enumerate(packed_members, start=1):
            batch.delete(Metadata.unpack(packed).data_key)
            batch.remove_set_members(INDEX, packed)
            if count % COPY_BATCH_SIZE == 0:
                batch.execute()
                logger.info('Cleared %s/%s pynto series', count, total)
                batch = self.connection.batch()
        batch.execute()
        if total:
            logger.info('Cleared %s/%s pynto series', total, total)

    def __setitem__(self, key: str, pandas: pd.Series | pd.DataFrame) -> None:
        saved: dict[tuple[str, str], Any] = {}
        series: list[tuple[str, str, pd.Series]] = []
        frame, column, row = self.split_key(key)
        assert (
            column is None or isinstance(pandas, pd.Series) or pandas.shape[1] == 1
        ), 'Can only assign one column to a specific column key'
        safe_key = self.make_safe(frame, column, row)
        for packed in self.connection.set_members_by_prefix(INDEX, safe_key):
            md = Metadata.unpack(packed)
            saved[(md.col_header, md.row_header)] = (md, packed)
        if isinstance(pandas, pd.Series):
            series.append((column or pandas.name or '', row or '', pandas))  # type: ignore[arg-type]
        else:
            columns = _check_dups(pandas.columns.values)
            if isinstance(pandas.index, pd.MultiIndex):
                assert len(pandas.index.levels) == 2, 'Too many axes.'
                # rows = _check_dups(pandas.loc[pandas.index[0][0]].index.values, 'Rows')
                rows = _check_dups(
                    np.array(list(pandas.index.get_level_values(1).unique())), 'Rows'
                )
                index = pandas.index.remove_unused_levels().levels[0]
                flat = pandas.values.reshape(len(index), len(columns) * len(rows))
                for i, (row, col) in enumerate(
                    zip(np.repeat(rows, len(columns)), np.tile(columns, len(rows)))
                ):
                    series.append(
                        (str(col), str(row), pd.Series(flat[:, i], index=index))
                    )
            else:
                series.extend([(column or str(h), '', s) for h, s in pandas.items()])
        batch = self.connection.batch()
        toadd, todel = [], []
        for col, row, s in series:
            assert not isinstance(s.dtype, pd.api.extensions.ExtensionDtype)
            type_ = DataType.from_dtype(s.dtype.str)
            md_tuple = saved.get((col, row))
            s_trimmed: pd.Series | None = None
            if not md_tuple:
                s_trimmed = _trim_values(s)
                if s_trimmed is None:
                    continue
                s = s_trimmed
            range_ = Range.from_index(s.index)  # type: ignore[arg-type]
            data = s.to_numpy()
            if not md_tuple:
                data_offset = 0
                series_md = Metadata(
                    range_.start,
                    range_.stop,
                    range_.periodicity,
                    type_,
                    frame,
                    len(saved),
                    col,
                    row,
                )
                saved[(col, row)] = series_md
                toadd.append(series_md)
            else:
                series_md, packed = md_tuple
                assert series_md.periodicity.code == range_.periodicity.code, (
                    f'Periodicity does match saved for {key}'
                )
                assert series_md.type_ == type_, f'Datatype does match saved for {key}'
                assert (
                    series_md.start <= range_.stop and series_md.stop >= range_.start
                ), f'Data not contiguous with saved for {key}'
                if series_md.start > range_.start:
                    if range_.stop < series_md.stop:
                        existing_start = range_.stop - series_md.start
                        data = np.hstack(
                            [
                                data,
                                self.connection.get(series_md.data_key)[
                                    existing_start:
                                ],
                            ]
                        )
                    data_offset = 0
                    series_md.start = range_.start
                else:  # series_md.range_.stop <= range_.start
                    data_offset = range_.start - series_md.start
                series_md.stop = max(range_.stop, series_md.stop)
                toadd.append(series_md)
                todel.append(packed)
            batch.set_range(
                series_md.data_key, data_offset * type_.length, data.tobytes()
            )
        if todel:
            batch.remove_set_members(INDEX, *todel)
        if toadd:
            batch.add_set_members(INDEX, (m.pack() for m in toadd))
        batch.execute()

    def __delitem__(self, key: str) -> None:
        key = self.make_safe(*self.split_key(key))
        p = self.connection.batch()
        for packed in self.connection.set_members_by_prefix(INDEX, key):
            p.delete(Metadata.unpack(packed).data_key)
            p.remove_set_members(INDEX, packed)
        p.execute()

    def __getitem__(self, args: str | tuple[str, slice]) -> pd.DataFrame:
        if isinstance(args, str):
            key = args
            start, stop = None, None
        else:
            key = args[0]
            start, stop = args[1].start, args[1].stop
        safe_key = self.make_safe(*self.split_key(key))
        mds = [
            Metadata.unpack(d)
            for d in self.connection.set_members_by_prefix(INDEX, safe_key)
        ]
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
        ary = np.full((stop - start, len(mds)), mds[0].type_.pad_value, order='F')
        p = self.connection.batch()
        offsets = [self._req(md, start, stop, p) for md in mds]
        for i, md, offset, bytes_ in zip(range(len(mds)), mds, offsets, p.execute()):
            if len(bytes_) > 0:
                data = np.frombuffer(bytes_, md.type_.dtype)
                ary[offset : offset + len(data), i] = data
        cols = np.array([md.col_header for md in mds])
        index = Range(start, stop, per).to_index()
        if mds[0].row_header:
            columns: dict[str, None] = {}  # dict not set preserves order
            rows: dict[str, None] = {}
            for md in mds:
                rows[md.row_header] = None
                columns[md.col_header] = None
            row_index_list = list(rows.keys())
            row_index = pd.CategoricalIndex(
                row_index_list, categories=row_index_list, ordered=True
            )
            index = pd.MultiIndex.from_product([index, row_index])  # type: ignore[assignment]
            # index = pd.MultiIndex.from_product([index, list(rows.keys())])
            ary = ary.reshape(ary.shape[0] * len(rows), len(columns))
            cols = np.array(columns.keys())
        df = pd.DataFrame(ary, columns=cols, index=index)
        return df

    def _req(
        self, saved: Metadata, start: int, stop: int, p: Batch
    ) -> int:
        offset: int
        if start < saved.stop and stop >= saved.start:
            offset = max(0, saved.start - start)
            start = max(start, saved.start)
            stop = min(stop, saved.stop)
            p.get_range(
                saved.data_key,
                (start - saved.start) * saved.type_.length,
                (stop - saved.start) * saved.type_.length - 1,
            )
        else:  # no overlap with saved
            offset = -1
            p.get_range(saved.data_key, -1, 0)
        return offset
