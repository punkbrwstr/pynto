from __future__ import annotations

import base64
import importlib
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote

import redis
from boto3.dynamodb.conditions import Key  # type: ignore[import-untyped]
from redis.connection import UnixDomainSocketConnection

logger = logging.getLogger(__name__)

DYNAMODB_CHUNK_SIZE = 64 * 1024
DYNAMODB_DEFAULT_DATA_TABLE = 'pynto-data'
DYNAMODB_DEFAULT_SET_TABLE = 'pynto-set'
DYNAMODB_MAX_WORKERS = 16
SQLITE_DEFAULT_PATH = '.pynto/pynto.sqlite3'
S3_BUCKET_ENV_VAR = 'PYNTO_S3_BUCKET'
S3_DATA_PREFIX = 'obj/'
S3_HASH_PREFIX = 'hash/'
S3_SET_PREFIX = 'set/'
S3_MAX_KEY_LENGTH = 1024
S3_MULTIPART_MIN_SIZE = 5 * 1024 * 1024
S3_MAX_DELETE_OBJECTS = 1000
S3_MAX_WORKERS = 16
DYNAMODB_DEFAULT_HASH_TABLE = 'pynto-hash'


class UnsupportedConnectionFeature(NotImplementedError):
    pass


def _to_bytes(value: bytes | str | int | float) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode()


def _to_text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _lex_bytes(value: bytes | str) -> bytes:
    text = _to_text(value)
    if text in ('+', '-'):
        raise ValueError(f'Open lexical bound {text!r} is not supported')
    return text[1:].encode() if text[:1] in ('[', '(') else text.encode()


def _lex_inclusive(value: bytes | str) -> bool:
    return not _to_text(value).startswith('(')


def _encode_bytes(value: bytes) -> str:
    return base64.b16encode(value).decode('ascii').lower()


def _decode_bytes(value: str) -> bytes:
    return base64.b16decode(value.upper().encode('ascii'))


def _prefix_upper_bound(prefix: bytes) -> bytes | None:
    for i in range(len(prefix) - 1, -1, -1):
        if prefix[i] != 0xFF:
            return prefix[:i] + bytes([prefix[i] + 1])
    return None


def _encode_key(key: bytes) -> str:
    return _encode_bytes(key)


def _encode_set_key(set_key: str) -> str:
    return quote(set_key, safe='')


def _decode_set_key(set_key: str) -> str:
    return unquote(set_key)


def _is_s3_error(error: Exception, *codes: str) -> bool:
    response = getattr(error, 'response', {})
    code = response.get('Error', {}).get('Code')
    return code in codes


def _s3_bucket_from_env() -> str:
    return os.environ[S3_BUCKET_ENV_VAR]


class Batch(ABC):
    @abstractmethod
    def delete(self, key: bytes) -> None:
        pass

    @abstractmethod
    def remove_set_members(self, set_key: str, *members: bytes) -> None:
        pass

    @abstractmethod
    def add_set_members(self, set_key: str, members: Iterable[bytes]) -> None:
        pass

    @abstractmethod
    def set_range(self, key: bytes, offset: int, value: bytes) -> None:
        pass

    @abstractmethod
    def get_range(self, key: bytes, start: int, stop: int) -> None:
        pass

    @abstractmethod
    def execute(self) -> list[Any]:
        pass

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        raise UnsupportedConnectionFeature('Hash writes are not supported')

    def hmget(self, key: str, fields: Iterable[str]) -> None:
        raise UnsupportedConnectionFeature('Hash reads are not supported')

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> None:
        raise UnsupportedConnectionFeature('Lexical set reads are not supported')


class Connection(ABC):
    @abstractmethod
    def set_members(self, set_key: str) -> list[bytes]:
        pass

    @abstractmethod
    def set_members_by_prefix(self, set_key: str, prefix: str) -> list[bytes]:
        pass

    @abstractmethod
    def get(self, key: bytes) -> bytes:
        pass

    @abstractmethod
    def batch(self) -> Batch:
        pass

    def clear(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def pipeline(self) -> Batch:
        return self.batch()

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        raise UnsupportedConnectionFeature('Hash writes are not supported')

    def hmget(self, key: str, fields: Iterable[str]) -> list[bytes | None]:
        raise UnsupportedConnectionFeature('Hash reads are not supported')

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        raise UnsupportedConnectionFeature('Hash reads are not supported')

    def hash_keys(self, prefix: str = '') -> list[str]:
        raise UnsupportedConnectionFeature('Hash key listing is not supported')

    def zadd(self, set_key: str, mapping: dict[str, float]) -> None:
        raise UnsupportedConnectionFeature('Lexical set writes are not supported')

    def zrem(self, set_key: str, *members: str) -> None:
        raise UnsupportedConnectionFeature('Lexical set writes are not supported')

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> list[bytes]:
        raise UnsupportedConnectionFeature('Lexical set reads are not supported')

    def lex_set_keys(self, prefix: str = '') -> list[str]:
        raise UnsupportedConnectionFeature('Lexical set key listing is not supported')

    def delete(self, key: bytes | str) -> None:
        raise UnsupportedConnectionFeature('Key deletion is not supported')

    def incr(self, key: str) -> int:
        raise UnsupportedConnectionFeature('Counters are not supported')

    def publish(self, channel: str, message: str) -> int:
        raise UnsupportedConnectionFeature('Pub/sub is only supported by RedisConnection')

    def pubsub(self, **kwargs: Any) -> Any:
        raise UnsupportedConnectionFeature('Pub/sub is only supported by RedisConnection')


class RedisBatch(Batch):
    def __init__(self, pipeline: redis.client.Pipeline) -> None:
        self._pipeline = pipeline

    def delete(self, key: bytes) -> None:
        self._pipeline.delete(key)

    def remove_set_members(self, set_key: str, *members: bytes) -> None:
        self._pipeline.zrem(set_key, *members)

    def add_set_members(self, set_key: str, members: Iterable[bytes]) -> None:
        self._pipeline.zadd(set_key, {member: 0.0 for member in members})

    def set_range(self, key: bytes, offset: int, value: bytes) -> None:
        self._pipeline.setrange(key, offset, value)

    def get_range(self, key: bytes, start: int, stop: int) -> None:
        self._pipeline.getrange(key, start, stop)

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        self._pipeline.hset(key, mapping=cast(Any, mapping))

    def hmget(self, key: str, fields: Iterable[str]) -> None:
        self._pipeline.hmget(key, list(fields))

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> None:
        self._pipeline.zrevrangebylex(set_key, max_, min_, start, num)

    def execute(self) -> list[Any]:
        result: list[Any] = self._pipeline.execute()
        return result


class RedisConnection(Connection):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault('protocol', 2)
        if 'path' in kwargs:
            kwargs['connection_class'] = UnixDomainSocketConnection
        self._pool = redis.ConnectionPool(**kwargs)

    @property
    def _client(self) -> redis.Redis:
        return redis.Redis(connection_pool=self._pool)

    def set_members(self, set_key: str) -> list[bytes]:
        return cast(list[bytes], self._client.zrange(set_key, 0, -1))

    def set_members_by_prefix(self, set_key: str, prefix: str) -> list[bytes]:
        return cast(
            list[bytes],
            self._client.zrangebylex(set_key, f'[{prefix}', f'[{prefix}\xff'),
        )

    def get(self, key: bytes) -> bytes:
        return cast(bytes, self._client.get(key))

    def batch(self) -> Batch:
        return RedisBatch(self._client.pipeline())

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        self._client.hset(key, mapping=cast(Any, mapping))

    def hmget(self, key: str, fields: Iterable[str]) -> list[bytes | None]:
        return cast(list[bytes | None], self._client.hmget(key, list(fields)))

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        return cast(dict[bytes, bytes], self._client.hgetall(key))

    def hash_keys(self, prefix: str = '') -> list[str]:
        return [
            key.decode()
            for key in self._client.scan_iter(match=f'{prefix}*', _type='hash')
        ]

    def zadd(self, set_key: str, mapping: dict[str, float]) -> None:
        self._client.zadd(set_key, mapping)

    def zrem(self, set_key: str, *members: str) -> None:
        self._client.zrem(set_key, *members)

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> list[bytes]:
        return cast(
            list[bytes],
            self._client.zrevrangebylex(set_key, max_, min_, start, num),
        )

    def lex_set_keys(self, prefix: str = '') -> list[str]:
        return [
            key.decode()
            for key in self._client.scan_iter(match=f'{prefix}*', _type='zset')
        ]

    def delete(self, key: bytes | str) -> None:
        self._client.delete(key)

    def incr(self, key: str) -> int:
        return self._client.incr(key)

    def publish(self, channel: str, message: str) -> int:
        return self._client.publish(channel, message)

    def pubsub(self, **kwargs: Any) -> Any:
        return self._client.pubsub(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class SQLiteBatch(Batch):
    def __init__(self, connection: SQLiteConnection) -> None:
        self._connection = connection
        self._operations: list[tuple[str, tuple[Any, ...]]] = []

    def delete(self, key: bytes) -> None:
        self._operations.append(('delete', (key,)))

    def remove_set_members(self, set_key: str, *members: bytes) -> None:
        self._operations.append(('remove_set_members', (set_key, members)))

    def add_set_members(self, set_key: str, members: Iterable[bytes]) -> None:
        self._operations.append(('add_set_members', (set_key, list(members))))

    def set_range(self, key: bytes, offset: int, value: bytes) -> None:
        self._operations.append(('set_range', (key, offset, value)))

    def get_range(self, key: bytes, start: int, stop: int) -> None:
        self._operations.append(('get_range', (key, start, stop)))

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        self._operations.append(('hmset', (key, mapping)))

    def hmget(self, key: str, fields: Iterable[str]) -> None:
        self._operations.append(('hmget', (key, list(fields))))

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> None:
        self._operations.append(('zrevrangebylex', (set_key, max_, min_, start, num)))

    def execute(self) -> list[Any]:
        results: list[Any] = []
        with self._connection._db:
            for operation, args in self._operations:
                result: Any = None
                if operation == 'delete':
                    self._connection._delete_now(*args)
                elif operation == 'remove_set_members':
                    self._connection._remove_set_members_now(*args)
                elif operation == 'add_set_members':
                    self._connection._add_set_members_now(*args)
                elif operation == 'set_range':
                    self._connection._set_range_now(*args)
                elif operation == 'get_range':
                    result = self._connection._get_range_now(*args)
                elif operation == 'hmset':
                    self._connection.hmset(*args)
                elif operation == 'hmget':
                    result = self._connection.hmget(*args)
                elif operation == 'zrevrangebylex':
                    result = self._connection.zrevrangebylex(*args)
                else:
                    raise ValueError(f'Unknown SQLite batch operation: {operation}')
                results.append(result)
        return results


class SQLiteConnection(Connection):
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path or os.environ.get('PYNTO_SQLITE_PATH', SQLITE_DEFAULT_PATH))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.execute('PRAGMA synchronous=NORMAL')
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if self._has_old_schema():
            logger.info('Dropping old SQLite pynto schema at %s', self.path)
            self._db.execute('DROP TABLE IF EXISTS data')
            self._db.execute('DROP TABLE IF EXISTS set_members')
        self._db.execute(
            'CREATE TABLE IF NOT EXISTS data (key BLOB PRIMARY KEY, value BLOB NOT NULL)'
        )
        self._db.execute(
            'CREATE TABLE IF NOT EXISTS set_members ('
            'set_key TEXT NOT NULL, '
            'member BLOB NOT NULL, '
            'PRIMARY KEY (set_key, member)'
            ')'
        )
        self._db.execute(
            'CREATE TABLE IF NOT EXISTS hash_values ('
            'hash_key TEXT NOT NULL, '
            'field TEXT NOT NULL, '
            'value BLOB NOT NULL, '
            'PRIMARY KEY (hash_key, field)'
            ')'
        )
        self._db.commit()

    def _has_old_schema(self) -> bool:
        data_columns = {
            row[1]: row[2].upper()
            for row in self._db.execute("PRAGMA table_info('data')")
        }
        set_columns = {
            row[1]: row[2].upper()
            for row in self._db.execute("PRAGMA table_info('set_members')")
        }
        return (
            data_columns.get('key') == 'TEXT'
            or 'member_key' in set_columns
            or set_columns.get('member') == 'TEXT'
        )

    def set_members(self, set_key: str) -> list[bytes]:
        return [
            row[0]
            for row in self._db.execute(
                'SELECT member FROM set_members WHERE set_key = ? ORDER BY member',
                (set_key,),
            )
        ]

    def set_members_by_prefix(self, set_key: str, prefix: str) -> list[bytes]:
        prefix_key = prefix.encode()
        upper_key = _prefix_upper_bound(prefix_key)
        query = (
            'SELECT member FROM set_members '
            'WHERE set_key = ? AND member >= ? '
        )
        params: tuple[Any, ...]
        if upper_key is None:
            params = (set_key, prefix_key)
        else:
            query += 'AND member < ? '
            params = (set_key, prefix_key, upper_key)
        query += 'ORDER BY member'
        return [
            row[0]
            for row in self._db.execute(query, params)
        ]

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        with self._db:
            self._db.executemany(
                'INSERT OR REPLACE INTO hash_values (hash_key, field, value) '
                'VALUES (?, ?, ?)',
                ((key, field, _to_bytes(value)) for field, value in mapping.items()),
            )

    def hmget(self, key: str, fields: Iterable[str]) -> list[bytes | None]:
        return [
            self._hget(key, field)
            for field in fields
        ]

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        return {
            field.encode(): value
            for field, value in self._db.execute(
                'SELECT field, value FROM hash_values WHERE hash_key = ?',
                (key,),
            )
        }

    def hash_keys(self, prefix: str = '') -> list[str]:
        upper = _prefix_upper_bound(prefix.encode())
        if upper is None:
            query = 'SELECT DISTINCT hash_key FROM hash_values WHERE hash_key >= ?'
            params: tuple[Any, ...] = (prefix,)
        else:
            query = (
                'SELECT DISTINCT hash_key FROM hash_values '
                'WHERE hash_key >= ? AND hash_key < ?'
            )
            params = (prefix, upper.decode())
        query += ' ORDER BY hash_key'
        return [row[0] for row in self._db.execute(query, params)]

    def zadd(self, set_key: str, mapping: dict[str, float]) -> None:
        with self._db:
            self._add_set_members_now(
                set_key, [_to_bytes(member) for member in mapping]
            )

    def zrem(self, set_key: str, *members: str) -> None:
        with self._db:
            self._remove_set_members_now(
                set_key, tuple(_to_bytes(member) for member in members)
            )

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> list[bytes]:
        rows = self._lex_members(set_key, max_, min_, reverse=True)
        start = start or 0
        return rows[start : None if num is None else start + num]

    def lex_set_keys(self, prefix: str = '') -> list[str]:
        upper = _prefix_upper_bound(prefix.encode())
        if upper is None:
            query = 'SELECT DISTINCT set_key FROM set_members WHERE set_key >= ?'
            params: tuple[Any, ...] = (prefix,)
        else:
            query = (
                'SELECT DISTINCT set_key FROM set_members '
                'WHERE set_key >= ? AND set_key < ?'
            )
            params = (prefix, upper.decode())
        query += ' ORDER BY set_key'
        return [row[0] for row in self._db.execute(query, params)]

    def delete(self, key: bytes | str) -> None:
        with self._db:
            if isinstance(key, bytes):
                self._delete_now(key)
            else:
                self._db.execute('DELETE FROM hash_values WHERE hash_key = ?', (key,))
                self._db.execute('DELETE FROM set_members WHERE set_key = ?', (key,))

    def _hget(self, key: str, field: str) -> bytes | None:
        row = self._db.execute(
            'SELECT value FROM hash_values WHERE hash_key = ? AND field = ?',
            (key, field),
        ).fetchone()
        return row[0] if row else None

    def _lex_members(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        *,
        reverse: bool = False,
    ) -> list[bytes]:
        max_value = _lex_bytes(max_)
        min_value = _lex_bytes(min_)
        max_op = '<=' if _lex_inclusive(max_) else '<'
        min_op = '>=' if _lex_inclusive(min_) else '>'
        direction = 'DESC' if reverse else 'ASC'
        return [
            row[0]
            for row in self._db.execute(
                f'SELECT member FROM set_members WHERE set_key = ? '
                f'AND member {min_op} ? AND member {max_op} ? '
                f'ORDER BY member {direction}',
                (set_key, min_value, max_value),
            )
        ]

    def get(self, key: bytes) -> bytes:
        row = self._db.execute(
            'SELECT value FROM data WHERE key = ?', (key,)
        ).fetchone()
        return row[0] if row else b''

    def batch(self) -> Batch:
        return SQLiteBatch(self)

    def clear(self) -> bool:
        with self._db:
            self._db.execute('DELETE FROM data')
            self._db.execute('DELETE FROM set_members')
            self._db.execute('DELETE FROM hash_values')
        return True

    def close(self) -> None:
        self._db.close()

    def _delete_now(self, key: bytes) -> None:
        self._db.execute('DELETE FROM data WHERE key = ?', (key,))

    def _remove_set_members_now(self, set_key: str, members: tuple[bytes, ...]) -> None:
        self._db.executemany(
            'DELETE FROM set_members WHERE set_key = ? AND member = ?',
            ((set_key, member) for member in members),
        )

    def _add_set_members_now(self, set_key: str, members: list[bytes]) -> None:
        self._db.executemany(
            'INSERT OR REPLACE INTO set_members (set_key, member) VALUES (?, ?)',
            ((set_key, member) for member in members),
        )

    def _set_range_now(self, key: bytes, offset: int, value: bytes) -> None:
        if offset < 0:
            raise ValueError('SQLite set_range offset must be non-negative')
        if not value:
            return
        row = self._db.execute(
            'SELECT length(value) FROM data WHERE key = ?', (key,)
        ).fetchone()
        size = row[0] if row else 0
        end = offset + len(value)
        if row is None:
            self._db.execute(
                'INSERT INTO data (key, value) VALUES (?, zeroblob(?))',
                (key, end),
            )
        elif end > size:
            self._db.execute(
                'UPDATE data SET value = CAST(value || zeroblob(?) AS BLOB) '
                'WHERE key = ?',
                (end - size, key),
            )
        rowid = self._db.execute(
            'SELECT rowid FROM data WHERE key = ?', (key,)
        ).fetchone()[0]
        blob = self._db.blobopen('data', 'value', rowid, readonly=False)
        try:
            blob.seek(offset)
            blob.write(value)
        finally:
            blob.close()

    def _get_range_now(self, key: bytes, start: int, stop: int) -> bytes:
        if start < 0 or stop < start:
            return b''
        row = self._db.execute(
            'SELECT substr(value, ?, ?) FROM data WHERE key = ?',
            (start + 1, stop - start + 1, key),
        ).fetchone()
        return row[0] if row else b''


class DynamoDBBatch(Batch):
    def __init__(self, connection: DynamoDBConnection) -> None:
        self._connection = connection
        self._operations: list[tuple[str, tuple[Any, ...]]] = []

    def delete(self, key: bytes) -> None:
        self._operations.append(('delete', (key,)))

    def remove_set_members(self, set_key: str, *members: bytes) -> None:
        self._operations.append(('remove_set_members', (set_key, members)))

    def add_set_members(self, set_key: str, members: Iterable[bytes]) -> None:
        self._operations.append(('add_set_members', (set_key, list(members))))

    def set_range(self, key: bytes, offset: int, value: bytes) -> None:
        self._operations.append(('set_range', (key, offset, value)))

    def get_range(self, key: bytes, start: int, stop: int) -> None:
        self._operations.append(('get_range', (key, start, stop)))

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        self._operations.append(('hmset', (key, mapping)))

    def hmget(self, key: str, fields: Iterable[str]) -> None:
        self._operations.append(('hmget', (key, list(fields))))

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> None:
        self._operations.append(('zrevrangebylex', (set_key, max_, min_, start, num)))

    def execute(self) -> list[Any]:
        if all(operation == 'get_range' for operation, _ in self._operations):
            with ThreadPoolExecutor(max_workers=DYNAMODB_MAX_WORKERS) as executor:
                return list(
                    executor.map(
                        lambda args: self._connection._get_range_now(*args),
                        (args for _, args in self._operations),
                    )
                )
        results: list[Any] = []
        i = 0
        while i < len(self._operations):
            operation, args = self._operations[i]
            result: Any = None
            if operation == 'set_range':
                set_ranges: list[tuple[bytes, int, bytes]] = []
                while (
                    i < len(self._operations)
                    and self._operations[i][0] == 'set_range'
                ):
                    set_ranges.append(cast(tuple[bytes, int, bytes], self._operations[i][1]))
                    i += 1
                self._connection._set_ranges_now(set_ranges)
                results.extend([None] * len(set_ranges))
                continue
            elif operation == 'delete':
                self._connection._delete_now(*args)
            elif operation == 'remove_set_members':
                self._connection._remove_set_members_now(*args)
            elif operation == 'add_set_members':
                self._connection._add_set_members_now(*args)
            elif operation == 'get_range':
                result = self._connection._get_range_now(*args)
            elif operation == 'hmset':
                self._connection.hmset(*args)
            elif operation == 'hmget':
                result = self._connection.hmget(*args)
            elif operation == 'zrevrangebylex':
                result = self._connection.zrevrangebylex(*args)
            else:
                raise ValueError(f'Unknown DynamoDB batch operation: {operation}')
            results.append(result)
            i += 1
        return results


class DynamoDBConnection(Connection):
    def __init__(
        self,
        *,
        data_table: str | None = None,
        set_table: str | None = None,
        hash_table: str | None = None,
        resource: Any | None = None,
        create_tables: bool = True,
    ) -> None:
        self.data_table_name = data_table or os.environ.get(
            'PYNTO_DYNAMODB_DATA_TABLE', DYNAMODB_DEFAULT_DATA_TABLE
        )
        self.set_table_name = set_table or os.environ.get(
            'PYNTO_DYNAMODB_SET_TABLE', DYNAMODB_DEFAULT_SET_TABLE
        )
        self.hash_table_name = hash_table or os.environ.get(
            'PYNTO_DYNAMODB_HASH_TABLE', DYNAMODB_DEFAULT_HASH_TABLE
        )
        self._dynamodb = resource or self._make_resource()
        if create_tables:
            self._ensure_tables()
        self._data = self._dynamodb.Table(self.data_table_name)
        self._sets = self._dynamodb.Table(self.set_table_name)
        self._hashes = self._dynamodb.Table(self.hash_table_name)

    def set_members(self, set_key: str) -> list[bytes]:
        items = self._query_set(Key('set_key').eq(set_key))
        return [bytes(item['member']) for item in items]

    def set_members_by_prefix(self, set_key: str, prefix: str) -> list[bytes]:
        condition = Key('set_key').eq(set_key) & Key('member').begins_with(
            prefix.encode()
        )
        return [bytes(item['member']) for item in self._query_set(condition)]

    def get(self, key: bytes) -> bytes:
        return b''.join(
            bytes(item['value'])
            for item in self._query_data(Key('key').eq(key))
        )

    def batch(self) -> Batch:
        return DynamoDBBatch(self)

    def _make_resource(self) -> Any:
        boto3 = importlib.import_module('boto3')
        config_module = importlib.import_module('botocore.config')
        config = config_module.Config(max_pool_connections=DYNAMODB_MAX_WORKERS)
        return boto3.resource('dynamodb', config=config)

    def _ensure_tables(self) -> None:
        existing = self._existing_table_names()
        if self.data_table_name not in existing:
            self._create_data_table()
        if self.set_table_name not in existing:
            self._create_set_table()
        if self.hash_table_name not in existing:
            self._create_hash_table()
        for table_name in (
            self.data_table_name,
            self.set_table_name,
            self.hash_table_name,
        ):
            waiter = self._dynamodb.meta.client.get_waiter('table_exists')
            waiter.wait(TableName=table_name)

    def clear(self) -> bool:
        client = self._dynamodb.meta.client
        existing = self._existing_table_names()
        for table_name in (
            self.data_table_name,
            self.set_table_name,
            self.hash_table_name,
        ):
            if table_name not in existing:
                continue
            logger.info('Deleting DynamoDB table %s', table_name)
            self._dynamodb.Table(table_name).delete()
            client.get_waiter('table_not_exists').wait(TableName=table_name)
            logger.info('Deleted DynamoDB table %s', table_name)
        logger.info('Recreating DynamoDB pynto tables')
        self._ensure_tables()
        self._data = self._dynamodb.Table(self.data_table_name)
        self._sets = self._dynamodb.Table(self.set_table_name)
        self._hashes = self._dynamodb.Table(self.hash_table_name)
        logger.info('DynamoDB pynto tables are ready')
        return True

    def _existing_table_names(self) -> set[str]:
        names: set[str] = set()
        paginator = self._dynamodb.meta.client.get_paginator('list_tables')
        for page in paginator.paginate():
            names.update(page.get('TableNames', []))
        return names

    def _create_data_table(self) -> None:
        self._dynamodb.create_table(
            TableName=self.data_table_name,
            KeySchema=[
                {'AttributeName': 'key', 'KeyType': 'HASH'},
                {'AttributeName': 'chunk', 'KeyType': 'RANGE'},
            ],
            AttributeDefinitions=[
                {'AttributeName': 'key', 'AttributeType': 'B'},
                {'AttributeName': 'chunk', 'AttributeType': 'N'},
            ],
            BillingMode='PAY_PER_REQUEST',
        )

    def _create_set_table(self) -> None:
        self._dynamodb.create_table(
            TableName=self.set_table_name,
            KeySchema=[
                {'AttributeName': 'set_key', 'KeyType': 'HASH'},
                {'AttributeName': 'member', 'KeyType': 'RANGE'},
            ],
            AttributeDefinitions=[
                {'AttributeName': 'set_key', 'AttributeType': 'S'},
                {'AttributeName': 'member', 'AttributeType': 'B'},
            ],
            BillingMode='PAY_PER_REQUEST',
        )

    def _create_hash_table(self) -> None:
        self._dynamodb.create_table(
            TableName=self.hash_table_name,
            KeySchema=[
                {'AttributeName': 'hash_key', 'KeyType': 'HASH'},
                {'AttributeName': 'field', 'KeyType': 'RANGE'},
            ],
            AttributeDefinitions=[
                {'AttributeName': 'hash_key', 'AttributeType': 'S'},
                {'AttributeName': 'field', 'AttributeType': 'S'},
            ],
            BillingMode='PAY_PER_REQUEST',
        )

    def _query_set(self, condition: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs = {'KeyConditionExpression': condition}
        while True:
            response = self._sets.query(**kwargs)
            items.extend(response.get('Items', []))
            key = response.get('LastEvaluatedKey')
            if not key:
                break
            kwargs['ExclusiveStartKey'] = key
        return items

    def _query_data(self, condition: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs = {'KeyConditionExpression': condition}
        while True:
            response = self._data.query(**kwargs)
            items.extend(response.get('Items', []))
            key = response.get('LastEvaluatedKey')
            if not key:
                break
            kwargs['ExclusiveStartKey'] = key
        return items

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        with self._hashes.batch_writer(
            overwrite_by_pkeys=['hash_key', 'field']
        ) as writer:
            for field, value in mapping.items():
                writer.put_item(
                    Item={
                        'hash_key': key,
                        'field': field,
                        'value': _to_bytes(value),
                    }
                )

    def hmget(self, key: str, fields: Iterable[str]) -> list[bytes | None]:
        return [self._hget(key, field) for field in fields]

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        result: dict[bytes, bytes] = {}
        kwargs: dict[str, Any] = {
            'KeyConditionExpression': Key('hash_key').eq(key),
        }
        while True:
            response = self._hashes.query(**kwargs)
            for item in response.get('Items', []):
                result[item['field'].encode()] = bytes(item['value'])
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                return result
            kwargs['ExclusiveStartKey'] = last_key

    def hash_keys(self, prefix: str = '') -> list[str]:
        return self._scan_string_keys(
            self._hashes,
            'hash_key',
            prefix,
            ('hash_key', 'field'),
        )

    def zadd(self, set_key: str, mapping: dict[str, float]) -> None:
        self._add_set_members_now(set_key, [_to_bytes(member) for member in mapping])

    def zrem(self, set_key: str, *members: str) -> None:
        self._remove_set_members_now(
            set_key, tuple(_to_bytes(member) for member in members)
        )

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> list[bytes]:
        condition = self._lex_condition(set_key, max_, min_)
        kwargs: dict[str, Any] = {
            'KeyConditionExpression': condition,
            'ScanIndexForward': False,
        }
        if (start is None or start == 0) and num is not None:
            kwargs['Limit'] = num
        rows: list[bytes] = []
        while True:
            response = self._sets.query(**kwargs)
            rows.extend(bytes(item['member']) for item in response.get('Items', []))
            if num is not None and (start is None or start == 0) and len(rows) >= num:
                return rows[:num]
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
            kwargs['ExclusiveStartKey'] = last_key
        start = start or 0
        return rows[start : None if num is None else start + num]

    def lex_set_keys(self, prefix: str = '') -> list[str]:
        return self._scan_string_keys(
            self._sets,
            'set_key',
            prefix,
            ('set_key', 'member'),
        )

    def delete(self, key: bytes | str) -> None:
        if isinstance(key, bytes):
            self._delete_now(key)
            return
        with self._hashes.batch_writer() as writer:
            for field in self.hgetall(key):
                writer.delete_item(Key={'hash_key': key, 'field': field.decode()})
        with self._sets.batch_writer() as writer:
            for member in self.set_members(key):
                writer.delete_item(Key={'set_key': key, 'member': member})

    def _hget(self, key: str, field: str) -> bytes | None:
        response = self._hashes.get_item(Key={'hash_key': key, 'field': field})
        item = response.get('Item')
        return bytes(item['value']) if item else None

    def _scan_string_keys(
        self,
        table: Any,
        attribute: str,
        prefix: str,
        key_attributes: tuple[str, str],
    ) -> list[str]:
        keys: set[str] = set()
        kwargs: dict[str, Any] = {
            'ProjectionExpression': attribute,
        }
        while True:
            response = table.scan(**kwargs)
            for item in response.get('Items', []):
                value = item[attribute]
                if value.startswith(prefix):
                    keys.add(value)
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
            kwargs['ExclusiveStartKey'] = last_key
        return sorted(keys)

    def _lex_condition(self, set_key: str, max_: bytes | str, min_: bytes | str) -> Any:
        condition = Key('set_key').eq(set_key)
        max_value = _lex_bytes(max_)
        min_value = _lex_bytes(min_)
        if not _lex_inclusive(min_):
            min_value += b'\x00'
        if not _lex_inclusive(max_):
            raise UnsupportedConnectionFeature(
                'DynamoDB lexical ranges do not support exclusive upper bounds'
            )
        return condition & Key('member').between(min_value, max_value)

    def _delete_now(self, key: bytes) -> None:
        with self._data.batch_writer() as writer:
            for item in self._query_data(Key('key').eq(key)):
                writer.delete_item(Key={'key': key, 'chunk': item['chunk']})

    def _remove_set_members_now(self, set_key: str, members: tuple[bytes, ...]) -> None:
        with self._sets.batch_writer() as writer:
            for member in members:
                writer.delete_item(
                    Key={'set_key': set_key, 'member': member}
                )

    def _add_set_members_now(self, set_key: str, members: list[bytes]) -> None:
        with self._sets.batch_writer(overwrite_by_pkeys=['set_key', 'member']) as writer:
            for member in members:
                writer.put_item(
                    Item={
                        'set_key': set_key,
                        'member': member,
                    }
                )

    def _set_range_now(self, key: bytes, offset: int, value: bytes) -> None:
        self._set_ranges_now([(key, offset, value)])

    def _set_ranges_now(self, ranges: list[tuple[bytes, int, bytes]]) -> None:
        if not ranges:
            return
        existing_chunks = self._get_existing_chunks(ranges)
        with self._data.batch_writer(overwrite_by_pkeys=['key', 'chunk']) as writer:
            for key, offset, value in ranges:
                self._write_range(writer, key, offset, value, existing_chunks)

    def _write_range(
        self,
        writer: Any,
        key: bytes,
        offset: int,
        value: bytes,
        existing_chunks: dict[tuple[bytes, int], bytes] | None = None,
    ) -> None:
        if offset < 0:
            raise ValueError('DynamoDB set_range offset must be non-negative')
        if not value:
            return
        start_chunk = offset // DYNAMODB_CHUNK_SIZE
        end = offset + len(value)
        end_chunk = (end - 1) // DYNAMODB_CHUNK_SIZE
        pos = 0
        for chunk in range(start_chunk, end_chunk + 1):
            chunk_start = chunk * DYNAMODB_CHUNK_SIZE
            write_start = max(offset, chunk_start)
            write_end = min(end, chunk_start + DYNAMODB_CHUNK_SIZE)
            value_start = write_start - offset
            value_end = value_start + (write_end - write_start)
            existing = (
                existing_chunks.get((key, chunk), b'')
                if existing_chunks is not None
                else self._get_chunk(key, chunk)
            )
            local_start = write_start - chunk_start
            local_end = local_start + (write_end - write_start)
            size = max(len(existing), local_end)
            merged = bytearray(size)
            merged[: len(existing)] = existing
            merged[local_start:local_end] = value[value_start:value_end]
            writer.put_item(Item={'key': key, 'chunk': chunk, 'value': bytes(merged)})
            pos += value_end - value_start
        assert pos == len(value)

    def _get_existing_chunks(
        self, ranges: list[tuple[bytes, int, bytes]]
    ) -> dict[tuple[bytes, int], bytes]:
        keys: list[dict[str, Any]] = []
        seen: set[tuple[bytes, int]] = set()
        for key, offset, value in ranges:
            if offset < 0:
                raise ValueError('DynamoDB set_range offset must be non-negative')
            if not value:
                continue
            start_chunk = offset // DYNAMODB_CHUNK_SIZE
            end_chunk = (offset + len(value) - 1) // DYNAMODB_CHUNK_SIZE
            for chunk in range(start_chunk, end_chunk + 1):
                item_key = (key, chunk)
                if item_key not in seen:
                    seen.add(item_key)
                    keys.append({'key': key, 'chunk': chunk})
        existing: dict[tuple[bytes, int], bytes] = {}
        for i in range(0, len(keys), 100):
            request_items = {
                self.data_table_name: {
                    'Keys': keys[i : i + 100],
                    'ProjectionExpression': '#k, chunk, #v',
                    'ExpressionAttributeNames': {'#k': 'key', '#v': 'value'},
                }
            }
            while request_items:
                response = self._dynamodb.batch_get_item(RequestItems=request_items)
                for item in response.get('Responses', {}).get(self.data_table_name, []):
                    existing[(bytes(item['key']), int(item['chunk']))] = bytes(
                        item['value']
                    )
                request_items = response.get('UnprocessedKeys', {})
        return existing

    def _get_range_now(self, key: bytes, start: int, stop: int) -> bytes:
        if start < 0 or stop < start:
            return b''
        start_chunk = start // DYNAMODB_CHUNK_SIZE
        end_chunk = stop // DYNAMODB_CHUNK_SIZE
        condition = Key('key').eq(key) & Key('chunk').between(
            start_chunk, end_chunk
        )
        pieces: list[bytes] = []
        for item in self._query_data(condition):
            chunk = int(item['chunk'])
            data = bytes(item['value'])
            chunk_start = chunk * DYNAMODB_CHUNK_SIZE
            local_start = max(start - chunk_start, 0)
            local_stop = min(stop - chunk_start + 1, len(data))
            if local_start < local_stop:
                pieces.append(data[local_start:local_stop])
        return b''.join(pieces)

    def _get_chunk(self, key: bytes, chunk: int) -> bytes:
        response = self._data.get_item(Key={'key': key, 'chunk': chunk})
        item = response.get('Item')
        return bytes(item['value']) if item else b''


class S3Batch(Batch):
    def __init__(self, connection: S3Connection) -> None:
        self._connection = connection
        self._operations: list[tuple[str, tuple[Any, ...]]] = []

    def delete(self, key: bytes) -> None:
        self._operations.append(('delete', (key,)))

    def remove_set_members(self, set_key: str, *members: bytes) -> None:
        self._operations.append(('remove_set_members', (set_key, members)))

    def add_set_members(self, set_key: str, members: Iterable[bytes]) -> None:
        self._operations.append(('add_set_members', (set_key, list(members))))

    def set_range(self, key: bytes, offset: int, value: bytes) -> None:
        self._operations.append(('set_range', (key, offset, value)))

    def get_range(self, key: bytes, start: int, stop: int) -> None:
        self._operations.append(('get_range', (key, start, stop)))

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        self._operations.append(('hmset', (key, mapping)))

    def hmget(self, key: str, fields: Iterable[str]) -> None:
        self._operations.append(('hmget', (key, list(fields))))

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> None:
        self._operations.append(('zrevrangebylex', (set_key, max_, min_, start, num)))

    def execute(self) -> list[Any]:
        if all(operation == 'get_range' for operation, _ in self._operations):
            with ThreadPoolExecutor(max_workers=S3_MAX_WORKERS) as executor:
                return list(
                    executor.map(
                        lambda args: self._connection._get_range_now(*args),
                        (args for _, args in self._operations),
                    )
                )

        results: list[Any] = []
        for operation, args in self._operations:
            result: Any = None
            if operation == 'delete':
                self._connection._delete_now(*args)
            elif operation == 'remove_set_members':
                self._connection._remove_set_members_now(*args)
            elif operation == 'add_set_members':
                self._connection._add_set_members_now(*args)
            elif operation == 'set_range':
                self._connection._set_range_now(*args)
            elif operation == 'get_range':
                result = self._connection._get_range_now(*args)
            elif operation == 'hmset':
                self._connection.hmset(*args)
            elif operation == 'hmget':
                result = self._connection.hmget(*args)
            elif operation == 'zrevrangebylex':
                result = self._connection.zrevrangebylex(*args)
            else:
                raise ValueError(f'Unknown S3 batch operation: {operation}')
            results.append(result)
        return results


class S3Connection(Connection):
    def __init__(
        self,
        bucket: str | None = None,
        *,
        client: Any | None = None,
        prefix: str = '',
    ) -> None:
        self.bucket = bucket or _s3_bucket_from_env()
        self.prefix = prefix.strip('/')
        self._s3 = client or self._make_client()

    def set_members(self, set_key: str) -> list[bytes]:
        return [
            _decode_bytes(key.rsplit('/', 1)[-1])
            for key in self._list_keys(self._set_prefix(set_key))
        ]

    def set_members_by_prefix(self, set_key: str, prefix: str) -> list[bytes]:
        object_prefix = self._set_prefix(set_key) + _encode_bytes(prefix.encode())
        return [
            _decode_bytes(key.rsplit('/', 1)[-1])
            for key in self._list_keys(object_prefix)
        ]

    def get(self, key: bytes) -> bytes:
        return self._get_object_bytes(self._data_key(key))

    def batch(self) -> Batch:
        return S3Batch(self)

    def hmset(self, key: str, mapping: dict[str, Any]) -> None:
        with ThreadPoolExecutor(max_workers=S3_MAX_WORKERS) as executor:
            list(
                executor.map(
                    lambda item: self._put_object_bytes(
                        self._hash_field_key(key, item[0]), _to_bytes(item[1])
                    ),
                    mapping.items(),
                )
            )

    def hmget(self, key: str, fields: Iterable[str]) -> list[bytes | None]:
        return [
            self._hget(key, field)
            for field in fields
        ]

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        prefix = self._hash_prefix(key)
        return {
            unquote(object_key.rsplit('/', 1)[-1]).encode(): self._get_object_bytes(
                object_key
            )
            for object_key in self._list_keys(prefix)
        }

    def hash_keys(self, prefix: str = '') -> list[str]:
        object_prefix = self._key(f'{S3_HASH_PREFIX}{_encode_set_key(prefix)}')
        trim_prefix = self._key(S3_HASH_PREFIX)
        keys = {
            _decode_set_key(
                object_key[len(trim_prefix) :].split('/', 1)[0]
            )
            for object_key in self._list_keys(object_prefix)
        }
        return sorted(keys)

    def zadd(self, set_key: str, mapping: dict[str, float]) -> None:
        self._add_set_members_now(set_key, [_to_bytes(member) for member in mapping])

    def zrem(self, set_key: str, *members: str) -> None:
        self._remove_set_members_now(
            set_key, tuple(_to_bytes(member) for member in members)
        )

    def zrevrangebylex(
        self,
        set_key: str,
        max_: bytes | str,
        min_: bytes | str,
        start: int | None = None,
        num: int | None = None,
    ) -> list[bytes]:
        members = [
            _decode_bytes(key.rsplit('/', 1)[-1])
            for key in self._list_keys(self._set_prefix(set_key) + self._lex_prefix(max_, min_))
        ]
        min_value = _lex_bytes(min_)
        max_value = _lex_bytes(max_)
        min_ok = (lambda value: value >= min_value) if _lex_inclusive(min_) else (
            lambda value: value > min_value
        )
        max_ok = (lambda value: value <= max_value) if _lex_inclusive(max_) else (
            lambda value: value < max_value
        )
        rows = [member for member in reversed(members) if min_ok(member) and max_ok(member)]
        start = start or 0
        return rows[start : None if num is None else start + num]

    def lex_set_keys(self, prefix: str = '') -> list[str]:
        object_prefix = self._key(f'{S3_SET_PREFIX}{_encode_set_key(prefix)}')
        trim_prefix = self._key(S3_SET_PREFIX)
        keys = {
            _decode_set_key(
                object_key[len(trim_prefix) :].split('/', 1)[0]
            )
            for object_key in self._list_keys(object_prefix)
        }
        return sorted(keys)

    def delete(self, key: bytes | str) -> None:
        if isinstance(key, bytes):
            self._delete_now(key)
            return
        self._delete_objects(self._list_keys(self._hash_prefix(key)))
        self._delete_objects(self._list_keys(self._set_prefix(key)))

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f'{self.__class__.__name__} does not expose native S3 client method '
            f'{name!r}'
        )

    def _make_client(self) -> Any:
        try:
            boto3 = importlib.import_module('boto3')
            config_module = importlib.import_module('botocore.config')
        except ImportError as e:
            raise ImportError('S3Connection requires boto3 to be installed') from e
        config = config_module.Config(max_pool_connections=S3_MAX_WORKERS)
        return boto3.client('s3', config=config)

    def _key(self, key: str) -> str:
        return f'{self.prefix}/{key}' if self.prefix else key

    def _data_key(self, key: bytes) -> str:
        return self._key(f'{S3_DATA_PREFIX}{_encode_bytes(key)}')

    def _hash_prefix(self, key: str) -> str:
        return self._key(f'{S3_HASH_PREFIX}{_encode_set_key(key)}/')

    def _hash_field_key(self, key: str, field: str) -> str:
        return self._hash_prefix(key) + quote(field, safe='')

    def _set_prefix(self, set_key: str) -> str:
        return self._key(f'{S3_SET_PREFIX}{_encode_set_key(set_key)}/')

    def _set_member_key(self, set_key: str, member: bytes) -> str:
        object_key = self._set_prefix(set_key) + _encode_bytes(member)
        if len(object_key.encode()) > S3_MAX_KEY_LENGTH:
            raise ValueError(
                f'S3 set member key is too long for {set_key!r}: '
                f'{len(object_key.encode())} bytes'
            )
        return object_key

    def _hget(self, key: str, field: str) -> bytes | None:
        object_key = self._hash_field_key(key, field)
        try:
            response = self._s3.get_object(Bucket=self.bucket, Key=object_key)
        except Exception as e:
            if _is_s3_error(e, 'NoSuchKey', '404', 'NotFound'):
                return None
            raise
        return cast(bytes, response['Body'].read())

    def _lex_prefix(self, max_: bytes | str, min_: bytes | str) -> str:
        max_value = _lex_bytes(max_)
        min_value = _lex_bytes(min_)
        prefix = os.path.commonprefix([max_value.decode(), min_value.decode()])
        return _encode_bytes(prefix.encode())

    def _list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        kwargs: dict[str, Any] = {
            'Bucket': self.bucket,
            'Prefix': prefix,
            'MaxKeys': 1000,
        }
        while True:
            response = self._s3.list_objects_v2(**kwargs)
            keys.extend(obj['Key'] for obj in response.get('Contents', []))
            token = response.get('NextContinuationToken')
            if not token:
                break
            kwargs['ContinuationToken'] = token
        return keys

    def _get_object_bytes(self, object_key: str) -> bytes:
        try:
            response = self._s3.get_object(Bucket=self.bucket, Key=object_key)
        except Exception as e:
            if _is_s3_error(e, 'NoSuchKey', '404', 'NotFound'):
                return b''
            raise
        return cast(bytes, response['Body'].read())

    def _get_object_size(self, object_key: str) -> int:
        try:
            response = self._s3.head_object(Bucket=self.bucket, Key=object_key)
        except Exception as e:
            if _is_s3_error(e, 'NoSuchKey', '404', 'NotFound'):
                return 0
            raise
        return cast(int, response['ContentLength'])

    def _get_object_range(self, object_key: str, start: int, stop: int) -> bytes:
        if start < 0 or stop < start:
            return b''
        try:
            response = self._s3.get_object(
                Bucket=self.bucket,
                Key=object_key,
                Range=f'bytes={start}-{stop}',
            )
        except Exception as e:
            if _is_s3_error(e, 'NoSuchKey', '404', 'NotFound', 'InvalidRange', '416'):
                return b''
            raise
        return cast(bytes, response['Body'].read())

    def _delete_now(self, key: bytes) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=self._data_key(key))

    def _remove_set_members_now(self, set_key: str, members: tuple[bytes, ...]) -> None:
        object_keys = [self._set_member_key(set_key, member) for member in members]
        self._delete_objects(object_keys)

    def _add_set_members_now(self, set_key: str, members: list[bytes]) -> None:
        with ThreadPoolExecutor(max_workers=S3_MAX_WORKERS) as executor:
            list(
                executor.map(
                    lambda member: self._s3.put_object(
                        Bucket=self.bucket,
                        Key=self._set_member_key(set_key, member),
                        Body=b'',
                    ),
                    members,
                )
            )

    def _set_range_now(self, key: bytes, offset: int, value: bytes) -> None:
        if offset < 0:
            raise ValueError('S3 set_range offset must be non-negative')
        if not value:
            return
        object_key = self._data_key(key)
        size = self._get_object_size(object_key)
        end = offset + len(value)
        if offset == 0 and end >= size:
            self._put_object_bytes(object_key, value)
        elif offset == size and size >= S3_MULTIPART_MIN_SIZE:
            self._append_large_object(object_key, size, value)
        else:
            prefix = self._get_object_range(object_key, 0, offset - 1)
            if offset > size:
                prefix += b'\x00' * (offset - size)
            suffix = self._get_object_range(object_key, end, size - 1)
            self._put_object_bytes(object_key, prefix + value + suffix)

    def _get_range_now(self, key: bytes, start: int, stop: int) -> bytes:
        return self._get_object_range(self._data_key(key), start, stop)

    def _put_object_bytes(self, object_key: str, value: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=object_key, Body=value)

    def _append_large_object(self, object_key: str, size: int, value: bytes) -> None:
        upload = self._s3.create_multipart_upload(
            Bucket=self.bucket,
            Key=object_key,
        )
        upload_id = upload['UploadId']
        parts = []
        try:
            copied = self._s3.upload_part_copy(
                Bucket=self.bucket,
                Key=object_key,
                PartNumber=1,
                UploadId=upload_id,
                CopySource={'Bucket': self.bucket, 'Key': object_key},
                CopySourceRange=f'bytes=0-{size - 1}',
            )
            parts.append(
                {'PartNumber': 1, 'ETag': copied['CopyPartResult']['ETag']}
            )
            uploaded = self._s3.upload_part(
                Bucket=self.bucket,
                Key=object_key,
                PartNumber=2,
                UploadId=upload_id,
                Body=value,
            )
            parts.append({'PartNumber': 2, 'ETag': uploaded['ETag']})
            self._s3.complete_multipart_upload(
                Bucket=self.bucket,
                Key=object_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts},
            )
        except Exception:
            self._s3.abort_multipart_upload(
                Bucket=self.bucket,
                Key=object_key,
                UploadId=upload_id,
            )
            raise

    def _delete_objects(self, keys: list[str]) -> None:
        for i in range(0, len(keys), S3_MAX_DELETE_OBJECTS):
            chunk = keys[i : i + S3_MAX_DELETE_OBJECTS]
            if chunk:
                self._s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': [{'Key': key} for key in chunk]},
                )
