from typing import Any
from .vocabulary import vocab
from .base import Word, Column, toggle_debug
from .periods import Range, Periodicity, datelike
from .database import (
    Db,
    SyncResult,
    copy_data,
    copy_dynamodb_to_redis,
    copy_redis_to_dynamodb,
    copy_redis_to_s3,
    copy_redis_to_sqlite,
    copy_s3_to_redis,
    copy_sqlite_to_redis,
    get_client,
    sync_data,
    sync_dynamodb_to_redis,
    sync_redis_to_dynamodb,
    sync_redis_to_s3,
    sync_redis_to_sqlite,
    sync_s3_to_redis,
    sync_sqlite_to_redis,
)
from .database import use_dynamodb as _database_use_dynamodb
from .database import use_redis as _database_use_redis
from .database import use_s3 as _database_use_s3
from .database import use_sqlite as _database_use_sqlite
from .message_bus import (
    MessageBus,
    get_message_bus,
    use_redis_message_bus,
    use_sqs_message_bus,
)

db = get_client()


def use_redis(**kwargs: Any) -> Db:
    global db
    db = _database_use_redis(**kwargs)
    return db


def use_s3(**kwargs: Any) -> Db:
    global db
    db = _database_use_s3(**kwargs)
    return db


def use_dynamodb(**kwargs: Any) -> Db:
    global db
    db = _database_use_dynamodb(**kwargs)
    return db


def use_sqlite(**kwargs: Any) -> Db:
    global db
    db = _database_use_sqlite(**kwargs)
    return db


class _Definer:
    def __setitem__(self, name: str, word: Word) -> None:
        vocab[name] = ('Ad-hoc', name, lambda n, v, w=word: Word(n, v) + w)


define = _Definer()


def now():
    return Periodicity.B.current()[-1]


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(vocab.keys()))


def __getattr__(name: str) -> Any:
    if not name.startswith('__'):
        word = vocab.resolve(name)
        if word:
            return word
    if name not in globals():
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return globals()[name]


__all__ = [
    # vocabulary exports
    'Word',
    'Column',
    'resolve',
    'toggle_debug',
    'vocab',
    # periods exports
    'Range',
    'Periodicity',
    'datelike',
    # database exports / instances
    'get_client',
    'db',
    'use_redis',
    'use_s3',
    'use_dynamodb',
    'use_sqlite',
    'SyncResult',
    'copy_data',
    'copy_redis_to_s3',
    'copy_s3_to_redis',
    'copy_redis_to_dynamodb',
    'copy_dynamodb_to_redis',
    'copy_redis_to_sqlite',
    'copy_sqlite_to_redis',
    'sync_data',
    'sync_redis_to_s3',
    'sync_s3_to_redis',
    'sync_redis_to_dynamodb',
    'sync_dynamodb_to_redis',
    'sync_redis_to_sqlite',
    'sync_sqlite_to_redis',
    # message bus exports
    'MessageBus',
    'get_message_bus',
    'use_redis_message_bus',
    'use_sqs_message_bus',
    # convenience helpers
    'define',
    'now',
]
