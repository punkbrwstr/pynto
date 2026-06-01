from __future__ import annotations

import json
import os
import socket
import time
import uuid
import atexit
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

from .connection import RedisConnection
from .database import _has_redis_env, _redis_kwargs_from_env

REDIS_REQUEST_CHANNEL_ENV_VAR = 'PYNTO_REDIS_MESSAGE_REQUEST_CHANNEL'
REDIS_RESPONSE_CHANNEL_ENV_VAR = 'PYNTO_REDIS_MESSAGE_RESPONSE_CHANNEL'
REDIS_COUNTER_KEY_ENV_VAR = 'PYNTO_REDIS_MESSAGE_COUNTER_KEY'
REDIS_DEFAULT_REQUEST_CHANNEL = 'pynto:req'
REDIS_DEFAULT_RESPONSE_CHANNEL = 'pynto:res:{}'
REDIS_DEFAULT_COUNTER_KEY = 'pynto:req:id'
SQS_REQUEST_QUEUE_ENV_VAR = 'PYNTO_SQS_REQUEST_QUEUE'
SQS_RESPONSE_QUEUE_PREFIX_ENV_VAR = 'PYNTO_SQS_RESPONSE_QUEUE_PREFIX'
SQS_DEFAULT_REQUEST_QUEUE = 'pynto-requests'
SQS_DEFAULT_RESPONSE_QUEUE_PREFIX = 'pynto-response'
SQS_WAIT_SECONDS = 20
SQS_VISIBILITY_TIMEOUT = 300

Handler = Callable[[str, dict[str, Any]], Any]

_MESSAGE_BUS: MessageBus | None = None


class MessageBus(ABC):
    @abstractmethod
    def request(
        self,
        request_type: str,
        request: dict[str, Any],
        *,
        encoder: type[json.JSONEncoder] | None = None,
        timeout_seconds: float = 180.0,
    ) -> Any:
        pass

    @abstractmethod
    def serve(
        self,
        handler: Handler,
        *,
        encoder: type[json.JSONEncoder] | None = None,
    ) -> None:
        pass


class RedisMessageBus(MessageBus):
    def __init__(
        self,
        connection: RedisConnection | None = None,
        *,
        request_channel: str | None = None,
        response_channel: str | None = None,
        counter_key: str | None = None,
    ) -> None:
        self.connection = connection or RedisConnection(**_redis_kwargs_from_env())
        self.request_channel = request_channel or os.environ.get(
            REDIS_REQUEST_CHANNEL_ENV_VAR, REDIS_DEFAULT_REQUEST_CHANNEL
        )
        self.response_channel = response_channel or os.environ.get(
            REDIS_RESPONSE_CHANNEL_ENV_VAR, REDIS_DEFAULT_RESPONSE_CHANNEL
        )
        self.counter_key = counter_key or os.environ.get(
            REDIS_COUNTER_KEY_ENV_VAR, REDIS_DEFAULT_COUNTER_KEY
        )

    def request(
        self,
        request_type: str,
        request: dict[str, Any],
        *,
        encoder: type[json.JSONEncoder] | None = None,
        timeout_seconds: float = 180.0,
    ) -> Any:
        req_id = self.connection.incr(self.counter_key)
        payload = {'id': req_id, 'type': request_type}
        payload.update(request)
        pubsub = self.connection.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(self.response_channel.format(req_id))
        self.connection.publish(
            self.request_channel, json.dumps(payload, cls=encoder)
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            while time.monotonic() < deadline:
                message = pubsub.get_message()
                if message:
                    response = json.loads(message['data'])
                    if 'error' in response:
                        raise Exception(response['error'])
                    return response['response']
                time.sleep(0.001)
        finally:
            pubsub.close()
        raise IOError(f'Response timeout for request #{req_id}')

    def serve(
        self,
        handler: Handler,
        *,
        encoder: type[json.JSONEncoder] | None = None,
    ) -> None:
        pubsub = self.connection.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(self.request_channel)
        try:
            while True:
                message = pubsub.get_message()
                if message:
                    payload = json.loads(message['data'])
                    req_type = payload.pop('type')
                    req_id = payload.pop('id')
                    try:
                        response = {'response': handler(req_type, payload)}
                    except Exception as e:
                        response = {'error': str(e)}
                    self.connection.publish(
                        self.response_channel.format(req_id),
                        json.dumps(response, cls=encoder),
                    )
                time.sleep(0.001)
        finally:
            pubsub.close()


class SqsMessageBus(MessageBus):
    def __init__(
        self,
        *,
        request_queue_name: str | None = None,
        response_queue_prefix: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.request_queue_name = request_queue_name or os.environ.get(
            SQS_REQUEST_QUEUE_ENV_VAR, SQS_DEFAULT_REQUEST_QUEUE
        )
        self.response_queue_prefix = response_queue_prefix or os.environ.get(
            SQS_RESPONSE_QUEUE_PREFIX_ENV_VAR, SQS_DEFAULT_RESPONSE_QUEUE_PREFIX
        )
        self._sqs = client or self._make_client()
        self.request_queue_url = self._queue_url(self.request_queue_name)
        self.response_queue_url: str | None = None

    def request(
        self,
        request_type: str,
        request: dict[str, Any],
        *,
        encoder: type[json.JSONEncoder] | None = None,
        timeout_seconds: float = 180.0,
    ) -> Any:
        response_queue_url = self._response_queue_url()
        req_id = uuid.uuid4().hex
        payload = {
            'id': req_id,
            'type': request_type,
            'reply_queue_url': response_queue_url,
        }
        payload.update(request)
        self._send_json(self.request_queue_url, payload, encoder)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            response = self._sqs.receive_message(
                QueueUrl=response_queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=min(
                    SQS_WAIT_SECONDS, max(1, int(deadline - time.monotonic()))
                ),
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
            )
            for message in response.get('Messages', []):
                body = json.loads(message['Body'])
                self._sqs.delete_message(
                    QueueUrl=response_queue_url,
                    ReceiptHandle=message['ReceiptHandle'],
                )
                if body.get('id') != req_id:
                    continue
                if 'error' in body:
                    raise Exception(body['error'])
                return body['response']
        raise IOError(f'Response timeout for request #{req_id}')

    def serve(
        self,
        handler: Handler,
        *,
        encoder: type[json.JSONEncoder] | None = None,
    ) -> None:
        while True:
            response = self._sqs.receive_message(
                QueueUrl=self.request_queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=SQS_WAIT_SECONDS,
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
            )
            for message in response.get('Messages', []):
                payload = json.loads(message['Body'])
                req_type = payload.pop('type')
                req_id = payload.pop('id')
                reply_queue_url = payload.pop('reply_queue_url')
                try:
                    result = {'id': req_id, 'response': handler(req_type, payload)}
                except Exception as e:
                    result = {'id': req_id, 'error': str(e)}
                self._send_json(reply_queue_url, result, encoder)
                self._sqs.delete_message(
                    QueueUrl=self.request_queue_url,
                    ReceiptHandle=message['ReceiptHandle'],
                )

    def _make_client(self) -> Any:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]

        return boto3.client('sqs', config=Config(max_pool_connections=16))

    def _queue_url(self, name: str) -> str:
        response = self._sqs.create_queue(
            QueueName=name,
            Attributes={
                'ReceiveMessageWaitTimeSeconds': str(SQS_WAIT_SECONDS),
                'VisibilityTimeout': str(SQS_VISIBILITY_TIMEOUT),
                'MessageRetentionPeriod': str(SQS_VISIBILITY_TIMEOUT),
            },
        )
        return cast(str, response['QueueUrl'])

    def _response_queue_url(self) -> str:
        if self.response_queue_url is None:
            suffix = f'{socket.gethostname()[:12]}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
            self.response_queue_url = self._queue_url(
                f'{self.response_queue_prefix}-{suffix}'[:80]
            )
            atexit.register(self.close)
        return self.response_queue_url

    def close(self) -> None:
        if self.response_queue_url is None:
            return
        try:
            self._sqs.delete_queue(QueueUrl=self.response_queue_url)
        except Exception:
            pass
        self.response_queue_url = None

    def _send_json(
        self,
        queue_url: str,
        payload: dict[str, Any],
        encoder: type[json.JSONEncoder] | None,
    ) -> None:
        self._sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(payload, cls=encoder),
        )


def get_message_bus() -> MessageBus:
    global _MESSAGE_BUS
    if _MESSAGE_BUS is None:
        bus = os.environ.get('PYNTO_MESSAGE_BUS', '').lower()
        if bus == 'sqs':
            _MESSAGE_BUS = SqsMessageBus()
        elif bus == 'redis':
            _MESSAGE_BUS = RedisMessageBus()
        elif _has_redis_env():
            _MESSAGE_BUS = RedisMessageBus()
        else:
            _MESSAGE_BUS = SqsMessageBus()
    return _MESSAGE_BUS


def set_message_bus(message_bus: MessageBus) -> MessageBus:
    global _MESSAGE_BUS
    _MESSAGE_BUS = message_bus
    return _MESSAGE_BUS


def use_redis_message_bus(**kwargs: Any) -> MessageBus:
    return set_message_bus(RedisMessageBus(**kwargs))


def use_sqs_message_bus(**kwargs: Any) -> MessageBus:
    return set_message_bus(SqsMessageBus(**kwargs))
