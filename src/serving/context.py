from __future__ import annotations

from contextvars import ContextVar

REQUEST_ID: ContextVar[str] = ContextVar('REQUEST_ID', default='-')


def get_request_id() -> str:
    return REQUEST_ID.get()


def set_request_id(value: str) -> None:
    REQUEST_ID.set(value)
