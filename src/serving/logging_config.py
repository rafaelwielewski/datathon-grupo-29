from __future__ import annotations

import logging

from src.serving.context import get_request_id


class RequestIDFilter(logging.Filter):
    """Injeta o request_id do ContextVar em todos os log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s [%(request_id)s] — %(message)s'))
    logging.root.setLevel(logging.INFO)
    logging.root.handlers = [handler]
