import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


logger = logging.getLogger("lab_compliance_agent")


def configure_logging() -> None:
    """Configure application-wide structured logging."""
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def generate_request_id() -> str:
    """Generate a unique ID for tracing a request."""
    return str(uuid.uuid4())


def log_event(
    event: str,
    request_id: str | None = None,
    **fields: Any,
) -> None:
    """Emit a structured JSON log event."""
    payload = {
        "timestamp": time.time(),
        "event": event,
    }

    if request_id:
        payload["request_id"] = request_id

    payload.update(fields)

    logger.info(json.dumps(payload, default=str))


@contextmanager
def measure_time(
    event: str,
    request_id: str | None = None,
    **fields: Any,
) -> Iterator[None]:
    """Measure execution time and emit start/end events."""
    start = time.perf_counter()

    log_event(
        f"{event}.started",
        request_id=request_id,
        **fields,
    )

    try:
        yield
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000

        log_event(
            f"{event}.failed",
            request_id=request_id,
            duration_ms=round(duration_ms, 2),
            error_type=type(exc).__name__,
            error=str(exc),
            **fields,
        )

        raise
    else:
        duration_ms = (time.perf_counter() - start) * 1000

        log_event(
            f"{event}.completed",
            request_id=request_id,
            duration_ms=round(duration_ms, 2),
            **fields,
        )
