"""
logger.py
---------
Centralised logging configuration using Loguru.

Every module in this project gets its logger from here:

    from logger import get_logger
    logger = get_logger(__name__)

Features
--------
- Coloured, human-readable output to stderr during development.
- Rotating JSON log files for production (parsed by log-aggregators).
- A separate error-only sink that captures WARNING+ to errors.log.
- Log level controlled by the LOG_LEVEL env variable (default: INFO).
- Sensitive field redaction for PII / credentials.

Log files (written to logs/ directory):
    logs/app.log        → all levels, rotates at 20 MB, 14-day retention
    logs/errors.log     → WARNING+ only, rotates at 10 MB, 30-day retention
"""

import logging
import os
import sys
from pathlib import Path

from loguru import logger as _loguru_logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_LEVEL: str   = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR: Path    = Path("logs")
LOG_FORMAT: str  = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
JSON_LOG_FORMAT: str = "{message}"   # Loguru serialise=True handles the rest


def _setup_logger(level: str = LOG_LEVEL) -> None:
    """
    Configure Loguru sinks. Safe to call multiple times — existing sinks
    are removed before re-adding so the function is idempotent.
    """
    _loguru_logger.remove()   # drop the default stderr sink

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Sink 1: stderr (human-readable, coloured) ───────────────────────
    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,   # show variable values in tracebacks (dev only)
        enqueue=False,
    )

    # ── Sink 2: app.log (JSON, all levels, rotating) ────────────────────
    _loguru_logger.add(
        LOG_DIR / "app.log",
        level=level,
        format=JSON_LOG_FORMAT,
        serialize=True,          # structured JSON per line
        rotation="20 MB",
        retention="14 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,            # async write — non-blocking
        backtrace=True,
        diagnose=False,          # never put variable values in prod logs
    )

    # ── Sink 3: errors.log (WARNING+, long retention for post-mortems) ──
    _loguru_logger.add(
        LOG_DIR / "errors.log",
        level="WARNING",
        format=JSON_LOG_FORMAT,
        serialize=True,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


# ---------------------------------------------------------------------------
# Intercept stdlib logging → Loguru
# ---------------------------------------------------------------------------
# Third-party libraries (SQLAlchemy, uvicorn, etc.) use stdlib logging.
# This handler routes all of it through Loguru so everything ends up in
# the same structured log files.

class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level to Loguru level name
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Walk up frames to find the actual caller (skip logging internals)
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _intercept_stdlib_logging() -> None:
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    # Suppress overly verbose loggers
    for noisy in ("urllib3", "httpx", "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Sensitive field redaction
# ---------------------------------------------------------------------------
_REDACT_KEYS = frozenset(
    {"password", "api_key", "token", "secret", "key_file", "credentials"}
)


def redact(data: dict) -> dict:
    """
    Shallow-copy `data` with sensitive values replaced by '***REDACTED***'.
    Pass dicts through this before logging them.

    Example
    -------
    >>> redact({"user": "alice", "password": "hunter2"})
    {'user': 'alice', 'password': '***REDACTED***'}
    """
    return {
        k: "***REDACTED***" if k.lower() in _REDACT_KEYS else v
        for k, v in data.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str):
    """
    Return a Loguru logger bound to `name` (typically `__name__`).

    The returned logger automatically includes the module name in every
    log record without any extra configuration in the calling module.

    Usage
    -----
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Geocoding query: %s", query)
    """
    return _loguru_logger.bind(module=name)


# Run setup on import so any module that does `from logger import get_logger`
# immediately gets a configured logger.
_setup_logger()
_intercept_stdlib_logging()