"""
Central logging configuration for InterviewDNA.

Import and call `configure_logging()` once, as early as possible, in each
entrypoint (api/main.py, frontend/streamlit_app.py). Every module in the
codebase then just does:

    import logging
    logger = logging.getLogger(__name__)

...and it inherits this configuration automatically.

Log level is controlled by the LOG_LEVEL env var (default INFO). Logs go to
stdout (so `docker compose logs -f api` and plain terminal output both work)
and, for local dev, to a rotating file at logs/interviewdna.log.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

_configured = False

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    try:
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "interviewdna.log", maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Read-only filesystem (some managed platforms) -- stdout logging
        # alone is still fine there.
        pass

    # Quiet down noisy third-party loggers unless DEBUG is explicitly requested.
    if level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("interviewdna").info(
        "Logging configured (level=%s, file=%s)", level_name, "logs/interviewdna.log"
    )
