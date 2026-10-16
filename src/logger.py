import logging

from . import config

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"


def _build_logger() -> logging.Logger:
    """Create the project logger: console output, plus a file if config.LOG_FILE is set."""
    log = logging.getLogger("ethachi")
    if log.handlers:  # already configured
        return log
    log.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.LOG_FILE is not None:
        config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.LOG_FILE, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


logger = _build_logger()
