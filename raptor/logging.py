import logging
import threading
from enum import Enum

from rich.console import Console
from rich.logging import RichHandler


class _LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


_logger_lock = threading.Lock()
_configured_loggers: dict[str, bool] = {}


def get_logger(name: str, level: str | _LogLevel = "INFO") -> logging.Logger:
    """
    Retrieves a logger with the given name and configures it to use RichHandler if not already configured.

    Args:
        name (str): The name of the logger.
        level (_LogLevel): The logging level to set on the logger. Defaults to `INFO`.

    Returns:
        logging.Logger: The configured logger.
    """
    with _logger_lock:
        if name not in _configured_loggers:
            console = Console()
            handler = RichHandler(console=console, rich_tracebacks=True)
            handler.setFormatter(logging.Formatter("%(message)s"))

            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(handler)
            logger.propagate = False

            _configured_loggers[name] = True
        else:
            logger = logging.getLogger(name)

    return logger
