"""
Simple logging utilities for WiLoR server.

Provides basic logging setup without external dependencies.
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text

# root logger
_root_logger = logging.getLogger()
_console_handler: Optional[logging.Handler] = None


def log_init():
    """Initialize root logger with DEBUG level."""
    _root_logger.setLevel(logging.DEBUG)


def enable_console(formatter=None):
    """Enable console logging with colored output."""
    global _root_logger, _console_handler

    if _console_handler is not None:
        return

    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    if formatter is None:
        formatter = ConsoleFormatter()
    _console_handler.setFormatter(formatter)
    _root_logger.addHandler(_console_handler)


def disable_console():
    """Disable console logging."""
    global _root_logger, _console_handler

    if _console_handler is None:
        return

    _root_logger.removeHandler(_console_handler)
    _console_handler = None


class Formatter(logging.Formatter):
    """Base logging formatter."""

    time_str = "[%(asctime)s]"
    level_str = "[%(levelname)s]"
    msg_str = "%(message)s"
    src_str = "(%(name)s @ %(filename)s:%(lineno)d)"

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ConsoleFormatter(Formatter):
    """Console formatter with colored output."""
    FORMATS = {
        logging.DEBUG:
            colored(" ".join([Formatter.time_str, Formatter.level_str, Formatter.src_str, ""]), "cyan", attrs=["dark"])
            + colored(Formatter.msg_str, "cyan"),
        logging.INFO:
            colored(" ".join([Formatter.time_str, Formatter.level_str, Formatter.src_str, ""]), "black", attrs=["dark"])
            + colored(Formatter.msg_str, "black"),
        logging.WARNING:
            colored(" ".join([Formatter.time_str, Formatter.level_str, Formatter.src_str, ""]), "yellow", attrs=["dark"])
            + colored(Formatter.msg_str, "yellow"),
        logging.ERROR:
            colored(" ".join([Formatter.time_str, Formatter.level_str, Formatter.src_str, ""]), "red", attrs=["dark"])
            + colored(Formatter.msg_str, "red"),
        logging.CRITICAL:
            colored(" ".join([Formatter.time_str, Formatter.level_str, Formatter.src_str, ""]), "red", attrs=["dark", "bold"])
            + colored(Formatter.msg_str, "red", attrs=["bold"]),
    }
