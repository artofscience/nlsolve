from __future__ import annotations

import logging
import sys


class CustomFormatter(logging.Formatter):
    white = '\x1b[5m'
    green = '\x1b[92m'
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\u001b[31m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'
    format = "%(levelname)-8s %(name)-15s %(message)s"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger(name, logging_level, formatter: logging.Formatter = None):
    # formatter
    formatter = formatter if formatter is not None else logging.Formatter('%(levelname)s: %(name)s - %(message)s')

    # handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # logger
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging_level)
    logger.propagate = False
    return logger
