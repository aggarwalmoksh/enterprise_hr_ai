import logging
import sys
from datetime import datetime


class AppFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(7)
        module = record.name.ljust(20)
        message = record.getMessage()
        return f"{timestamp} | {level} | {module} | {message}"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(AppFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
