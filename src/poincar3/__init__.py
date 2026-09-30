import logging as _logging

from .logging import configure_logger as configure_logger
from .logging import logger as _logger
from .model import CHECKPOINT_URL, Poincar3, SSLModel

if not any(not isinstance(h, _logging.NullHandler) for h in _logger.handlers):
    configure_logger()

__version__ = "1.0.0"

__all__ = ["CHECKPOINT_URL", "Poincar3", "SSLModel", "configure_logger"]
