import logging

from .high_level import _translate_stream, translate


log = logging.getLogger(__name__)

__version__ = "1.9.6"
__author__ = "Byaidu"
__all__ = ["translate", "_translate_stream"]
