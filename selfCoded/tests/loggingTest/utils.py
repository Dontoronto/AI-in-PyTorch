# utils.py
import logging

logger = logging.getLogger(__name__)

def some_util_function():
    logger.info("This is an info message from some_util_function")
    logger.critical("hallo critical")
    logger.debug("debug info")
