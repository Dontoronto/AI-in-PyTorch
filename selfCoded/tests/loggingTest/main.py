# main.py
from logging_config import setup_logging
setup_logging()

import logging
import utils
logger = logging.getLogger(__name__)

def main():
    logger.info("Application start")
    # Your main application code here
    utils.some_util_function()
    utils.some_util_function()
    utils.some_util_function()
    utils.some_util_function()
    logger.debug("hello")

if __name__ == "__main__":
    main()
