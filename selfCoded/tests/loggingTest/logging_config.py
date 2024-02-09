# logging_config.py
import logging
import logging.config

def setup_logging(default_path='config/logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    import os
    import json

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        print("not else")
    else:
        print("else")
        logging.basicConfig(level=default_level)
