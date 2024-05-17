# logging_config.py
import logging
import logging.config

def setup_logging(default_path='configs/logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    import os
    import json

    path = default_path
    value = os.getenv(env_key, None)
    logging.debug(f"Environment Variable env_key={value} was loaded")
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logging.debug(f"Logging Config was configured config from path {default_path}")
    else:
        logging.basicConfig(level=default_level)
        logging.debug(f"Default Loggin was set to logging_level={default_level}")
