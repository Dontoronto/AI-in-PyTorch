import logging

logger = logging.getLogger(__name__)

class AnalyzerConfigMapper:
    def __init__(self, analyzer, config):
        self.analyzer = analyzer
        self.config = config
        self.map_config_to_analyzer()

    def map_config_to_analyzer(self):
        if 'testrun' in self.config:
            self.analyzer.testrun = self.config.get('testrun')
            logger.debug(f"testrun set to {self.analyzer.testrun}")
        else:
            logger.error(f"AnalyzerConfigMapper was not able to map config: {self.config}")

        if 'save' in self.config:
            self.analyzer.save = self.config.get('save')
            logger.debug(f"save set to {self.analyzer.save}")
        else:
            logger.error(f"AnalyzerConfigMapper was not able to map config: {self.config}")

        if 'save_path' in self.config:
            self.analyzer.save_path = self.config.get('save_path')
            logger.debug(f"save_path set to {self.analyzer.save_path}")
        else:
            logger.error(f"AnalyzerConfigMapper was not able to map config: {self.config}")

        if 'name' in self.config:
            self.analyzer.name = self.config.get('name')
            logger.debug(f"name set to {self.analyzer.name}")
        else:
            logger.error(f"AnalyzerConfigMapper was not able to map config: {self.config}")

        if 'analysis_methods' in self.config:
            self.analyzer.analysis_methods = self.config.get('analysis_methods')
            logger.debug(f"analysis_methods set to {self.analyzer.analysis_methods}")
        else:
            logger.error(f"AnalyzerConfigMapper was not able to map config: {self.config}")

        if 'copy_config' in self.config:
            self.analyzer.copy_config = self.config.get('copy_config')
            logger.debug(f"copy_config set to {self.analyzer.copy_config}")
        else:
            logger.error(f"copy_config was not able to map config: {self.config}")

        if 'config_path' in self.config:
            self.analyzer.config_path = self.config.get('config_path')
            logger.debug(f"config_path set to {self.analyzer.config_path}")
        else:
            logger.error(f"config_path was not able to map config: {self.config}")

        logger.info(f"Analyzer config was loaded into Analyzer: {self.config}")