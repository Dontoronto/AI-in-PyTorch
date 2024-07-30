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

        adv_config = self.config.get('adv_settings', {})
        if 'adv_save_enabled' in adv_config:
            self.analyzer.adv_save_enabled = adv_config.get('adv_save_enabled')
            logger.debug(f"adv_save_enabled set to {self.analyzer.adv_save_enabled}")
        else:
            logger.error(f"adv_save_enabled was not able to map config: {adv_config}")

        if 'adv_original_save_enabled' in adv_config:
            self.analyzer.adv_original_save_enabled = adv_config.get('adv_original_save_enabled')
            logger.debug(f"adv_original_save_enabled set to {self.analyzer.adv_original_save_enabled}")
        else:
            logger.error(f"adv_original_save_enabled was not able to map config: {adv_config}")

        if 'adv_attack_selection' in adv_config:
            self.analyzer.adv_attack_selection = adv_config.get('adv_attack_selection')
            logger.debug(f"adv_attack_selection set to {self.analyzer.adv_attack_selection}")
        else:
            logger.error(f"adv_attack_selection was not able to map config: {adv_config}")

        if 'adv_attack_selection_range' in adv_config:
            self.analyzer.adv_attack_selection_range = adv_config.get('adv_attack_selection_range')
            logger.debug(f"adv_attack_selection_range set to {self.analyzer.adv_attack_selection_range}")
        else:
            logger.error(f"adv_attack_selection_range was not able to map config: {adv_config}")

        if 'adv_sample_range_start' in adv_config:
            self.analyzer.adv_sample_range_start = adv_config.get('adv_sample_range_start')
            logger.debug(f"adv_sample_range_start set to {self.analyzer.adv_sample_range_start}")
        else:
            logger.error(f"adv_sample_range_start was not able to map config: {adv_config}")

        if 'adv_sample_range_end' in adv_config:
            self.analyzer.adv_sample_range_end = adv_config.get('adv_sample_range_end')
            logger.debug(f"adv_sample_range_end set to {self.analyzer.adv_sample_range_end}")
        else:
            logger.error(f"adv_sample_range_end was not able to map config: {adv_config}")

        if 'adv_only_success_flag' in adv_config:
            self.analyzer.adv_only_success_flag = adv_config.get('adv_only_success_flag')
            logger.debug(f"adv_only_success_flag set to {self.analyzer.adv_only_success_flag}")
        else:
            logger.error(f"adv_only_success_flag was not able to map config: {adv_config}")

        if 'adv_shuffle' in adv_config:
            self.analyzer.adv_shuffle = adv_config.get('adv_shuffle')
            logger.debug(f"adv_shuffle set to {self.analyzer.adv_shuffle}")
        else:
            logger.error(f"adv_shuffle was not able to map config: {adv_config}")

        if 'indices_list_path' in adv_config:
            self.analyzer.indices_list_path = adv_config.get('indices_list_path')
            logger.debug(f"indices_list_path set to {self.analyzer.indices_list_path}")
        else:
            logger.error(f"indices_list_path was not able to map config: {adv_config}")

        if 'generate_indices_list' in adv_config:
            self.analyzer.generate_indices_list = adv_config.get('generate_indices_list')
            logger.debug(f"generate_indices_list set to {self.analyzer.generate_indices_list}")
        else:
            logger.error(f"generate_indices_list was not able to map config: {adv_config}")

        #if 'cuda_enabled' in self.config:
        #    self.analyzer.cuda_enabled = self.config.get('cuda_enabled')
        #    logger.debug(f"cuda_enabled set to {self.analyzer.cuda_enabled}")
        #else:
        #    logger.error(f"cuda_enabled was not able to map config: {self.config}")

        logger.info(f"Analyzer config was loaded into Analyzer: {self.config}")