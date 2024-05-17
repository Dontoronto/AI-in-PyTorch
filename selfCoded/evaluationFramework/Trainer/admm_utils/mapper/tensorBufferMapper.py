import logging

logger = logging.getLogger(__name__)

class TensorBufferConfigMapper:

    def __init__(self, handler, config):
        self.handler = handler
        self.config = config
        self.map_config_to_handler()

    def map_config_to_handler(self):

        # Inspecting kernel settings
        if 'inspecting_kernel' in self.config:
            kernel_config = self.config['inspecting_kernel']
            self.handler.layer_index = kernel_config.get('layer_index', [])
            self.handler.output_input_channel = kernel_config.get('output_input_channel', [])

            logger.debug(f"layer_index set to {self.handler.layer_index}")
            logger.debug(f"output_input_channel set to {self.handler.output_input_channel}")
        else:
            logger.error(f"TensorBufferMapper was not able to map TensorBufferConfig: \n {self.config}")

        # Processes settings
        if 'processes' in self.config:
            process_config = self.config['processes']
            self.handler.capacity = process_config.get('capacity', 0)
            self.handler.clear_files = process_config.get('clear_files', False)
            self.handler.clear_after = process_config.get('clear_after', False)
            self.handler.convert_to_png = process_config.get('convert_to_png', False)
            self.handler.convert_to_gif = process_config.get('convert_to_gif', False)

            logger.debug(f"capacity set to {self.handler.capacity}")
            logger.debug(f"clear_files set to {self.handler.clear_files}")
            logger.debug(f"clear_after set to {self.handler.clear_after}")
            logger.debug(f"convert_to_png set to {self.handler.convert_to_png}")
            logger.debug(f"convert_to_gif set to {self.handler.convert_to_gif}")
        else:
            logger.error(f"TensorBufferMapper was not able to map TensorBufferConfig: \n {self.config}")

        logger.info(f"Tensor Buffer Config was loaded into MultiProcessHandler: \n {self.config}")

