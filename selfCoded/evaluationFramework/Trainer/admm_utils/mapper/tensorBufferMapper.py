import logging

logger = logging.getLogger(__name__)

class TensorBufferConfigMapper:

    def __init__(self, handler, config):
        self.handler = handler
        self.config = config
        self.map_config_to_handler()

    def map_config_to_handler(self):
        logger.info("Tensor Buffer Config was loaded into MultiProcessHandler")

        # Inspecting kernel settings
        if 'inspecting_kernel' in self.config:
            kernel_config = self.config['inspecting_kernel']
            self.handler.layer_index = kernel_config.get('layer_index', [])
            self.handler.output_input_channel = kernel_config.get('output_input_channel', [])

            logger.info(f"layer_index set to {self.handler.layer_index}")
            logger.info(f"output_input_channel set to {self.handler.output_input_channel}")

        # Processes settings
        if 'processes' in self.config:
            process_config = self.config['processes']
            self.handler.file_path = process_config.get('file_path', '')
            self.handler.file_path_zero_matrices = process_config.get('file_path_zero_matrices', '')
            self.handler.capacity = process_config.get('capacity', 0)
            self.handler.clear_files = process_config.get('clear_files', False)
            self.handler.clear_after = process_config.get('clear_after', False)
            self.handler.convert_to_png = process_config.get('convert_to_png', False)

            logger.info(f"file_path set to {self.handler.file_path}")
            logger.info(f"file_path_zero_matrices set to {self.handler.file_path_zero_matrices}")
            logger.info(f"capacity set to {self.handler.capacity}")
            logger.info(f"clear_files set to {self.handler.clear_files}")
            logger.info(f"clear_after set to {self.handler.clear_after}")
            logger.info(f"convert_to_png set to {self.handler.convert_to_png}")

