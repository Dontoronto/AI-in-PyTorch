from abc import ABC, abstractmethod
from torch import save

import logging
logger = logging.getLogger(__name__)


class Trainer(ABC):
    def __init__(self, model, loss, optimizer):
        '''

        :param model: neuronal model
        :param dataloaderConfig: arguments for DataLoader Class saved as dict
        '''
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataloaderConfig = None
        self.snapshotConfig = None


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def prepareDataset(self):
        pass

    def getLossFunction(self):
        return self.loss

    # NOTE: this method is for saving a model state with a log-message
    def export_model(self, model_path):
        save(self.model.state_dict(), model_path)
        logger.info('Model state_dict saved to %s', model_path)
        # , onnx_path=None
        # if onnx_path is not None:
        #     assert input_shape is not None, 'input_shape must be specified to export onnx model'
        #     # input info needed
        #     if device is None:
        #         device = torch.device('cpu')
        #     input_data = torch.Tensor(*input_shape)
        #     torch.onnx.export(self._model_to_prune, input_data.to(device), onnx_path)
        #     _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

    def setDataLoaderSettings(self, kwargs: dict):
        '''
        sets custom Dataloader configuration
        '''
        self.dataloaderConfig = kwargs

    def setSnapshotSettings(self, kwargs: dict):
        '''
        sets snapshot configuration
        '''
        self.snapshotConfig = kwargs


