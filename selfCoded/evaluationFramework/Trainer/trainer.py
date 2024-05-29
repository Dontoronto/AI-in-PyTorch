import csv
from abc import ABC, abstractmethod

import torch.onnx
from torch import save

import logging

from torch.utils.data import DataLoader

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
        try:
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                torch.set_default_device('cuda')
                self.cuda_enabled = True
                print(f"Device= {device}")
        except Exception:
            print("Failed to set device automatically, please try set_device() manually.")
            self.cuda_enabled = False

    def getCudaState(self):
        return self.cuda_enabled

    def setCudaState(self, cuda_flag: bool):
        if isinstance(cuda_flag, bool):
            self.cuda_enabled = cuda_flag
            if cuda_flag is True:
                torch.set_default_device('cuda')
            else:
                torch.set_default_device('cpu')
        else:
            logger.critical("Cuda flag is not a bool value")

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def prepareDataset(self):
        pass

    def getLossFunction(self):
        return self.loss

    # NOTE: this method is for saving a model state with a log-message
    def export_model(self, model_path, onnx=False):
        if onnx is False:
            save(self.model.state_dict(), model_path + '.pth')
            logger.info(f'Model state_dict saved to {model_path}')
        else:
            # TODO: hard coded input format...
            torch.onnx.dynamo_export(self.model.model,torch.randn(1,1,28,28)).save(model_path + '.onnx')
            logger.info(f'Onnx-Model saved to {model_path}')

    def export_tensor_list_csv(self, csv_path, tensor_list):
        # with open(csv_path, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     for tensor in tensor_list:
        #         writer.writerow(['Tensor'])
        #         writer.writerows(tensor.tolist())
        #         writer.writerow([])  # Add a blank row for separation
        with open(csv_path, 'w') as f:
            for i, tensor in enumerate(tensor_list):
                if i == 0:
                    f.write('Tensor\n')

                for row in tensor.tolist():
                    f.write(','.join(map(str, row)) + '\n')

                f.write('\n')  # Add a blank row for separation

        logger.info("Tensors saved to tensors_multidim.csv")
        # , onnx_path=None
        # if onnx_path is not None:
        #     assert input_shape is not None, 'input_shape must be specified to export onnx model'
        #     # input info needed
        #     if device is None:
        #         device = torch.device('cpu')
        #     input_data = torch.Tensor(*input_shape)
        #     torch.onnx.export(self._model_to_prune, input_data.to(device), onnx_path)
        #     _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

    def createDataLoader(self, sampleDataset):
        if self.dataloaderConfig is not None:
            logger.info("Created Dataloader with settings: " + str(self.dataloaderConfig))
            if self.getCudaState():
                generator = torch.Generator(device='cuda')
                return DataLoader(sampleDataset, **self.dataloaderConfig, persistent_workers=True, prefetch_factor=2,
                                  generator=generator, collate_fn=collate_fn)
                # return DataLoader(sampleDataset, **self.dataloaderConfig, persistent_workers=True, prefetch_factor=2,
                #                   generator=generator, collate_fn=collate_fn)
            else:
                return DataLoader(sampleDataset, **self.dataloaderConfig)
        else:
            logger.warning("No Configs for Dataloader available, creating Dataloader with default arguments")
            if self.getCudaState():
                generator = torch.Generator(device='cuda')
                return DataLoader(sampleDataset, prefetch_factor=2,
                                  persistent_workers=True, num_workers=4, generator=generator, collate_fn=collate_fn)
            else:
                return DataLoader(sampleDataset)

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

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images).to('cuda')
    labels = torch.tensor(labels).to('cuda')
    return images, labels


