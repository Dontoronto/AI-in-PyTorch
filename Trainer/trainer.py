import csv
from abc import ABC, abstractmethod

import torch.onnx
from torch import save

import logging

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

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
        self.cuda_enabled = False
        self.lr_scheduler = None
        self.lr_scheduler_params = None
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

    def setLrScheduler(self, lr_scheduler_params):
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **lr_scheduler_params)
        self.lr_scheduler_params = lr_scheduler_params

    def scheduler_step(self):
        if self.lr_scheduler is not None:
            logger.debug(f"lr_scheduler step is called")
            self.lr_scheduler.step()

    def reset_scheduler(self):
        if self.lr_scheduler is not None:
            logger.debug(f"lr_scheduler will be resetted")
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.lr_scheduler_params)

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

    def export_model(self, model_path, onnx=False):
        if onnx is False:
            save(self.model.state_dict(), model_path + '.pth')
            logger.info(f'Model state_dict saved to {model_path}')
        else:
            # TODO: hard coded input format...
            torch.onnx.dynamo_export(self.model.model,torch.randn(1,1,28,28)).save(model_path + '.onnx')
            logger.info(f'Onnx-Model saved to {model_path}')

    def export_tensor_list_csv(self, csv_path, tensor_list):
        with open(csv_path, 'w') as f:
            for i, tensor in enumerate(tensor_list):
                if i == 0:
                    f.write('Tensor\n')

                for row in tensor.tolist():
                    f.write(','.join(map(str, row)) + '\n')

                f.write('\n')  # Add a blank row for separation

        logger.info("Tensors saved to tensors_multidim.csv")

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

    def createCustomDataloader(self, sampleDataset, **kwargs):
        if self.getCudaState():
            generator = torch.Generator(device='cuda')
            logger.info(f"Creating Custom DataLoader with arguments: {kwargs}")
            return DataLoader(sampleDataset, **kwargs, persistent_workers=True, prefetch_factor=2,
                               generator=generator, collate_fn=collate_fn)
        else:
            logger.info(f"Creating Custom DataLoader with arguments: {kwargs}")
            return DataLoader(sampleDataset, **kwargs)


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
    images = torch.stack(images).to('cuda', non_blocking=True)
    labels = torch.tensor(labels).to('cuda', non_blocking=True)
    return images, labels

