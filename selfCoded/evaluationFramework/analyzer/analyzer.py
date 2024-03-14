import logging

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

import matplotlib.pyplot as plt
import torch
#cam_extractor = SmoothGradCAMpp(model)

from .activationMaps.saliencyMap import saliency_map
from .featureMaps.gradCam import gradCamLayer

from .measurement.sparseMeasurement import pruningCounter

from .plotFuncs.plots import plot_original_vs_observation


# Note: saliency-map: https://arxiv.org/pdf/1312.6034.pdf
# Note: grad cam: https://arxiv.org/pdf/1610.02391.pdf
class Analyzer():
    def __init__(self, model, datahandler):
        '''

        :param model: neuronal model
        :param dataloaderConfig: arguments for DataLoader Class saved as dict
        '''
        self.model = model
        self.model_list = None
        self.datahandler = datahandler
        self.dataloaderConfig = None
        self.dataset = None

    def setModel(self, model):
        self.model = model

    def setDataset(self, dataset):
        self.dataset = dataset

    def setModelList(self, model_list):
        self.model_list = model_list

    def loadImage(self, path):
        return self.datahandler.loadImage(path)

    def dataset_extractor(self, index):
        '''
        :param index: index of dataset which needs to be extracted
        :return tuple(batch, sample, label): batched of sample tensor shape (1,x,x,x)
                                             sample file as tensor shape (x,x,x)
                                             label of sample
        '''
        sample, label = self.dataset[index]
        batch = sample.unsqueeze(0)
        return batch, sample, label

    def test(self, model, test_loader, loss_func):
        model.eval()
        test_loss = 0
        correct = 0
        test_loader = test_loader
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        logger.info(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

    def evaluate(self, model, img, single_batch, target_layer):
        img_tensor, grad_cam = gradCamLayer(model=model, original_image=img,
                          single_batch=single_batch, target_layer=target_layer)
        plot_original_vs_observation(img_as_tensor=img_tensor, result=grad_cam,
                                     text=f'The Image and Gradient CAM for layer: {target_layer}')

        img_tensor, saliency = saliency_map(model=model, original_image=img, single_batch=single_batch)
        plot_original_vs_observation(img_as_tensor=img_tensor, result=saliency,
                                     text="The Image and Its Saliency Map")

        pruningCounter(model=model)

    def gradCam_all_layers(self, model, original_image, single_batch):
        '''
        :param model: this is the pytorch model
        :param original_image: this is the image in PIL format
        :param single_batch:  this is the model input as a single batch as tensor
        :return:
        '''
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):

                img_tensor, grad_cam = gradCamLayer(model=model, original_image=original_image,
                                  single_batch=single_batch, target_layer=name)
                plot_original_vs_observation(img_as_tensor=img_tensor, result=grad_cam,
                                             text=f'The Image and Gradient CAM for layer: {name}')

    # TODO: schauen wie man das noch schöner für mehrere Models darstellen kann
    def run_single_model_test(self, test_index, test_end_index=None,
                              test_loader=None, loss_func=None,
                              target_layer='model.conv1'):

        if test_end_index is None:
            batch, sample, label = self.dataset_extractor(test_index)
            img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
            self.evaluate(model=self.model, img=img, single_batch=batch,
                          target_layer=target_layer)

        else:
            for index in range(test_index, test_end_index + 1):
                batch, sample, label = self.dataset_extractor(index)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                self.evaluate(model=self.model, img=img, single_batch=batch,
                              target_layer=target_layer)

        if (
                isinstance(test_loader, DataLoader) and
                isinstance(loss_func, _Loss)
        ):
            self.test(model=self.model, test_loader=test_loader, loss_func=loss_func)

    def grad_all(self, test_index):
        batch, sample, label = self.dataset_extractor(test_index)
        img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
        self.gradCam_all_layers(self.model, original_image=img, single_batch=batch)

