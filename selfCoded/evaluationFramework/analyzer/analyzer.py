import logging

import cv2

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torchvision.transforms as T
import numpy as np


from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import torch
#cam_extractor = SmoothGradCAMpp(model)


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


    def setDataLoaderSettings(self, kwargs: dict):
        '''
        sets custom Dataloader configuration
        '''
        self.dataloaderConfig = kwargs

    def setModel(self, model):
        self.model = model

    def setDataset(self, dataset):
        self.dataset = dataset

    def setModelList(self, model_list):
        self.model_list = model_list

    def loadImage(self, path):
        return self.datahandler.loadImage(path)

    def pruningCounter(self, model):
        # zeros_count = (tensor == 0).sum().item()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                # Count zeros and total weights
                zeros_count = torch.eq(parameter, 0).sum().item()
                total_weights = parameter.numel()
                zero_weights_percentage = (zeros_count / total_weights) * 100

                # Print layer information
                logger.info(f"Layer: {name}, Zero weights: {total_weights}/{zeros_count} ({zero_weights_percentage:.2f}%)")

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

    def evaluate(self, model, img, single_batch):
        self.gradCamLayer(model=model, original_image=img, single_batch=single_batch)

        self.saliency_map(model=model, original_image=img, single_batch=single_batch)

        self.pruningCounter(model=model)

    # TODO: target_layer noch ändern so dass man irgendwie per json mitgeben kann
    def gradCamLayer(self, model, original_image, single_batch, target_layer='model.conv1'):
        '''
        :param model: model to test
        :param original_image: PIL Image type
        :param single_batch: Tensor type batched shape (1,channel,width,height)
        :param target_layer: "string of layer name
        '''

        image = original_image.copy()
        image = image.convert(mode='RGB')
        image = T.Compose([T.ToTensor()])(image)
        img = T.Compose([T.ToTensor()])(original_image.copy())
        with SmoothGradCAMpp(model, target_layer=target_layer) as cam_extractor:
            # Preprocess your data and feed it to the model
            model.eval()
            out = model(single_batch).squeeze(0).softmax(0)

            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            # Display it
            # plt.imshow(result)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
            # logger.critical(result.size)
            # logger.critical(result.getdata())
            # logger.critical(result.info)

        fig, ax = plt.subplots(1, 2, facecolor='dimgray')
        if img.shape[0] == 1:
            # Note: this is just for single channel images gray with values from 0 to 1
            ax[0].imshow(img.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        else:
            # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
            ax[0].imshow(img.cpu().detach().clone().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(result)
        ax[1].axis('off')
        plt.tight_layout()
        fig.suptitle(f'The Image and Gradient CAM for layer: {target_layer}')
        plt.show()

    # TODO: maybe it needs some adjustments, to exhausted after fighting with matplotlib atm
    def saliency_map(self, model, original_image, single_batch):
        '''
        :param model: model to test
        :param original_image: PIL image type
        :param single_batch: Tensor type batched shape (1,channel,width,height)
        :return:
        '''
        model.eval()
        width, height = original_image.size
        img = T.Compose([T.ToTensor()])(original_image)

        # Set the requires_grad_ to the image for retrieving gradients
        single_batch.requires_grad_()

        # Retrieve output from the image
        output = model(single_batch)

        # Catch the output
        output_idx = output.argmax()
        output_max = output[0, output_idx]

        # Do backpropagation to get the derivative of the output based on the image
        output_max.backward()

        # Retireve the saliency map and also pick the maximum value from channels on each pixel.
        # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
        saliency, _ = torch.max(single_batch.grad.data.abs(), dim=1)
        saliency = saliency.reshape(width, height)

        # Visualize the image and the saliency map
        fig, ax = plt.subplots(1, 2, facecolor='dimgray')
        if img.shape[0] == 1:
            # Note: this is just for single channel images gray with values from 0 to 1
            ax[0].imshow(img.cpu().detach().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        else:
            # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
            ax[0].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(saliency.cpu(), cmap='gray')
        ax[1].axis('off')
        plt.tight_layout()
        fig.suptitle('The Image and Its Saliency Map')
        plt.show()
        model.eval()

    # TODO: schauen wie man das noch schöner für mehrere Models darstellen kann
    def run_single_model_test(self, test_index, test_end_index=None,
                              test_loader=None, loss_func=None):

        if test_end_index is None:
            sample, label = self.dataset[test_index]
            batch = sample.unsqueeze(0)
            img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
            self.evaluate(model=self.model, img=img, single_batch=batch)

        else:
            for index in range(test_index, test_end_index + 1):
                sample, label = self.dataset[index]
                batch = sample.unsqueeze(0)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                self.evaluate(model=self.model, img=img, single_batch=batch)

        if (
                isinstance(test_loader, DataLoader) and
                isinstance(loss_func, Optimizer)
        ):
            self.test(model=self.model, test_loader=test_loader, loss_func=loss_func)


#%%
