import logging

import cv2

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
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

    def loadImage(self, path):
        return self.datahandler.loadImage(path)

    def createDataLoader(self, dataset):
        if self.dataloaderConfig is not None:
            logger.info("Created Dataloader with settings: " + str(self.dataloaderConfig))
            return DataLoader(dataset, **self.dataloaderConfig)
        else:
            logger.warning("No Configs for Dataloader available, creating Dataloader with default arguments")
            return DataLoader(dataset)

    def gradCamLayer(self, original_image, batch, target_layer='model.conv1'):
        '''

        :param img: PIL Image type
        :param batch:  Tensor type batched shape (1,channel,width,height)
        :param target_layer: "string of layer name
        :return:
        '''

        image = original_image.copy()
        image = image.convert(mode='RGB')
        image = T.Compose([T.ToTensor()])(image)
        logger.critical(torch.unique(image))
        img = T.Compose([T.ToTensor()])(original_image.copy())
        with SmoothGradCAMpp(self.model, target_layer=target_layer) as cam_extractor:
            # Preprocess your data and feed it to the model
            self.model.eval()
            out = self.model(batch).squeeze(0).softmax(0)

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
    def saliency_map(self, original_image, batched_preprocessed_input):
        width, height = original_image.size
        img = T.Compose([T.ToTensor()])(original_image)

        # Set the requires_grad_ to the image for retrieving gradients
        batched_preprocessed_input.requires_grad_()

        # Retrieve output from the image
        output = self.model(batched_preprocessed_input)

        # Catch the output
        output_idx = output.argmax()
        output_max = output[0, output_idx]

        # Do backpropagation to get the derivative of the output based on the image
        output_max.backward()

        # Retireve the saliency map and also pick the maximum value from channels on each pixel.
        # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
        saliency, _ = torch.max(batched_preprocessed_input.grad.data.abs(), dim=1)
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

    def runtest(self):
        image, label = self.dataset[0]
        batch = image.unsqueeze(0)

        img = self.datahandler.preprocessBackwardsNonBatched(image)

        #img.show()
        #new = img.convert(mode='RGB')
        #new.show()
        # image = image.unsqueeze(0)
        #
        # pre = T.Compose([T.ToTensor()])
        # new_t = pre(img)#.unsqueeze(0)

        #print(new_t)

        #out = self.model(batch)
        #image = self.datahandler.preprocessBackwardsBatched(sample)
        #logger.critical(image.shape)
        #logger.critical(sample.shape)


        self.gradCamLayer(img, batch)
        self.saliency_map(original_image=img, batched_preprocessed_input=batch)


        #logger.critical(out)


#%%
