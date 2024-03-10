import logging
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
import torchvision.transforms as T


from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
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

    def gradCamLayer(self, img, batch, target_layer='model.conv1'):
        '''

        :param img: PIL Image type
        :param batch:  Tensor type
        :param target_layer: "string of layer name
        :return:
        '''
        image = img.convert(mode='RGB')
        image = T.Compose([T.ToTensor()])(image)
        with SmoothGradCAMpp(self.model, target_layer=target_layer) as cam_extractor:
            # Preprocess your data and feed it to the model
            self.model.eval()
            self.model.train()
            out = self.model(batch).squeeze(0).softmax(0)

            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            # Display it
            plt.imshow(result)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def runtest(self):
        image, label = self.dataset[0]
        batch = image.unsqueeze(0)

        img = self.datahandler.preprocessBackwardsNonBatched(image)
        #img.show()
        #new = img.convert(mode='RGB')
        #new.show()
        #pre = T.Compose([T.ToTensor()])
        #new_t = pre(new)
        #print(new_t)

        out = self.model(batch)
        #image = self.datahandler.preprocessBackwardsBatched(sample)
        #logger.critical(image.shape)
        #logger.critical(sample.shape)
        self.gradCamLayer(img, batch)
        #logger.critical(out)


#%%
