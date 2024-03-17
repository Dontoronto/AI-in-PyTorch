import copy
import logging

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

import torch

from .activationMaps.saliencyMap import SaliencyMap
from .featureMaps.gradCam import GradCAM

from .measurement.sparseMeasurement import pruningCounter
from .measurement.topPredictions import show_top_predictions

from .plotFuncs.plots import plot_original_vs_observation, plot_model_comparison

from .evaluationMapsStrategy import EvaluationMapsStrategy


# Note: saliency-map: https://arxiv.org/pdf/1312.6034.pdf
# Note: grad cam: https://arxiv.org/pdf/1610.02391.pdf
class Analyzer():
    def __init__(self, model, datahandler):
        '''

        :param model: neuronal model
        :param dataloaderConfig: arguments for DataLoader Class saved as dict
        '''
        self.model = model
        self.model_list = []
        self.datahandler = datahandler
        self.dataloaderConfig = None
        self.dataset = None

    def setModel(self, model):
        self.model = model

    def setDataset(self, dataset):
        self.dataset = dataset

    def setModelList(self, model_list):
        self.model_list = model_list

    def add_model(self, model):
        self.model_list.append(copy.deepcopy(model))

    def loadImage(self, path):
        return self.datahandler.loadImage(path)

    # TODO: anpassen damit gradCam auch noch so funktioniert, aktuell nur saliency map
    # TODO: allgemeine Methode überlegen so dass man easy entscheiden kann was geplottet werden soll
    def compare_models(self,test_index, test_end_index=None, eval_map_strategy: EvaluationMapsStrategy = None,
                       **kwargs):
        if eval_map_strategy is None:
            logger.warning("No evaluation Maps Strategy provided as Argument for func call")
            return
        input_images = []
        model_outputs = []

        # if there is only one image to process
        if test_end_index is None:
            batch, sample, label = self.dataset_extractor(test_index)
            img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
            temp = []
            img_tensor = None
            for model in self.model_list:
                img_tensor, saliency = eval_map_strategy.analyse(model=model,original_image=img,single_batch=batch,
                                                                 **kwargs)
                temp.append(saliency)
            model_outputs.append(temp)
            input_images.append(img_tensor)
        # loop if there are several images to be processed
        else:
            for index in range(test_index, test_end_index + 1):
                batch, sample, label = self.dataset_extractor(index)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                temp = []
                img_tensor=None
                for model in self.model_list:
                    img_tensor, saliency = eval_map_strategy.analyse(model=model,original_image=img,
                                                                     single_batch=batch,
                                                                     **kwargs)
                    temp.append(saliency)
                model_outputs.append(temp)
                input_images.append(img_tensor)

        plot_model_comparison(input_tensor_images=input_images, model_results=model_outputs)

    def runCompareTest(self, test_index,test_end_index=None, **kwargs):
        self.compare_models(test_index, test_end_index=test_end_index, eval_map_strategy=GradCAM(), **kwargs)

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
        '''
        test the model with testset and returns tuple of (correct classified samples, number of samples at all and
        percentage of right classified samples
        :param model: model to test
        :param test_loader: dataloader of test dataset
        :param loss_func: loss function to use
        :return: dictionary of correct classified samples, number of samples at all and percentage of right
                classified samples
        '''
        model.eval()
        test_loss = 0
        correct = 0
        dataset_length = len(test_loader.dataset)
        test_loader = test_loader
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= dataset_length

        percentage = 100. * correct / dataset_length

        logger.info(f'\nTest set: Average loss: {test_loss:.4f},'
                    f' Accuracy: {correct}/{dataset_length}'
                    f' ({percentage:.0f}%)')

        return {'correct_classified': correct,
                'dataset_length': dataset_length,
                'percentage_correct_classified': percentage}

    def evaluate(self, model, img, single_batch, target_layer):
        img_tensor, grad_cam = GradCAM().analyse(model=model, original_image=img,
                          single_batch=single_batch, target_layer=target_layer)
        plot_original_vs_observation(img_as_tensor=img_tensor, result=grad_cam,
                                     text=f'The Image and Gradient CAM for layer: {target_layer}')

        img_tensor, saliency = SaliencyMap().analyse(model=model, original_image=img, single_batch=single_batch)
        plot_original_vs_observation(img_as_tensor=img_tensor, result=saliency,
                                     text="The Image and Its Saliency Map")

        show_top_predictions(model=model, single_batch=single_batch, top_values=3)

        # TODO: funktion schreiben welche die Modell Liste vergleicht. Funktion soll Dict zurückgeben.
        # TODO: diese können wie die Modellliste gesammelt und dargestellt werden
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

                img_tensor, grad_cam = GradCAM().analyse(model=model, original_image=original_image,
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
            test = self.test(model=self.model, test_loader=test_loader, loss_func=loss_func)
            logger.critical(f'here is the data about the test evaluation:')
            logger.critical(test)

    def grad_all(self, test_index):
        batch, sample, label = self.dataset_extractor(test_index)
        img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
        self.gradCam_all_layers(self.model, original_image=img, single_batch=batch)

