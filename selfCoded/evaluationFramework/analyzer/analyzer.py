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
from .measurement.topPredictions import show_top_predictions, getSum_top_predictions
from .measurement.distributionDensity import calculate_distribution_density, plot_distribution_density

from .plotFuncs.plots import (plot_original_vs_observation, plot_model_comparison,
                              plot_model_comparison_with_table, model_comparison_table,
                              plot_float_lists_with_thresholds)

from .evaluationMapsStrategy import EvaluationMapsStrategy

from .utils import weight_export

from .adversial import adversarialAttacker


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
        self.model_name_list = []
        self.datahandler = datahandler
        self.dataloaderConfig = None
        self.dataset = None

    def setModel(self, model):
        self.model = model

    def setDataset(self, dataset):
        self.dataset = dataset

    def setModelList(self, model_list):
        self.model_list = model_list

    def add_model(self, model, name):
        self.model_list.append(copy.deepcopy(model))
        self.model_name_list.append(name)

    def loadImage(self, path):
        return self.datahandler.loadImage(path)

    def exportLayerWeights(self, layer_name, path='test_tensor.pt', model=None):
        if model is None:
            exporting_model = self.model
        else:
            exporting_model = model

        logger.info(f"Exporting {layer_name} weights of model to {path}")
        weight_export(exporting_model, layer_name, path)


    # TODO: anpassen damit gradCam auch noch so funktioniert, aktuell nur saliency map
    # TODO: allgemeine Methode überlegen so dass man easy entscheiden kann was geplottet werden soll
    def compare_models(self,test_index, test_end_index=None, eval_map_strategy: EvaluationMapsStrategy = None,
                       **kwargs):
        if eval_map_strategy is None:
            logger.warning("No evaluation Maps Strategy provided as Argument for func call")
            return
        input_images = []
        model_outputs = []
        topk_predictions = []

        # if there is only one image to process
        if test_end_index is None:
            batch, sample, label = self.dataset_extractor(test_index)
            img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)

            result_values, img_tensor, topk_values = self.evaluation_map(img,batch,eval_map_strategy,**kwargs)

            model_outputs.append(result_values)
            input_images.append(img_tensor)
            topk_predictions.append(topk_values)
        # loop if there are several images to be processed
        else:
            for index in range(test_index, test_end_index + 1):
                batch, sample, label = self.dataset_extractor(index)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)

                result_values, img_tensor, topk_values = self.evaluation_map(img,batch,eval_map_strategy,**kwargs)

                model_outputs.append(result_values)
                input_images.append(img_tensor)
                topk_predictions.append(topk_values)

        plot_model_comparison(input_tensor_images=input_images, model_results=model_outputs,
                              model_name_list=self.model_name_list)

        layer_rows, zeros_table = self.eval_zero_layers_table_format()

        plot_model_comparison_with_table(input_images, model_outputs, zeros_table, layer_rows, self.model_name_list)

        model_comparison_table(table_data=topk_predictions,
                               row_labels=['Image 1', 'Image 2', 'Image 3'], col_labels=self.model_name_list)

        test_list = self.accuracy_table_format(**kwargs)

        model_comparison_table(table_data=test_list, row_labels=['accuracy %', 'accuracy'],
                               col_labels=self.model_name_list)

    def accuracy_table_format(self, **kwargs):
        test_list_perc = []
        test_list_abs = []
        for model in self.model_list:
            test_eval = self.test(model, **kwargs)
            test_list_perc.append(test_eval['percentage_correct_classified'])
            test_list_abs.append(test_eval['correct_classified'])

        return [test_list_perc, test_list_abs]

    def eval_zero_layers_table_format(self):
        zeros_table = list()
        layer_rows = None
        for model in self.model_list:
            pruning_dict = pruningCounter(model)
            layer_name, layer_zero_percentage = self._pruning_data_split(pruning_dict)
            layer_rows = layer_name
            zeros_table.append([str(zero_percentage) for zero_percentage in layer_zero_percentage])

        return layer_rows, zeros_table

    def _pruning_data_split(self, pruning_dict):
        layer_names = list()
        layer_pruning_rates = list()

        for key, value in pruning_dict.items():
            layer_names.append(key)
            layer_pruning_rates.append(value['zero_weights_percentage'])

        return layer_names, layer_pruning_rates

    def evaluation_map(self, img, batch, eval_map_strategy: EvaluationMapsStrategy, **kwargs):
        results = []
        img_tensor = None
        topk_predictions = []
        for model in self.model_list:
            img_tensor, outputMap = eval_map_strategy.analyse(model=model,original_image=img,single_batch=batch,
                                                             **kwargs)
            topk_predictions.append(getSum_top_predictions(model,batch,3))
            results.append(outputMap)
        return results, img_tensor, topk_predictions


    def runCompareTest(self, test_index,test_end_index=None, **kwargs):
        self.compare_models(test_index, test_end_index=test_end_index, eval_map_strategy=SaliencyMap(), **kwargs)

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

    def test(self, model, test_loader, loss_func, **kwargs):
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
                output = model(data)
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
                                             text=f'Gradient CAM for layer: {name}')

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

    def eval_epsilon_distances(self, epsilon_listW, epsilon_listZ,
                               epsilon_threshold_W, epsilon_threshold_Z):
        epsilon_symbol = '\u03B5'
        plot_float_lists_with_thresholds(epsilon_listW, epsilon_listZ,
                                         f'{epsilon_symbol}-Distance W',
                                         f'{epsilon_symbol}-Distance Z',
                                         epsilon_threshold_W, epsilon_threshold_Z,
                                         f'{epsilon_symbol}-Threshold W',
                                         f'{epsilon_symbol}-Threshold Z',
                                         f'{epsilon_symbol}-Distances over ADMM-Iterations')


    # ------------ Adversarial Area
    # TODO: kann noch so erweitert werden, dass modell liste durchgegangen wird und jedes mal die self.model -Variable
    # TODO: geändert wird. Daraufhin bleibt die Referenz bestehen und die Modellwerte werden in alle Instanzen überneommen
    # TODO: wenn es die gleiche Variable ist

    # TODO: Funktionalität zum laden von eigenen Datasets und zum ablaufen lassen von tests auf diesen Datasets.
    # TODO: Note: wir können über die Testfunktion einfach einen eigenen Dataloader für Tests reinladen
    # TODO: Note: die Aufgabe zum erstellen von Dataloadern ist die von Datahandler nicht von Analyzer
    def init_adversarial_environment(self, save_adversarial_images=False, **kwargs):
        self.adversarial_module = adversarialAttacker.AdversarialAttacker(
                                                                        self.model,
                                                                        self.datahandler.getPreprocessBatchedFunction(),
                                                                        self.datahandler.getPreprocessBackwardsNonBatchedFunction(),
                                                                        save_adversarial_images,
                                                                        **kwargs)

    def set_threat_model_config(self, threat_model_config):
        self.adversarial_module.setThreatModel(threat_model_config)

    def set_provider_config(self, provider_config):
        self.adversarial_module.setProvider(provider_config)

    def set_attack_type_config(self, attack_type_config):
        self.adversarial_module.setAttackTypeConfig(attack_type_config)

    def select_attacks_from_config(self, start_index: int = None, amount_of_attacks: int = None):
        self.adversarial_module.selectAttacks(start_index, amount_of_attacks)

    def start_adversarial_evaluation(self, start, end):
        return self.adversarial_module.evaluate(start, end)

    def enable_adversarial_saving(self, path):
        self.adversarial_module.enableAdversarialSaveMode(True)
        self.adversarial_module.setAdversarialSavePath(path)

    def enable_original_saving(self, path):
        self.adversarial_module.enableOriginalSaveMode(True)
        self.adversarial_module.setOriginalSavePath(path)

    # -----------------------------

    # ------------ Density Analysis
    def density_evaluation(self, bins=100, density_range=None, log_scale=False):
        '''
        Berechnet und stellt das Verteilungsdiagramm der Modellgewichte dar

        - bins (int): Die Anzahl der Bins für das Histogramm.
        - density_range (tuple): Ein Tupel (min, max) zur Beschränkung des Wertebereichs.
        - log_scale (bool): Wenn True, wird die y-Achse logarithmisch skaliert.
        '''
        density, bin_edges = calculate_distribution_density(self.model, bins, density_range)
        plot_distribution_density(density, bin_edges, log_scale)

    # -----------------------------


#%%
