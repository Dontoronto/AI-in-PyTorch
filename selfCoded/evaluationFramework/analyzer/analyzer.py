import copy
import logging
import os, sys

import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
import matplotlib

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
                              plot_float_lists_with_thresholds,
                              combine_plots_vertically,
                              plot_table)

from .evaluationMapsStrategy import EvaluationMapsStrategy

from .utils import weight_export, copy_directory, create_directory

from .adversial import adversarialAttacker

from .mapper.analyzerMapper import AnalyzerConfigMapper

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from SharedServices.utils import copy_directory


# Note: saliency-map: https://arxiv.org/pdf/1312.6034.pdf
# Note: grad cam: https://arxiv.org/pdf/1610.02391.pdf
class Analyzer:
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
        self.trainer = None

        # ====== Note: can be set via config file
        self.save_path = ""
        self.testrun = ""
        self.save = False
        self.name = ""
        self.analysis_methods = None
        self.copy_config = False
        self.config_path = None

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

    def setSavePath(self, param):
        self.save_path = param

    # ============================== Note: used for automatic report generation

    def check_dataset(self):
        if self.dataset is not None:
            return True
        else:
            return False

    def setAnalyzerConfig(self, kwargs):
        AnalyzerConfigMapper(self, kwargs)

    def setTrainer(self, trainer_obj):
        self.trainer = trainer_obj

    def startTestrun(self, kwargs):

        self.adapt_trainer_configs_to_analyzer()

        self.trainer.train(**kwargs)

        model_filenames = self.load_model_path_from_path(self.save_path)

        for method, params in self.analysis_methods.items():
            self.get_analysis_method(method, params, model_filenames)

        if self.copy_config is True:
            if os.path.isdir(self.config_path) is True and os.path.isdir(self.save_path):
                base_dir = os.path.basename(self.config_path)
                copy_directory(self.config_path, os.path.join(self.save_path, base_dir))
            else:
                logger.error(f"config source path or destination path are not set properly")
                logger.error(f"config source path: \n {self.config_path}")
                logger.error(f"config destination path: \n {self.save_path}")

        # # postEvaluation
        # model_filenames = self.load_model_path_from_path(self.save_path)

        # self.report_grad_and_original(model_filenames, 27, grad_range=10)
        # self.report_grad(model_filenames, 27, 10)
        # self.report_saliency(model_filenames, 27, 10)
        # self.report_saliency_and_original(model_filenames, 27,10)
        # self.report_topk_accuracy(model_filenames,27,10)
        # self.report_pruning(model_filenames)
        # test = self.trainer.getLossFunction()
        # loader = self.trainer.getTestLoader()
        # self.report_accuracy(model_filenames, loader, test, titel='test')

        plt.close("all")

    def report_accuracy(self, model_filenames, test_loader, loss_func, titel='default_testset'):
        accuracy_path = os.path.join(self.save_path, 'Accuracy')
        create_directory(accuracy_path)

        accuracy_list = list()
        model_name_list = list()

        column_metric = ['Correct Classified Ratio', 'Correct Classified', 'Dataset length']

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))

            correct_classified, dataset_length, percentage = self.test(self.model, test_loader=test_loader,
                                                                       loss_func=loss_func)

            accuracy_list.append([percentage, correct_classified, dataset_length])
            model_name_list.append(model_name)

        fig = plot_table(accuracy_list, model_name_list, column_metric)
        fig.savefig(os.path.join(accuracy_path, f"{titel}_accuracy.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)



    def report_pruning(self, model_filenames):

        pruning_path = os.path.join(self.save_path, 'PruningStats')
        create_directory(pruning_path)

        pruning_stats_list = list()

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))

            pruning_dict = pruningCounter(self.model)
            layer_names, metric, weight_stats = self._pruning_data_split(pruning_dict)

            fig = plot_table(weight_stats, metric, layer_names)

            fig.savefig(os.path.join(pruning_path, f"{model_name}_pruning_state.png"),
                        dpi=300, facecolor='dimgray', bbox_inches='tight')
            plt.close(fig)

            total_zero_params = sum(weight_stats[0])
            total_model_params = sum(weight_stats[1])
            total_pruning_ratio = round(total_zero_params/total_model_params * 100, 4)

            # zero params sum
            pruning_stats_list.append([total_pruning_ratio, total_zero_params, total_model_params])

        metric = ['total_pruning_ratio', 'total_zero_params', 'total_model_params']
        model_names = [os.path.splitext(model_filename)[0] for model_filename in model_filenames]
        fig = plot_table(pruning_stats_list, model_names, metric)

        fig.savefig(os.path.join(pruning_path, f"model_pruning_ratio.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)


    def report_topk_accuracy(self, model_filenames, topk_start_index, topk_range):

        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if topk_range == 0:
            topk_range = 1

        # Create list of figures with orignal image and grad image combined
        topk_path = os.path.join(self.save_path, 'TopK_Predictions')
        create_directory(topk_path)

        topk_models_list = []
        topk_prediction_list = []
        for model_filename in model_filenames:
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            topk_list = []
            topk_pred_list = []

            for i in range(topk_range):
                batch, sample, label = self.dataset_extractor(topk_start_index+i)
                topk_value, topk_index = getSum_top_predictions(self.model,batch,2)
                #topk_list.append(getSum_top_predictions(self.model,batch,1))
                topk_list.append(topk_value)
                topk_pred_list.append(topk_index)

            topk_models_list.append(topk_list)
            topk_prediction_list.append(topk_pred_list)

        # Preparing header and data in needed format
        column_names = [f"Label: {self.dataset[topk_start_index+i][1]}" for i in range(len(topk_models_list[0]))]
        row_names = [os.path.splitext(model_filename)[0] for model_filename in model_filenames]
        table_topk_np = np.array(topk_models_list)

        # TODO: maybe a method for converting index to label in other cases
        # NOTE: or maybe can be like this label will be visible whole time
        table_label_np = np.array(topk_prediction_list)


        table_label_fig = plot_table(table_label_np, row_names, column_names)
        table_label_fig.savefig(os.path.join(topk_path, f"topK_labels.png"),
                                dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(table_label_fig)


        table_topk_fig = plot_table(table_topk_np, row_names, column_names)
        table_topk_fig.savefig(os.path.join(topk_path, f"topK_predictions.png"),
                               dpi=300, facecolor='dimgray', bbox_inches='tight', pad_inches=0)
        plt.close(table_topk_fig)
        plt.close('all')



    def report_saliency(self,  model_filenames, saliency_start_index, saliency_range):
        '''
        creates directory and saves all saliency map of specified layer for every model
        over index + range
        :return:
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if saliency_range == 0:
            saliency_range = 1

        # Create list of figures with orignal image and grad image combined
        saliency_path = os.path.join(self.save_path, 'SaliencyMap')
        create_directory(saliency_path)
        for model_filename in model_filenames:
            model_path = os.path.join(saliency_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)

            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            saliency_list = []


            for i in range(saliency_range):
                batch, sample, label = self.dataset_extractor(saliency_start_index+i)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                _, saliency_map = SaliencyMap().analyse(model=self.model, original_image=img, single_batch=batch)
                # fig, ax = plt.subplots(facecolor='dimgray')
                # ax.imshow(saliency_map)
                # plt.imshow(saliency_map, facecolor='dimgray')
                # fig = plt.gcf()
                # saliency_list.append(fig)
                # plt.close()
                saliency_list.append(saliency_map)

            for i, saliency_img in enumerate(saliency_list):
                plt.imshow(saliency_img, cmap='gray')
                plt.axis('off')  # Turn off the axis
                plt.gca().set_axis_off()  # Turn off the axis lines
                plt.tight_layout(pad=0)  # Adjust the padding to zero to remove unnecessary space
                plt.savefig(os.path.join(model_path, f"saliency_{i}.png"),
                              dpi=50, bbox_inches='tight', pad_inches=0, facecolor='dimgray')
                plt.close()

    def report_grad(self, model_filenames, grad_start_index, grad_range,
                    target_layer='model.conv1'):
        '''
        creates directory and saves all gradCam images of specified layer for every model
        over index + range
        :return:
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if grad_range == 0:
            grad_range = 1

        # Create list of figures with orignal image and grad image combined
        grad_path = os.path.join(self.save_path, 'GradCAM')
        create_directory(grad_path)
        for model_filename in model_filenames:
            model_path = os.path.join(grad_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            grad_list = []


            for i in range(grad_range):
                batch, sample, label = self.dataset_extractor(grad_start_index+i)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                _, grad_cam = GradCAM().analyse(model=self.model, original_image=img,
                                                single_batch=batch, target_layer=target_layer)
                grad_list.append(grad_cam)

            for i, grad_img in enumerate(grad_list):
                grad_img.save(os.path.join(model_path, f"grad_{i}.png"),
                              dpi=(30, 30))
                grad_img.close()

    def report_saliency_and_original(self, model_filenames, saliency_start_index, saliency_range):
        '''
        creates directory, displays and saves all gradCam images of every convolutional layer for every model
        over index + range
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if saliency_range == 0:
            saliency_range = 1

        # Create list of figures with orignal image and grad image combined
        saliency_path = os.path.join(self.save_path, 'Original_Saliency')
        create_directory(saliency_path)
        for model_filename in model_filenames:
            model_path = os.path.join(saliency_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)

            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            saliency_list = []

            for i in range(saliency_range):
                batch, sample, label = self.dataset_extractor(saliency_start_index+i)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                img_tensor, saliency = SaliencyMap().analyse(model=self.model, original_image=img, single_batch=batch)
                plot_obj = plot_original_vs_observation(img_as_tensor=img_tensor, result=saliency,
                                                        text="The Image and Its Saliency Map")
                saliency_list.append(plot_obj)
                # saliency_list.extend(self.grad_all(saliency_start_index+i))
                #
                # img_tensor, saliency = SaliencyMap().analyse(model=model, original_image=img, single_batch=single_batch)
                # plot_obj = plot_original_vs_observation(img_as_tensor=img_tensor, result=saliency,
                #                              text="The Image and Its Saliency Map")

            # save figures in reporting folder
            for i, saliency_plt in enumerate(saliency_list):
                plt.axis('off')  # Turn off the axis
                plt.gca().set_axis_off()  # Turn off the axis lines
                plt.tight_layout(pad=0)  # Adjust the padding to zero to remove unnecessary space
                saliency_plt.savefig(os.path.join(model_path, f"original_saliency_{i}.png"),
                                 dpi=30, bbox_inches='tight', pad_inches=0, facecolor='dimgray')
                plt.close(saliency_plt)


    def report_grad_and_original(self, model_filenames, grad_start_index, grad_range):
        '''
        creates directory, displays and saves all gradCam images of every convolutional layer for every model
        over index + range
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if grad_range == 0:
            grad_range = 1

        # Create list of figures with orignal image and grad image combined
        grad_path = os.path.join(self.save_path, 'Original_GradCAM')
        create_directory(grad_path)
        for model_filename in model_filenames:
            model_path = os.path.join(grad_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)

            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            grad_list = []

            for i in range(grad_range):
                grad_list.extend(self.grad_all(grad_start_index+i))

            # save figures in reporting folder
            for i, grad_plt in enumerate(grad_list):
                plt.axis('off')  # Turn off the axis
                plt.gca().set_axis_off()  # Turn off the axis lines
                plt.tight_layout(pad=0)  # Adjust the padding to zero to remove unnecessary space
                grad_plt.savefig(os.path.join(model_path, f"original_grad_{i}.png"),
                                 dpi=30, bbox_inches='tight', pad_inches=0, facecolor='dimgray')
                plt.close(grad_plt)


    def load_model_path_from_path(self, path):
        all_files = os.listdir(path)

        # Filter out the .pth files
        return [file for file in all_files if file.endswith('.pth')]

    def adapt_trainer_configs_to_analyzer(self):
        # change path of trainer
        self.trainer.changePaths(self.save_path)

        self.trainer.setModelName(self.name)

    # TODO: adversial block runter verschieben, sodass erst ausgeführt wenn im else block
    # TODO: evtl. bei großer Bildgenerierung von topK z.B. sollen mehrere Bilder erstellt werden
    def get_analysis_method(self, method, params, model_filenames):
        '''
        This method decides which analysis method will be called
        :param method: string of a alias for the method to be called
        :return: returns the evaluation results
        '''
        if method == 'epsilon_distance' and params.get("enabled", False) is True:

            histW, histZ, thrshW, thrshZ = self.trainer.getEpsilonResults()
            return self.eval_epsilon_distances(histW, histZ, thrshW, thrshZ)
        elif method == 'accuracy_testset' and params.get("enabled", False) is True:

            loss = self.trainer.getLossFunction()
            loader = self.trainer.getTestLoader()
            self.report_accuracy(model_filenames, loader, loss, titel=params.get('titel', 'test'))
        elif method == 'accuracy_adversarial' and params.get("enabled", False) is True:

            # init vars
            adv_titel = params.get("adv_titel", None)
            orig_titel = params.get("orig_titel", None)
            adv_path = params.get("adv_dataset_path", None)
            orig_path = params.get("orig_dataset_path", None)
            batch_size = params.get("batch_size", None)
            shuffle = params.get("shuffle", None)

            # check variables and run report test
            if (adv_path is None or orig_path is None or batch_size is None or
                    shuffle is None or orig_titel is None or adv_titel is None):
                logger.critical(f"Params for adversarial tests are not properly injected")
                return
            else:
                adv_dataset = self.datahandler.create_imageFolder_dataset(adv_path)
                orig_dataset = self.datahandler.create_imageFolder_dataset(orig_path)
                adv_dataloader = self.trainer.createCustomDataloader(adv_dataset,
                                                                     batch_size=batch_size, shuffle=shuffle)
                orig_dataloader = self.trainer.createCustomDataloader(orig_dataset,
                                                                      batch_size=batch_size, shuffle=shuffle)
                loss = self.trainer.getLossFunction()
                self.report_accuracy(model_filenames, orig_dataloader, loss, titel=orig_titel)
                self.report_accuracy(model_filenames, adv_dataloader, loss, titel=adv_titel)
        elif method == 'grad_and_original' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path)

            # init core params and check their status
            grad_start_index = params.get("grad_start_index", None)
            grad_range = params.get("grad_range", None)

            if grad_start_index is None or grad_range is None:
                logger.critical(f"Params for grad_and_original report are not properly injected")
                return
            else:
                self.report_grad_and_original(model_filenames, grad_start_index, grad_range)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'grad' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path)

            # init core params and check their status
            grad_start_index = params.get("grad_start_index", None)
            grad_range = params.get("grad_range", None)
            target_layer = params.get("target_layer", None)

            if grad_start_index is None or grad_range is None or target_layer is None:
                logger.critical(f"Params for grad report are not properly injected")
                return
            else:
                self.report_grad(model_filenames, grad_start_index, grad_range, target_layer=target_layer)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'saliency_and_original' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path)

            # init core params and check their status
            saliency_start_index = params.get("saliency_start_index", None)
            saliency_range = params.get("saliency_range", None)

            if saliency_start_index is None or saliency_range is None:
                logger.critical(f"Params for saliency_and_original report are not properly injected")
                return
            else:
                self.report_saliency_and_original(model_filenames, saliency_start_index, saliency_range)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'saliency' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path)

            # init core params and check their status
            saliency_start_index = params.get("saliency_start_index", None)
            saliency_range = params.get("saliency_range", None)

            if saliency_start_index is None or saliency_range is None:
                logger.critical(f"Params for saliency report are not properly injected")
                return
            else:
                self.report_saliency(model_filenames, saliency_start_index, saliency_range)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'topk_accuracy' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path)

            # init core params and check their status
            topk_start_index = params.get("topk_start_index", None)
            topk_range = params.get("topk_range", None)

            if topk_start_index is None or topk_range is None:
                logger.critical(f"Params for saliency report are not properly injected")
                return
            else:
                self.report_topk_accuracy(model_filenames, topk_start_index, topk_range)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'pruning' and params.get("enabled", False) is True:

            # creates a paper about pruning stats
            self.report_pruning(model_filenames)




    # ============================ Note: end of automatic report generation block


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

        layer_rows, zeros_table = self.zero_ratio_table_format()

        plot_model_comparison_with_table(input_images, model_outputs, zeros_table, layer_rows, self.model_name_list)

        model_comparison_table(table_data=topk_predictions,
                               row_labels=['Image 1', 'Image 2', 'Image 3'], col_labels=self.model_name_list)

        test_list = self.accuracy_table_format(**kwargs)

        model_comparison_table(table_data=test_list, row_labels=['accuracy %', 'accuracy'],
                               col_labels=self.model_name_list)

    # Note: this one is case specific
    def accuracy_table_format(self, **kwargs):
        test_list_perc = []
        test_list_abs = []
        for model in self.model_list:
            correct_classified, dataset_length, percentage = self.test(model, **kwargs)
            test_list_perc.append(percentage)
            test_list_abs.append(correct_classified)

        return [test_list_perc, test_list_abs]

    def zero_ratio_table_format(self):
        zeros_table = list()
        layer_rows = None
        for model in self.model_list:
            pruning_dict = pruningCounter(model)
            layer_name, metric, layer_zero_percentage = self._pruning_data_split(pruning_dict)
            layer_rows = layer_name
            zeros_table.append([str(zero_percentage) for zero_percentage in layer_zero_percentage[0]])

        return layer_rows, zeros_table

    def _pruning_data_split(self, pruning_dict):
        layer_names = list()
        layer_pruning_rates = list()
        layer_zero_params = list()
        layer_total_params = list()

        metric = ['zero_params', 'total_params', 'zero_ratio']

        for key, value in pruning_dict.items():
            layer_names.append(key)
            layer_zero_params.append(value['zero_params'])
            layer_total_params.append(value['total_params'])
            layer_pruning_rates.append(value['zero_ratio'])

        return layer_names, metric, (layer_zero_params, layer_total_params, layer_pruning_rates)

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
        correct_classified = 0
        dataset_length = len(test_loader.dataset)
        test_loader = test_loader
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_classified += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= dataset_length

        percentage = 100. * correct_classified / dataset_length

        logger.info(f'\nTest set: Average loss: {test_loss:.4f},'
                    f' Accuracy: {correct_classified}/{dataset_length}'
                    f' ({percentage:.0f}%)')

        return correct_classified, dataset_length, percentage

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
        plot_list = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):

                img_tensor, grad_cam = GradCAM().analyse(model=model, original_image=original_image,
                                  single_batch=single_batch, target_layer=name)
                plt_obj = plot_original_vs_observation(img_as_tensor=img_tensor, result=grad_cam,
                                             text=f'Gradient CAM for layer: {name}')
                plot_list.append(plt_obj)

        return plot_list


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
        grad_list = self.gradCam_all_layers(self.model, original_image=img, single_batch=batch)
        return grad_list

    def eval_epsilon_distances(self, epsilon_listW, epsilon_listZ,
                               epsilon_threshold_W, epsilon_threshold_Z):

        if ((not epsilon_listW and all(isinstance(item, float) for item in epsilon_listW)
                or isinstance(epsilon_threshold_W, float) is False)
                or (not epsilon_listZ and all(isinstance(item, float) for item in epsilon_listZ)
                or isinstance(epsilon_threshold_Z, float) is False)):
            logger.error("Not able to plot epsilon distance graph because either the input lists are\n"
                         "empty or the values inside are not of type float. Another problem could be\n"
                         "the threshold values are not of type float.")
            return

        epsilon_symbol = '\u03B5'
        plt_obj = plot_float_lists_with_thresholds(epsilon_listW, epsilon_listZ,
                                         f'{epsilon_symbol}-Distance W',
                                         f'{epsilon_symbol}-Distance Z',
                                         epsilon_threshold_W, epsilon_threshold_Z,
                                         f'{epsilon_symbol}-Threshold W',
                                         f'{epsilon_symbol}-Threshold Z',
                                         f'{epsilon_symbol}-Distances over ADMM-Iterations')

        if self.save is True:
            if os.path.isdir(self.save_path) is True:
                plt_obj.savefig(os.path.join(self.save_path, "epsilon_threshold.png"))
                logger.info(f"epsilon_threshold.png was saved at directory: {self.save_path}")
                plt.close(plt_obj)
                return plt_obj
            else:
                logger.error(f"SavePath was not a valid path. save_path={self.save_path}")
                plt.close(plt_obj)
                return None


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


