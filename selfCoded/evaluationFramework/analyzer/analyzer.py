import copy
import logging
import os, sys
import threading

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
from .featureMaps import featureMap
from .activationMaps.scoreCAM import ScoreCAM
from .activationMaps import cic

from .measurement.sparseMeasurement import pruningCounter
from .measurement.topPredictions import show_top_predictions, getSum_top_predictions, calculate_topk_accuracy
from .measurement.distributionDensity import (calculate_distribution_density, plot_distribution_density,
                                              calculate_optimal_density_range)

from .plotFuncs.plots import (plot_original_vs_observation, plot_model_comparison,
                              plot_model_comparison_with_table, model_comparison_table,
                              plot_float_lists_with_thresholds,
                              combine_plots_vertically,
                              plot_table)

from .evaluationMapsStrategy import EvaluationMapsStrategy

from .utils import weight_export, copy_directory, create_directory, subsample

from .adversial import adversarialAttacker

from .mapper.analyzerMapper import AnalyzerConfigMapper

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from SharedServices.utils import copy_directory

#torch.set_default_device('cuda')


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

        self.adv_save_enabled = False
        self.adv_original_save_enabled = False
        self.adv_attack_selection = None
        self.adv_attack_selection_range = None
        self.adv_sample_range_start = None
        self.adv_sample_range_end = None
        self.adv_only_success_flag = False
        self.adv_shuffle = False

        self.cuda_enabled = False
        try:
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                torch.set_default_device('cuda')
                self.cuda_enabled = True
                print(f"Device= {device}")
        except Exception:
            device = None
            print("Failed to set device automatically, please try set_device() manually.")
            self.cuda_enabled = False

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

    def startTestrun(self, kwargs, run_training=True):

        self.adapt_trainer_configs_to_analyzer()

        if run_training is True:
            self.trainer.train(**kwargs)
        else:
            pass

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

    def report_adv_attack(self, model_filenames, titel='adversarial dynamic'):
        accuracy_path = os.path.join(self.save_path, 'Adversarial_Dynamic')
        create_directory(accuracy_path)

        adv_dynamic_list = list()
        adv_topk_dynamic_list = list()
        model_name_list = list()

        column_metric = ['Adversarial Success Ratio', 'Adversarial Success', 'Dataset length',
                         'l2-norm overflow', 'No Perturbation Failure']

        adv_topk_column_metric = ['Adversarial Success Ratio', 'Top1', 'Top2', 'Top3', 'Top5', 'Dataset length']

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))

            dic_ratio, dic_success, dic_total, dict_above, dict_no_pert, dict_topk = self.start_adversarial_evaluation_preconfigured()


            for key in dic_ratio:
                model_attack = str(key) + ": " + str(model_name)
                adv_dynamic_list.append([dic_ratio[key], dic_success[key], dic_total[key],
                                         dict_above[key], dict_no_pert[key]])
                model_name_list.append(model_attack)
                adv_topk_dynamic_list.append([dic_ratio[key], *dict_topk[key], dic_total[key]])
            # accuracy_list.append([percentage, correct_classified, dataset_length])
            # model_name_list.append(model_name)

        combined = list(zip(model_name_list, adv_dynamic_list))
        combined.sort()
        sorted_strings, sorted_values = zip(*combined)
        model_name_list_sorted = list(sorted_strings)
        adv_dynamic_list_sorted = list(sorted_values)
        fig = plot_table(adv_dynamic_list_sorted, model_name_list_sorted, column_metric)
        fig.savefig(os.path.join(accuracy_path, f"{titel}_success.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)

        combined = list(zip(model_name_list, adv_topk_dynamic_list))
        combined.sort()
        sorted_strings, sorted_values = zip(*combined)
        model_name_list_sorted = list(sorted_strings)
        adv_topk_dynamic_list_sorted = list(sorted_values)
        fig = plot_table(adv_topk_dynamic_list_sorted, model_name_list_sorted, adv_topk_column_metric)
        fig.savefig(os.path.join(accuracy_path, f"{titel}_topk.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)

    def report_accuracy(self, model_filenames, test_loader, loss_func, titel='default_testset'):
        accuracy_path = os.path.join(self.save_path, 'Accuracy')
        create_directory(accuracy_path)

        accuracy_list = list()
        topk_accuracy_list = list()
        model_name_list = list()

        column_metric = ['Correct Classified Ratio', 'Correct Classified', 'Dataset length']
        topk_column_metric = ['Top1', 'Top2', 'Top3', 'Top5', 'Dataset length']

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))

            correct_classified, dataset_length, percentage, topk = self.test(self.model, test_loader=test_loader,
                                                                       loss_func=loss_func)

            accuracy_list.append([percentage, correct_classified, dataset_length])
            topk_accuracy_list.append([*topk,dataset_length])
            model_name_list.append(model_name)

        fig = plot_table(accuracy_list, model_name_list, column_metric)
        fig.savefig(os.path.join(accuracy_path, f"{titel}_accuracy.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)

        fig = plot_table(topk_accuracy_list, model_name_list, topk_column_metric)
        fig.savefig(os.path.join(accuracy_path, f"{titel}_topk_accuracy.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)

    def report_cic(self, model_filenames, test_loader):
        cic_path = os.path.join(self.save_path, "CIC")
        create_directory(cic_path)

        model_name_list = list()
        positive_list = list()
        negative_list = list()
        filter_sum_pos = list()
        filter_sum_neg = list()

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(
                torch.load(os.path.join(self.save_path, model_filename))
            )

            conv_layer_names_list, _filter_sum_pos, _filter_sum_neg, positive_scores, negative_scores = (
                cic.get_cic(
                    self.model,
                    test_loader
                )
            )

            # conv_layer_names_list, positive_scores, negative_scores = (
            #     cic.get_all_layers_cic(
            #         self.model,
            #         test_loader
            #     )
            # )

            positive_list.append(positive_scores)
            negative_list.append(negative_scores)
            model_name_list.append(model_name)

            # _, _filter_sum_pos, _filter_sum_neg = cic.get_all_single_layers_cic(
            #     self.model, input_shape, batch_size, test_loader
            # )

            filter_sum_pos.append(_filter_sum_pos)
            filter_sum_neg.append(_filter_sum_neg)

        # TODO: noch die einzelnen layerplots einfügen vllt noch gleich in der oberen schleife sammeln
        # TODO: evtl. auch eigene schleife weil hier layer gesammelt werden

        new_list_pos = cic.combine_single_value_tensors(positive_list)

        fig = cic.plot_cic_scatter_single_layer(
            model_name_list, new_list_pos, titel="CIC-Layers Plot: Positive Influence"
        )

        fig.savefig(
            os.path.join(cic_path, f"cic_positive_architecture.png"),
            dpi=300,
            facecolor="dimgray",
            bbox_inches="tight",
        )
        plt.close(fig)
        plt.close('all')

        pos = cic.display_table(model_name_list, new_list_pos, conv_layer_names_list)
        pos_norm = cic.display_table_norm(
            model_name_list, new_list_pos, conv_layer_names_list
        )
        cic.display_safe_table_new(pos, cic_path, f"cic_positive_architecture")
        cic.display_safe_table_new(pos_norm, cic_path, "cic_positive_architecture_norm")

        new_list_neg = cic.combine_single_value_tensors(negative_list)

        fig = cic.plot_cic_scatter_single_layer(
            model_name_list, new_list_neg, titel="CIC-Layers Plot: Negative Influence"
        )

        fig.savefig(
            os.path.join(cic_path, f"cic_negative_architecture.png"),
            dpi=300,
            facecolor="dimgray",
            bbox_inches="tight",
        )
        plt.close(fig)
        plt.close('all')

        neg = cic.display_table(model_name_list, new_list_neg, conv_layer_names_list)
        neg_norm = cic.display_table_norm(
            model_name_list, new_list_neg, conv_layer_names_list
        )
        cic.display_safe_table_new(neg, cic_path, f"cic_neg_architecture")
        cic.display_safe_table_new(neg_norm, cic_path, "cic_neg_architecture_norm")

        inverted_comb_norm = cic.display_table_combi_norm(model_name_list, new_list_pos, new_list_neg,
                                                          conv_layer_names_list)
        cic.display_safe_table_new(inverted_comb_norm, cic_path, f"cic_inverted_combi_norm")


        filter_list_pos = cic.combine_tensors(filter_sum_pos)

        for layer_tensor, layer_name in zip(filter_list_pos, conv_layer_names_list):
            fig = cic.plot_cic_scatter_single_layer(
                model_name_list,
                layer_tensor,
                titel=f"CIC-{layer_name}-Layer Plot: Positive Influence",
            )

            fig.savefig(
                os.path.join(cic_path, f"cic_pos_{layer_name}_plot.png"),
                dpi=300,
                facecolor="dimgray",
                bbox_inches="tight",
            )
            plt.close(fig)
            plt.close('all')

            pos_filter = cic.display_table(model_name_list, layer_tensor)
            pos_norm_filter = cic.display_table_norm(model_name_list, layer_tensor)
            cic.display_safe_table_new(pos_filter, cic_path, f"cic_pos_{layer_name}")
            cic.display_safe_table_new(
                pos_norm_filter, cic_path, f"cic_pos_norm_{layer_name}"
            )

        filter_list_neg = cic.combine_tensors(filter_sum_neg)

        for layer_tensor, layer_name in zip(filter_list_neg, conv_layer_names_list):
            fig = cic.plot_cic_scatter_single_layer(
                model_name_list,
                layer_tensor,
                titel=f"CIC-{layer_name}-Layer Plot: Negative Influence",
            )

            fig.savefig(
                os.path.join(cic_path, f"cic_neg_{layer_name}_plot.png"),
                dpi=300,
                facecolor="dimgray",
                bbox_inches="tight",
            )
            plt.close(fig)
            plt.close('all')

            neg_filter = cic.display_table(model_name_list, layer_tensor)
            neg_norm_filter = cic.display_table_norm(model_name_list, layer_tensor)
            cic.display_safe_table_new(neg_filter, cic_path, f"cic_neg_{layer_name}")
            cic.display_safe_table_new(
                neg_norm_filter, cic_path, f"cic_neg_norm_{layer_name}"
            )

    def report_noisy_accuracy(self, model_filenames, test_loader, loss_func, noise_ratio, mean, std, titel='noisy_testset'):
        accuracy_path = os.path.join(self.save_path, 'NoisyAccuracy')
        create_directory(accuracy_path)

        accuracy_list = list()
        topk_accuracy_list = list()
        model_name_list = list()

        column_metric = ['Correct Classified Ratio', 'Correct Classified', 'Dataset length']
        topk_column_metric = ['Top1', 'Top2', 'Top3', 'Top5', 'Dataset length']

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))

            correct_classified, dataset_length, percentage, topk = self.noisy_test(self.model, test_loader=test_loader,
                                                                       loss_func=loss_func, noise_ratio=noise_ratio,
                                                                             mean=mean, std=std)

            accuracy_list.append([percentage, correct_classified, dataset_length])
            topk_accuracy_list.append([*topk,dataset_length])
            model_name_list.append(model_name)

        fig = plot_table(accuracy_list, model_name_list, column_metric)
        fig.savefig(os.path.join(accuracy_path, f"{titel}_{noise_ratio}_ratio_accuracy.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)

        fig = plot_table(topk_accuracy_list, model_name_list, topk_column_metric)
        fig.savefig(os.path.join(accuracy_path, f"{titel}_topk_{noise_ratio}_ratio_accuracy.png"),
                    dpi=300, facecolor='dimgray', bbox_inches='tight')
        plt.close(fig)



    def report_pruning(self, model_filenames):

        pruning_path = os.path.join(self.save_path, 'PruningStats')
        create_directory(pruning_path)

        pruning_stats_list = list()
        density_range = (0,0)

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))

            pruning_dict = pruningCounter(self.model)
            layer_names, metric, weight_stats = self._pruning_data_split(pruning_dict)

            fig = plot_table(weight_stats, metric, layer_names)

            fig.savefig(os.path.join(pruning_path, f"{model_name}_pruning_state.png"),
                        dpi=300, facecolor='dimgray', bbox_inches='tight')
            plt.close(fig)

            # fig_density = self.density_evaluation(bins=450,log_scale=True)
            # fig_density.savefig(os.path.join(pruning_path, f"{model_name}_density_histogram.png"),
            #             dpi=300, facecolor='white', bbox_inches='tight')
            # plt.close(fig_density)
            density_range_measure = calculate_optimal_density_range(self.model)
            if density_range_measure[1] > density_range[1]:
                density_range = density_range_measure

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

        for model_filename in model_filenames:
            model_name = os.path.splitext(model_filename)[0]
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))

            if density_range[0] == 0:
                density_range = None

            fig_density = self.density_evaluation(density_range=density_range , log_scale=False)
            fig_density.savefig(os.path.join(pruning_path, f"{model_name}_density_histogram.png"),
                        dpi=300, facecolor='white', bbox_inches='tight')
            plt.close(fig_density)




    def report_topk_accuracy(self, model_filenames, topk_start_index, topk_range, topk_pred):

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
                topk_value, topk_index = getSum_top_predictions(self.model,batch, topk_pred)
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
                converted_saliency = saliency_img.cpu()
                plt.imshow(converted_saliency, cmap='gray')
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

    def report_scoreCAM(self, model_filenames, score_start_index, score_range,
                    target_layer='model.conv1'):
        '''
        creates directory and saves all gradCam images of specified layer for every model
        over index + range
        :return:
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if score_range == 0:
            score_range = 1

        # Create list of figures with orignal image and grad image combined
        score_path = os.path.join(self.save_path, 'score_cam')
        create_directory(score_path)
        for model_filename in model_filenames:
            model_path = os.path.join(score_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            score_cam_list = []

            for i in range(score_range):
                batch, sample, label = self.dataset_extractor(score_start_index+i)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                _, score_cam = ScoreCAM().analyse(model=self.model, original_image=img,
                                                  single_batch=batch, target_layer=target_layer)
                score_cam_list.append(score_cam)

            for i, grad_img in enumerate(score_cam_list):
                grad_img.save(os.path.join(model_path, f"scoreCAM_{i}.png"),
                              dpi=(30, 30))
                grad_img.close()

    def report_featureMap(self, model_filenames, feature_start_index, feature_range, target_layer):
        '''
        creates directory and saves all gradCam images of specified layer for every model
        over index + range
        :return:
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if feature_range == 0:
            feature_range = 1

        # Create list of figures with orignal image and grad image combined
        feature_path = os.path.join(self.save_path, 'feature_map')
        create_directory(feature_path)
        for model_filename in model_filenames:
            model_path = os.path.join(feature_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            feature_map_list = []

            for i in range(feature_range):
                batch, sample, label = self.dataset_extractor(feature_start_index+i)
                # _, score_cam = ScoreCAM().analyse(model=self.model, original_image=img,
                #                                   single_batch=batch, target_layer=target_layer)
                feature_map, layer_name = featureMap.extract_single_feature_map(self.model, batch, target_layer)
                feature_map_list.append(feature_map)

            def process_feature(semaphore, feature, target_layer, model_path, i):
                with semaphore:
                    feature_plot = featureMap.plot_single_feature_map(feature, target_layer)
                    feature_plot.savefig(os.path.join(model_path, f"{i}_imgfeatureMap_{target_layer}.png"), dpi=100)
                    plt.close(feature_plot)

            def run_in_parallel(feature_map_list, target_layer, model_path):
                threads = []
                semaphore = threading.Semaphore(2)  # Maximal 5 Threads gleichzeitig

                for i, feature in enumerate(feature_map_list):
                    t = threading.Thread(target=process_feature, args=(semaphore, feature, target_layer, model_path, i))
                    threads.append(t)
                    t.start()

                # Warte auf alle Threads, bis sie fertig sind
                for t in threads:
                    t.join()

            run_in_parallel(feature_map_list, target_layer, model_path)

            #for i, feature in enumerate(feature_map_list):
            #    feature_plot = featureMap.plot_single_feature_map(feature, target_layer)
            #    feature_plot.savefig(os.path.join(model_path, f"featureMap_{target_layer}_{i}.png"),
            #                  dpi=400)
            #    plt.close(feature_plot)

    def report_featureMap_allLayer(self, model_filenames, feature_start_index, feature_range):
        '''
        creates directory, displays and saves all gradCam images of every convolutional layer for every model
        over index + range
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if feature_range == 0:
            feature_range = 1

        # Create list of figures with orignal image and grad image combined
        feature_path = os.path.join(self.save_path, 'FeatureMap_allLayer')
        create_directory(feature_path)
        for model_filename in model_filenames:
            model_path = os.path.join(feature_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)

            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            total_feature_map_list = []

            for i in range(feature_range):
                batch, sample, label = self.dataset_extractor(feature_start_index+i)
                img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
                feature_maps, layer_names = featureMap.extract_all_feature_maps(self.model, batch)
                total_feature_map_list.append((feature_maps, layer_names))

            # for i in range(feature_range):
            #     feature_map_list.extend(self.grad_all(feature_start_index+i))

            def process_feature_maps(semaphore, feature_maps_list, layer_names_list, model_path, i):
                with semaphore:
                    feature_figures = featureMap.plot_all_feature_maps(feature_maps_list, layer_names_list)
                    for feature, lname in zip(feature_figures, layer_names_list):
                        feature.savefig(os.path.join(model_path, f"{i}_img_featureMap_{lname}.png"), dpi=100)
                        plt.close(feature)

            def run_in_parallel(total_feature_map_list, model_path):
                threads = []
                semaphore = threading.Semaphore(2)  # Maximal 5 Threads gleichzeitig

                for i, (feature_maps_list, layer_names_list) in enumerate(total_feature_map_list):
                    t = threading.Thread(target=process_feature_maps, args=(semaphore, feature_maps_list,
                                                                            layer_names_list, model_path, i))
                    threads.append(t)
                    t.start()

                # Warte auf alle Threads, bis sie fertig sind
                for t in threads:
                    t.join()

            run_in_parallel(total_feature_map_list, model_path)


        #for i, (feature_maps_list, layer_names_list) in enumerate(total_feature_map_list):
        #        feature_figures = featureMap.plot_all_feature_maps(feature_maps_list, layer_names_list)
        #        for feature, lname in zip(feature_figures,layer_names_list):
        #            #feature_plot = featureMap.plot_single_feature_map(feature, lname)
        #            feature.savefig(os.path.join(model_path, f"{i}_img_featureMap_{lname}.png"), dpi=100)
        #            plt.close(feature)


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

    def report_scoreCAM_and_original(self, model_filenames, score_start_index, score_range):
        '''
        creates directory, displays and saves all gradCam images of every convolutional layer for every model
        over index + range
        '''
        if self.check_dataset() is False:
            self.setDataset(self.datahandler.loadDataset(testset=True))

        # iterating and saving all gradcams to folder
        if score_range == 0:
            score_range = 1

        # Create list of figures with orignal image and grad image combined
        score_path = os.path.join(self.save_path, 'Original_ScoreCAM')
        create_directory(score_path)
        for model_filename in model_filenames:
            model_path = os.path.join(score_path, os.path.splitext(model_filename)[0])
            create_directory(model_path)

            self.model.load_state_dict(torch.load(os.path.join(self.save_path, model_filename)))
            score_list = []

            for i in range(score_range):
                score_list.extend(self.scoreCam_all(score_start_index+i))

            # save figures in reporting folder
            for i, score_plt in enumerate(score_list):
                plt.axis('off')  # Turn off the axis
                plt.gca().set_axis_off()  # Turn off the axis lines
                plt.tight_layout(pad=0)  # Adjust the padding to zero to remove unnecessary space
                score_plt.savefig(os.path.join(model_path, f"original_scoreCAM_{i}.png"),
                                 dpi=30, bbox_inches='tight', pad_inches=0, facecolor='dimgray')
                plt.close(score_plt)


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
            # return self.eval_epsilon_distances(histW, histZ, thrshW, thrshZ)
            # Check if any value is None
            if histW is None or histZ is None or thrshW is None or thrshZ is None:
                # Handle the case where any of the values is None, if necessary
                pass
            else:
                return self.eval_epsilon_distances(histW, histZ, thrshW, thrshZ)
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
                adv_dataset = self.datahandler.create_imageFolder_dataset(adv_path, adversarialTransformer=True)
                #adv_dataset = self.trainer.create_imageFolder_dataset(adv_path)
                orig_dataset = self.datahandler.create_imageFolder_dataset(orig_path, adversarialTransformer=True)
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
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

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
        elif method == 'feature_maps_all_layer' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

            # init core params and check their status
            feature_start_index = params.get("feature_start_index", None)
            feature_range = params.get("feature_range", None)

            if feature_start_index is None or feature_range is None:
                logger.critical(f"Params for featureMaps all Layers report are not properly injected")
                return
            else:
                #self.report_grad_and_original(model_filenames, feature_start_index, feature_range)
                self.report_featureMap_allLayer(model_filenames, feature_start_index, feature_range)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'scoreCAM_and_original' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

            # init core params and check their status
            score_start_index = params.get("score_start_index", None)
            score_range = params.get("score_range", None)

            if score_start_index is None or score_range is None:
                logger.critical(f"Params for scoreCAM_and_original report are not properly injected")
                return
            else:
                self.report_scoreCAM_and_original(model_filenames, score_start_index, score_start_index)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'grad' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

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
        elif method == 'feature_map' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

            # init core params and check their status
            feature_start_index = params.get("feature_start_index", None)
            feature_range = params.get("feature_range", None)
            target_layer = params.get("target_layer", None)

            if feature_start_index is None or feature_range is None or target_layer is None:
                logger.critical(f"Params for featureMap report are not properly injected")
                return
            else:
                self.report_featureMap(model_filenames, feature_start_index, feature_range, target_layer=target_layer)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'score_cam' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

            # init core params and check their status
            score_start_index = params.get("score_start_index", None)
            score_range = params.get("score_range", None)
            target_layer = params.get("target_layer", None)

            if score_start_index is None or score_range is None or target_layer is None:
                logger.critical(f"Params for ScoreCAM report are not properly injected")
                return
            else:
                self.report_scoreCAM(model_filenames, score_start_index, score_range, target_layer=target_layer)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'saliency_and_original' and params.get("enabled", False) is True:

            # check if adversarial attack is specified
            adv_dataset_path = params.get("adv_dataset_path", None)
            if adv_dataset_path is not None:
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

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
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

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
                self.dataset = self.datahandler.create_imageFolder_dataset(adv_dataset_path, adversarialTransformer=True)

            # init core params and check their status
            topk_start_index = params.get("topk_start_index", None)
            topk_range = params.get("topk_range", None)
            topk_pred = params.get("topk_pred", 2)

            if topk_start_index is None or topk_range is None:
                logger.critical(f"Params for saliency report are not properly injected")
                return
            else:
                self.report_topk_accuracy(model_filenames, topk_start_index, topk_range, topk_pred)

                # reset dataset if adv_dataset
                if adv_dataset_path is not None:
                    self.dataset = None
        elif method == 'pruning' and params.get("enabled", False) is True:

            # creates a paper about pruning stats
            self.report_pruning(model_filenames)
        elif method == 'accuracy_testset' and params.get("enabled", False) is True:

            loss = self.trainer.getLossFunction()
            loader = self.trainer.getTestLoader()
            self.report_accuracy(model_filenames, loader, loss, titel=params.get('titel', 'test'))
        elif method == 'cic_test' and params.get("enabled", False) is True:

            # init core params and check their status
            num_samples = params.get("num_samples", None)
            batch_size = params.get("batch_size", None)

            if num_samples is None or batch_size is None:
                logger.critical(f"Params for cic test are not properly injected")
                return
            else:
                loader = subsample(self.datahandler.loadDataset(testset=True), num_samples, batch_size,
                                            self.cuda_enabled)
                self.report_cic(model_filenames, loader)
                if self.cuda_enabled is True:
                    del loader
                    torch.cuda.empty_cache()
                    plt.close('all')

        elif method == 'adversarial_success' and params.get("enabled", False) is True:

            self.report_adv_attack(model_filenames, titel=params.get('titel', 'test'))

        elif method == 'noisy_accuracy' and params.get("enabled", False) is True:

            loss = self.trainer.getLossFunction()
            loader = self.trainer.getTestLoader()

            noise_ratio = params.get("noise_ratio", 0.2)
            noise_steps = params.get("noise_steps", 0)
            mean, std = self.datahandler.getNormalization_params()

            noise_ratio_steps = (1 - noise_ratio) / (noise_steps+1)

            for i in range(noise_steps+1):
                noise_ratio_mod = noise_ratio + i*noise_ratio_steps
                self.report_noisy_accuracy(model_filenames, loader, loss, noise_ratio_mod, mean, std,
                                           titel=params.get('titel', 'test'))




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
        if self.cuda_enabled is True:
            return batch.to('cuda'), sample.to('cuda'), label
        else:
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
        topk_correct = [0, 0, 0, 0]
        ks = [1, 2, 3, 5]
        dataset_length = len(test_loader.dataset)
        test_loader = test_loader
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_classified += pred.eq(target.view_as(pred)).sum().item()

                for i, k in enumerate(ks):
                    topk_correct[i] += calculate_topk_accuracy(output, target, k)

        test_loss /= dataset_length

        percentage = 100. * correct_classified / dataset_length
        topk_accuracy = [correct / dataset_length * 100 for correct in topk_correct]

        logger.info(f'\nTest set: Average loss: {test_loss:.4f},'
                    f' Accuracy: {correct_classified}/{dataset_length}'
                    f' ({percentage:.0f}%)')

        for i, k in enumerate(ks):
            logger.info(f"Top-{k} Accuracy: {topk_accuracy[i]:.2f}%")

        return correct_classified, dataset_length, percentage, topk_accuracy

    def noisy_test(self, model, test_loader, loss_func, noise_ratio, mean, std,  **kwargs):
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
        topk_correct = [0, 0, 0, 0]
        ks = [1, 2, 3, 5]
        dataset_length = len(test_loader.dataset)
        test_loader = test_loader
        mean_tensor = torch.tensor(mean).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std).view(1, -1, 1, 1)
        with torch.no_grad():
            for data, target in test_loader:
                noise_overlay = (torch.randn_like(data) - mean_tensor) / std_tensor * noise_ratio
                data += noise_overlay
                output = model(data)
                test_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_classified += pred.eq(target.view_as(pred)).sum().item()

                for i, k in enumerate(ks):
                    topk_correct[i] += calculate_topk_accuracy(output, target, k)

        test_loss /= dataset_length

        percentage = 100. * correct_classified / dataset_length
        topk_accuracy = [correct / dataset_length * 100 for correct in topk_correct]

        logger.info(f'\nTest set: Average loss: {test_loss:.4f},'
                    f' Accuracy: {correct_classified}/{dataset_length}'
                    f' ({percentage:.0f}%)')

        for i, k in enumerate(ks):
            logger.info(f"Top-{k} Accuracy: {topk_accuracy[i]:.2f}%")

        return correct_classified, dataset_length, percentage, topk_accuracy

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

    def scoreCam_all_layers(self, model, original_image, single_batch):
        '''
        :param model: this is the pytorch model
        :param original_image: this is the image in PIL format
        :param single_batch:  this is the model input as a single batch as tensor
        :return:
        '''
        plot_list = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                img_tensor, score_cam = ScoreCAM().analyse(model=model, original_image=original_image,
                                                           single_batch=single_batch, target_layer=name)
                plt_obj = plot_original_vs_observation(img_as_tensor=img_tensor, result=score_cam,
                                                       text=f'Score-CAM for layer: {name}')
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

    def scoreCam_all(self, test_index):
        batch, sample, label = self.dataset_extractor(test_index)
        img = self.datahandler.preprocessBackwardsNonBatched(tensor=sample)
        score_cam_list = self.scoreCam_all_layers(self.model, original_image=img, single_batch=batch)

        return score_cam_list

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

    def set_adv_dataset_generation_settings(self):
        '''
        adv settings loading via AnalyzerConfig.json
        '''
        self.adversarial_module.setAdvShuffle(self.adv_shuffle)
        if self.adv_save_enabled is True:
            adv_path = os.path.join(self.save_path, "adv_image_generation/adv_images")
            create_directory(adv_path)
            self.enable_adversarial_saving(adv_path)
        if self.adv_original_save_enabled is True:
            original_path = os.path.join(self.save_path, "adv_image_generation/original_images")
            create_directory(original_path)
            self.enable_original_saving(original_path)
        if self.adv_attack_selection is not None:
            if isinstance(self.adv_attack_selection, int):
                self.adversarial_module.set_adv_only_success_flag(self.adv_only_success_flag)
                if (self.adv_attack_selection_range is not None and
                        self.adv_original_save_enabled is False and
                        self.adv_save_enabled is False):
                    self.select_attacks_from_config(self.adv_attack_selection,
                                                    self.adv_attack_selection_range)
                else:
                    self.select_attacks_from_config(self.adv_attack_selection, 1)

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

    def getSingleAttack(self, model, attack_name, **kwargs):
        return self.adversarial_module.getSingleAttack(model, attack_name, **kwargs)


    def start_adversarial_evaluation_preconfigured(self):
        '''
        start mechanism if adv_configs were set via AnalyzerConfig.json
        :return: result of true positiv classification of adv samples in between Lp norm
        '''
        self.adversarial_module.enable_threshold_saving()
        if self.adv_sample_range_start is not None or self.adv_sample_range_end is not None:
            if (isinstance(self.adv_sample_range_start, int) and isinstance(self.adv_sample_range_end, int) and
                    self.adv_sample_range_start < self.adv_sample_range_end):
                return self.adversarial_module.evaluate(self.adv_sample_range_start, self.adv_sample_range_end)


    def enable_adversarial_saving(self, path):
        self.adversarial_module.enableAdversarialSaveMode(True)
        self.adversarial_module.setAdversarialSavePath(path)

    def enable_original_saving(self, path):
        self.adversarial_module.enableOriginalSaveMode(True)
        self.adversarial_module.setOriginalSavePath(path)

    # -----------------------------

    # ------------ Density Analysis
    def density_evaluation(self, bins=None, density_range=None, log_scale=False):
        '''
        Berechnet und stellt das Verteilungsdiagramm der Modellgewichte dar

        - bins (int): Die Anzahl der Bins für das Histogramm.
        - density_range (tuple): Ein Tupel (min, max) zur Beschränkung des Wertebereichs.
        - log_scale (bool): Wenn True, wird die y-Achse logarithmisch skaliert.
        '''
        density, bin_edges = calculate_distribution_density(self.model, bins, density_range)
        fig_density = plot_distribution_density(density, bin_edges, log_scale)
        return fig_density

    # -----------------------------


