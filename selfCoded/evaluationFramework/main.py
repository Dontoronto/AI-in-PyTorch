# main.py
import argparse
import warnings
import time

warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import sys, os
sys.path.append(os.getcwd())
import torch

import modelWrapper
import dataHandler
import configurator
import analyzer.analyzer as analyzer

from Trainer.trainerFactory import TrainerFactory
from SharedServices.modelArchitectureExtractor import ModelArchitectureExtractor
from SharedServices.logging_config import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

from Trainer.admm_utils.tensorBuffer import TensorBuffer

# NOTE: currently we are testing with LeNet-model
from models.lenet import LeNet


def main():
    # Resnet settings work
    # _weights = ResNet101_Weights.IMAGENET1K_V1
    # _model = resnet101(weights=_weights)
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.eval()
    start_time = time.time()
    warnings.filterwarnings(action='once')

    # LeNet Test
    _model = LeNet()
    Model = modelWrapper.ModelWrapper(_model)
    Model.load_state_dict(torch.load("LeNet_admm_train.pth"))
    # Model.load_state_dict(torch.load("LeNet_epsiolon_test_admm_retrain.pth"))
    Model.eval()
    # Model.load_state_dict(torch.load("LeNet_epsiolon_test_admm_retrain.pth"))

    # Model.load_state_dict(torch.load("models/LeNet/raw_LeNet_v3.pth"))
    # Model.load_state_dict(torch.load("experiment/LeNet/v2/retrained_dynamic_mask_v2.pth"))
    # Model.load_state_dict(torch.load("LeNet_admm_train.pth"))
    # Model.load_state_dict(torch.load("LeNet_epsiolon_test_admm_admm.pth"))


    Configurator = configurator.Configurator()
    DataHandler = dataHandler.DataHandler(Configurator)

    #DataHandler.setTransformer(_weights.transforms())
    DataHandler.setTransformer(Configurator.loadTransformer())

    # img = DataHandler.loadImage("testImages/tisch_v2.jpeg")


    # ------------- Note: test of adv

    Analyzer = analyzer.Analyzer(Model, DataHandler)
    Analyzer.init_adversarial_environment()
    Analyzer.set_threat_model_config(Configurator.loadConfigFromRegistry("adversarial_threat_model"))
    Analyzer.set_provider_config(Configurator.loadConfigFromRegistry("adversarial_provider"))
    Analyzer.set_attack_type_config(Configurator.loadConfigFromRegistry("adversarial_attacks"))
    Analyzer.select_attacks_from_config(0, 1)
    Analyzer.enable_saving("experiment/adversarial_images")

    test1 = Analyzer.start_adversarial_evaluation(0, 5)
    print(f"First evaluation:")
    print(test1)

    # ------------- Note: end test of adv

    # Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    # Trainer.setDataLoaderSettings(Configurator.loadDataloaderConfig())
    # Trainer.setSnapshotSettings(Configurator.loadSnapshotConfig())
    # Trainer.setADMMArchitectureConfig(Configurator.loadConfigFromRegistry("admm_model_architecture"))
    # Trainer.setADMMConfig(Configurator.loadConfigFromRegistry("admm_settings"))
    #
    # Trainer.train(test=False, onnx_enabled=False, tensor_buffering=True)
    #
    # histW = Trainer.getHistoryEpsilonW()
    # histZ = Trainer.getHistoryEpsilonZ()
    # thrshW = Trainer.epsilon_W
    # thrshZ = Trainer.epsilon_Z
    # Analyzer = analyzer.Analyzer(Model, DataHandler)
    # Analyzer.eval_epsilon_distances(histW, histZ, thrshW, thrshZ)
    # #
    # TensorBuffer.create_two_matrix_gif('experiment/data/frames_w',
    #                                    'experiment/data/frames_z',
    #                                    'experiment/data/handlerGIF.gif')
    #
    # TensorBuffer.create_two_matrix_gif('experiment/data/frames_w2',
    #                                    'experiment/data/frames_z2',
    #                                    'experiment/data/handlerGIF2.gif')
    #
    # TensorBuffer.create_two_matrix_gif('experiment/data/frames_w3',
    #                                    'experiment/data/frames_z3',
    #                                    'experiment/data/handlerGIF3.gif')
    #
    # TensorBuffer.create_two_matrix_gif('experiment/data/frames_w4',
    #                                    'experiment/data/frames_z4',
    #                                    'experiment/data/handlerGIF4.gif')
    #
    # TensorBuffer.create_two_matrix_gif('experiment/data/frames_w5',
    #                                    'experiment/data/frames_z5',
    #                                    'experiment/data/handlerGIF5.gif')
    #
    # TensorBuffer.create_two_matrix_gif('experiment/data/frames_w6',
    #                                    'experiment/data/frames_z6',
    #                                    'experiment/data/handlerGIF6.gif')
    #
    # logger.critical(f"Time for algo is: {time.time()-start_time}")
    #
    # # Note: this is just for visualization
    # test_loader = Trainer.getTestLoader()
    # loss_func = Trainer.getLossFunction()
    # Analyzer = analyzer.Analyzer(Model, DataHandler)
    #
    # Analyzer.setDataset(DataHandler.loadDataset(testset=True))
    #
    # Model.load_state_dict(torch.load("LeNet_admm_train.pth"))
    # Analyzer.add_model(Model, "Default Model")
    # # Analyzer.setModel(Model)
    # #Analyzer.run_single_model_test(101, test_end_index=None, test_loader=test_loader, loss_func=loss_func)
    # #Analyzer.grad_all(0)
    #
    # Model.load_state_dict(torch.load("LeNet_epsiolon_test_admm_admm.pth"))
    # # Analyzer.setModel(Model)
    # Analyzer.add_model(Model, "ADMM Model")
    #
    #
    # Model.load_state_dict(torch.load("LeNet_epsiolon_test_admm_retrain.pth"))
    # # Analyzer.setModel(Model)
    # Analyzer.add_model(Model, "Retrained Model")
    #
    # #Analyzer.compare_models(10, 11)
    # Analyzer.runCompareTest(10, test_end_index=12, target_layer='model.conv1',
    #                         loss_func=loss_func, test_loader=test_loader)
    # Analyzer.setModel(Model)
    # Analyzer.grad_all(22)


    # TODO: überlegen wie man schön und geordnet Models hochladen kann und sie testen kann
    # TODO: Ordner wird cobenötigt oder irgendwas damit man strukturiert die Models speichert




if __name__ == '__main__':
    main()

#%%

#%%
