# main.py
import argparse
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import sys, os
sys.path.append(os.getcwd())
import torch

import modelWrapper
import dataHandler
import configurator
import analyzer.analyzer as analyzer

from Trainer.trainerFactory import TrainerFactory
from SharedServices.logging_config import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# NOTE: currently we are testing with LeNet-model
from models.lenet import LeNet


def main():
    # Resnet settings work
    # _weights = ResNet101_Weights.IMAGENET1K_V1
    # _model = resnet101(weights=_weights)
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.eval()

    # LeNet Test
    _model = LeNet()
    Model = modelWrapper.ModelWrapper(_model)
    # Model.load_state_dict(torch.load("models/LeNet/raw_LeNet_v3.pth"))
    # Model.load_state_dict(torch.load("pruned_dynamic_mask_v2.pth"))
    # Model.load_state_dict(torch.load("retrained_mask_update_every_epoch.pth"))


    Configurator = configurator.Configurator()
    DataHandler = dataHandler.DataHandler(Configurator)

    #DataHandler.setTransformer(_weights.transforms())
    DataHandler.setTransformer(Configurator.loadTransformer())

    # img = DataHandler.loadImage("testImages/tisch_v2.jpeg")


    Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    Trainer.setDataLoaderSettings(Configurator.loadDataloaderConfig())
    Trainer.setSnapshotSettings(Configurator.loadSnapshotConfig())
    Trainer.setADMMArchitectureConfig(Configurator.loadConfigFromRegistry("admm_model_architecture"))
    Trainer.setADMMConfig(Configurator.loadConfigFromRegistry("admm_settings"))
    Trainer.train(test=True)
    # torch.save(Model.state_dict(), 'retrained_dynamic_mask_v3.pth')
    # test_loader = Trainer.getTestLoader()
    # loss_func = Trainer.getLossFunction()
    # Analyzer = analyzer.Analyzer(Model, DataHandler)
    #
    # Analyzer.setDataset(DataHandler.loadDataset(testset=True))
    # Analyzer.run_single_model_test(0, test_end_index=None, test_loader=test_loader, loss_func=loss_func)
    #
    # Model.load_state_dict(torch.load("retrained_dynamic_mask_v2.pth"))
    # Analyzer.setModel(Model)
    # Analyzer.run_single_model_test(0, test_end_index=3, test_loader=test_loader, loss_func=loss_func)

    # TODO: überlegen wie man schön und geordnet Models hochladen kann und sie testen kann
    # TODO: Ordner wird benötigt oder irgendwas damit man strukturiert die Models speichert

    #Trainer.layerArchitectureExtractor()

    # ['layer1.0.conv1.weight']

    #torch.save(Model.state_dict(), 'model_weights.pth')



if __name__ == '__main__':
    main()

#%%

#%%
