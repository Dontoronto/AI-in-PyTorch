# main.py
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
    # _model.load_state_dict(torch.load("models/LeNet/raw_LeNet.pth")) #['model_state_dict']
    Model = modelWrapper.ModelWrapper(_model)
    Model.load_state_dict(torch.load("models/LeNet/raw_LeNet.pth"))


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
    #Trainer.train(test=True)
    Analyzer = analyzer.Analyzer(Model, DataHandler)
    Analyzer.setDataset(DataHandler.loadDataset(testset=True))
    Analyzer.runtest()

    #Trainer.layerArchitectureExtractor()

    # ['layer1.0.conv1.weight']

    #torch.save(Model.state_dict(), 'model_weights.pth')



if __name__ == '__main__':
    main()

#%%

#%%
