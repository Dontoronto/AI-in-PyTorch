# main.py
import sys, os
sys.path.append(os.getcwd())
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader
import torch
import numpy as np
import torchvision.transforms as T

import modelWrapper
import dataHandler
import configurator

from Trainer.trainerFactory import TrainerFactory
from SharedServices.logging_config import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)


def main():
    # Step 1: Initialize model with the best available weights
    _weights = ResNet101_Weights.IMAGENET1K_V1
    _model = resnet101(weights=_weights)
    Model = modelWrapper.ModelWrapper(_model)
    Model.eval()


    Configurator = configurator.Configurator()
    DataHandler = dataHandler.DataHandler(Configurator)

    DataHandler.setTransformer(_weights.transforms())
    #DataHandler.setTransformer(Configurator.loadTransformer())

    # img = DataHandler.loadImage("testImages/tisch_v2.jpeg")




    # for name, weight in Model.state_dict().items():
    #     print(f"{name}: {type(weight)}")

    Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    Trainer.setDataLoaderSettings(Configurator.loadDataloaderConfig())
    #logger.critical(Configurator.loadConfigFromRegistry("admm_settings"))
    Trainer.setADMMArchitectureConfig(Configurator.loadConfigFromRegistry("admm_model_architecture"))
    #Trainer.train(test=False)
    Trainer.setADMMConfig(Configurator.loadConfigFromRegistry("admm_settings"))

    # Iterate over model parameters
    # for name, param in Model.named_parameters():
    #     logger.info(f"Layer Name: {name} Parameter Tensor Type: {type(param.data)} Gradients Tensor Type: {type(param.grad)}")

    #Trainer.layerArchitectureExtractor()

    # for name, module in Model.named_modules():
    #     logger.info(f"Layer Name: {name}  Module Type: {type(module)} ")


    # # Access gradients of a specific layer
    # for name, parameter in Model.named_parameters():
    #     if parameter.requires_grad:
    #         #print(f"{name} gradients:", parameter.grad)
    #         return

    #logger.critical(getattr(Model, "model.conv1.weight"))
    #logger.critical(Model._modules["model.layer1.0.conv1.weight"])
    # logger.critical(Model.state_dict()['model.layer4.2.bn3.weight'])
    # ['layer1.0.conv1.weight']

    #torch.save(Model.state_dict(), 'model_weights.pth')



    # count = 0
    # for batch, labels in dataloader:
    #     count +=1
    #     if count == 2:
    #         break
    #     with torch.no_grad():
    #         if len(batch) == 1:
    #             # Step 4: Use the model and print the predicted category
    #             prediction = Model(batch.clone().detach()).squeeze(0).softmax(0)
    #             class_id = prediction.argmax().item()
    #             score = prediction[class_id].item()
    #             category_name = _weights.meta["categories"][class_id]
    #             print(f"{category_name}: {100 * score:.1f}%")
    #         else:
    #             # Assuming 'Model' is your model instance and 'batch' is your input batch of tensors
    #             predictions = Model(batch.clone().detach())
    #             dataset_image = DataHandler.preprocessBackwardsBatched(batch)
    #             for i in dataset_image:
    #                 i.show()
    #
    #             softmax_predictions = torch.softmax(predictions, dim=1)  # Apply softmax on the correct dimension
    #
    #             for i, prediction in enumerate(softmax_predictions):
    #                 class_id = prediction.argmax().item()
    #                 score = prediction[class_id].item()
    #                 category_name = _weights.meta["categories"][class_id]
    #                 print(f"Tensor {labels[i]}: {category_name}: {100 * score:.1f}%")






if __name__ == '__main__':
    main()
