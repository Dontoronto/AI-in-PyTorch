# main.py
from torchvision.models import resnet101, ResNet101_Weights
import torch

import modelWrapper
import dataHandler
import configurator

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

    # DataHandler.setTransformer(_weights.transforms())
    DataHandler.loadTransformer()

    img = DataHandler.loadImage("testImages/tisch_v2.jpeg")


    # TODO: Datahandler soll batching noch übernehmen, schauen wie man das mit Dataset Klasse vereinen kann
    batch = DataHandler.preprocess(img).unsqueeze(0)
    with torch.no_grad():
        # Step 4: Use the model and print the predicted category
        prediction = Model(batch.clone().detach()).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = _weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")






if __name__ == '__main__':
    main()
