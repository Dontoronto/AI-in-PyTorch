import torch
import os
import torch.nn as nn
import json

import logging
logger = logging.getLogger(__name__)


# TODO: Datahandler mit Architektur syncen und evtl. erneut sachen verschieben
class ModelWrapper(nn.Module):
    def __init__(self, _model):
        super(ModelWrapper, self).__init__()
        self.model = _model  # Instance of the pretrained ResNet model
        logger.info("ModelWrapper was initialized")

    def forward(self, x):
        # Delegate the call to the ResNet model's forward method
        return self.model(x)

    def __getattr__(self, name: str):
        """
        Umleiten von Zugriffen auf Attribute, die nicht direkt in der Wrapper-Klasse definiert sind,
        an das PyTorch-Modell.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


    def __getitem__(self, key):
        # Forward item access to the model
        return self.model[key]


