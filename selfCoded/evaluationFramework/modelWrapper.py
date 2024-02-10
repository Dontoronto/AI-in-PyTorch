import torch
import torch.nn as nn


# TODO: Datahandler mit Architektur syncen und evtl. erneut sachen verschieben
class ModelWrapper(nn.Module):
    def __init__(self, _model):
        super(ModelWrapper, self).__init__()
        self.model = _model  # Instance of the pretrained ResNet model

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




# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()
#
# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)
#
# # Step 4: Use the model and print the predicted category
# prediction = model(batch.clone().detach()).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")