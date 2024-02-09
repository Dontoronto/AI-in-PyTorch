import torch
import torch.nn as nn
import torchvision.models as models

# TODO: Datahandler mit Architektur syncen und evtl. erneut sachen verschieben
# TODO: schauen ob man hier die layer extrahieren kann, ob sie namen haben Standard herausfinden etc.
#       TODO: ModelWrapper soll sich wie das richtige Model verhalten
class ModelWrapper(nn.Module):
    def __init__(self, _model):
        super(ModelWrapper, self).__init__()
        self.model = _model  # Instance of the pretrained ResNet model

    # Example of an additional method
    def new_method(self):
        print("test of wrapper class")
        pass

    def forward(self, x):
        # Delegate the call to the ResNet model's forward method
        return self.model(x)



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