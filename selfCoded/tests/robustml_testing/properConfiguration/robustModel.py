import robustml
import placeholderThreatModel
import torchvision.transforms as transforms

class RobustModel(robustml.model.Model):
    '''
    Interface for a model (classifier).

    Besides the required methods below, a model should do a reasonable job of
    providing easy access to internals to make white box attacks easier. For
    example, a model using TensorFlow might want to provide access to the input
    tensor placeholder and the tensor representing the logits output of the
    classifier.
    '''
    def __init__(self, model, dataset_name, threat_model=None):
        self._model = model
        if threat_model == None:
            self._threat_model = placeholderThreatModel.ThreatModel()
        else:
            self._threat_model = threat_model

        if dataset_name == "CIFAR10":
            self._dataset = robustml.dataset.CIFAR10()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        '''
        Returns the label for the input x (as a Python integer).
        '''

        x_in = self.transform(x).unsqueeze(0)
        prediction = self._model(x_in).squeeze(0)
        class_id = prediction.argmax().item()
        return class_id

#%%
