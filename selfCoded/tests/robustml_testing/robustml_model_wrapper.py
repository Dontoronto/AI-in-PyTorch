import robustml

class RobustMLResNet18(robustml.model.Model):
    def __init__(self, model, dataset, threat_model):
        self._model = model
        self._dataset = dataset
        self._threat_model = threat_model

    def predict(self, x):
        # Assuming x is a PyTorch tensor
        logits = self._model(x)
        return logits.argmax(dim=1).item()

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model