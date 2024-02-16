from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self, model):
        self.model = model
        #self.Configurator = configurator

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def prepareDataset(self):
        pass

    # def setConfigurator(self, configurator):
    #     self.Configurator = configurator
    #     pass
