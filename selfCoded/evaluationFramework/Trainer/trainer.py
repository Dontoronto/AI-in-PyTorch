from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self, model):
        '''

        :param model: neuronal model
        :param dataloaderConfig: arguments for DataLoader Class saved as dict
        '''
        self.model = model
        self.dataloaderConfig = None
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

    def setDataLoaderSettings(self, kwargs: dict):
        '''
        sets custom Dataloader configuration
        '''
        self.dataloaderConfig = kwargs

