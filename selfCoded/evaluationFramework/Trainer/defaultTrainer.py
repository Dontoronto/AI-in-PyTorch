from .trainer import Trainer
from torch.utils.data import DataLoader

import torch
import logging
import functools
logger = logging.getLogger(__name__)


# TODO: Abhängigkeiten noch nicht fix
# TODO: Analyzer evtl. übergeben mit Preconfigured settings oder
# TODO: DefaultTrainer(..,Analyzer.defaultTrainingMode() -> "Configured Analyzer",..)

class DefaultTrainer(Trainer):
    def __init__(self,
                 model,
                 dataHandler,
                 loss,
                 optimizer,
                 epoch=1
                 ):
        super().__init__(model)
        self.optimizer = optimizer
        self.loss = loss
        self.DataHandler = dataHandler
        self.epoch = epoch
        self.dataset = None
        self.testset = None

        logger.info("Trainer was configured")
        logger.info("Epochs: " + str(self.epoch))
        logger.info("Optimizer: " + str(self.optimizer))
        logger.info("Lossfunction: " + str(self.loss))
        logger.info("DataHandler: " + str(self.DataHandler))
        pass

    def prepareDataset(self, testset=False):
        if testset is False and self.dataset is None:
            self.dataset = self.DataHandler.loadDataset()
            logger.info("Dataset was loaded in Trainer class")
        elif testset is True and self.testset is None:
            self.testset = self.DataHandler.loadDataset(testset=testset)
            logger.info("Testet was loaded in Trainer class")

    def checkModelOutputFeatures(self, sampleLabel):
        '''
        Inspecting the model to find the output size (amount of neurons)
        :param sampleLabel: just a label example of dataset
        :return: Bool of Dimensions of model and label matches
        '''
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):  # Checking for the last Linear layer
                last_linear_layer_name, last_linear_layer = name, module
        return last_linear_layer.out_features == sampleLabel.size(0)

    def getAmountModelOutputClasses(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):  # Checking for the last Linear layer
                last_linear_layer_name, last_linear_layer = name, module
        return last_linear_layer.out_features

    def checkLabelEncoding(self, SampleDataset):
        '''
        Generic Label Loss-function adapter function
        checks output format onehot/int of dataset __getitem__() output. If output doesn't match with loss function
        target_transform variable will be adapted that it works
        target_transform: You can give a method which will be executed to transform the input. When None the input
        won't be transformed
        '''
        modelOutputClasses = self.getAmountModelOutputClasses()
        if modelOutputClasses != SampleDataset.classes:
            # TODO: production mode is with lines below uncommented
            logger.critical("Model Output Neurons {} and Dataset classes {} "
                            "doesn't match".format(modelOutputClasses,len(SampleDataset.classes)))
            logger.critical("Training of Model is not possible")
            # raise ValueError("Model Output Neurons {} and Dataset classes {} "
            #                  "doesn't match".format(modelOutputClasses,len(self.dataset.classes)))
        first_label = SampleDataset[0][1]
        if isinstance(self.loss, torch.nn.CrossEntropyLoss):
            if isinstance(first_label, int):
                logger.info("Labels of Dataset are in the correct form to be processed with " + str(self.loss))
            elif SampleDataset.target_transform is None:
                SampleDataset.target_transform = labelHotEncodedToInt
                logger.warning("Labels of Dataset are transformed to Integer with target_transform of Dataset"
                            "to be processed with " + str(self.loss))
            else:
                SampleDataset.target_transform = None
                if isinstance(SampleDataset[0][1], int):
                    logger.warning("Labels of Dataset transformed to Integer deleted target_transform var of Dataset "
                                   "to be processed with " + str(self.loss))
                else:
                    logger.critical("tried to change transformation of label for using loss function still not integer")
        elif isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            if isinstance(first_label, torch.Tensor):
                logger.info("Labels of Dataset are in the correct form to be processed with " + str(self.loss))
            else:
                SampleDataset.target_transform = functools.partial(labelIntToHotEncoded, size = modelOutputClasses)
                logger.warning("Labels of Dataset are transformed to one-hot-encoding "
                            "to be processed with " + str(self.loss))

    def preTrainingChecks(self):
        self.prepareDataset()
        if self.dataset is None:
            logger.warning("No DatasetConfigs were found in DataHandlerConfig.json")
            return
        self.checkLabelEncoding(self.dataset)

    def createDataLoader(self, sampleDataset):
        if self.dataloaderConfig is not None:
            logger.info("Created Dataloader with settings: " + str(self.dataloaderConfig))
            return DataLoader(sampleDataset, **self.dataloaderConfig)
        else:
            logger.warning("No Configs for Dataloader available, creating Dataloader with default arguments")
            return DataLoader(sampleDataset)




    def train(self, test = False):
        self.preTrainingChecks()
        dataloader = self.createDataLoader(self.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):

            # remove existing settings
            self.optimizer.zero_grad()

            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss(pred, y)

            # Backpropagation
            loss.backward()

            return
            # Apply optimization with gradients
            self.optimizer.step()

            if batch % 2 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")


        if test is True:
            self.test()

        pass




    def test(self):
        self.prepareDataset(testset=True)
        if self.testset is None:
            logger.warning("No DatasetConfigs were found in DataHandlerConfig.json")
            return
        self.checkLabelEncoding(self.testset)
        # TODO: evtl. andere Configs für Dataloader bei tests
        dataloader = self.createDataLoader(self.testset)
        self.model.eval()
        for batch, (X, y) in enumerate(dataloader):

            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss(pred, y)

            if batch % 2 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
                break
        pass



def labelIntToHotEncoded(y,size):
    return torch.zeros(size, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

def labelHotEncodedToInt(y):
    return torch.argmax(y, dim=0).item()