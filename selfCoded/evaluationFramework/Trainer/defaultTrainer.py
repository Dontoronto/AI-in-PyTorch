from .trainer import Trainer
from torch.utils.data import DataLoader

import torch
import logging
logger = logging.getLogger(__name__)


# TODO: Abhängigkeiten noch nicht fix
# TODO: Analyzer evtl. übergeben mit Preconfigured settings oder
# TODO: DefaultTrainer(..,Analyzer.defaultTrainingMode() -> "Configured Analyzer",..)

# TODO: extra variable per __init__ übergeben wo seperate Dataloader Optionen sind, evtl. TrainerConfig.json erstellen

# TODO: falls man andere Loss function benutzt kann man hier Dataset anpassen sodass target_transformiert wird
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


        logger.info("Trainer was configured")
        logger.info("Epochs: " + str(self.epoch))
        logger.info("Optimizer: " + str(self.optimizer))
        logger.info("Lossfunction: " + str(self.loss))
        logger.info("DataHandler: " + str(self.DataHandler))
        pass

    def prepareDataset(self):
        self.dataset = self.DataHandler.loadDataset()
        logger.info("Dataset was prepared for Trainer class")


    def train(self):
        self.prepareDataset()
        if self.dataset is None:
            logger.warning("No DatasetConfigs were found in DataHandlerConfig.json")
            return
        dataloader = DataLoader(self.dataset,batch_size=16, shuffle=True)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):

            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 2 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")





        pass

    def test(self):
        pass