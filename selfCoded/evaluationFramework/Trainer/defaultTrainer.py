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

    def checkLabelEncoding(self):
        # Nehmen Sie das erste Label, um die Dimension und den Wert zu überprüfen
        first_label = self.dataset[0][1]
        logger.critical(type(first_label))
        if isinstance(self.loss, torch.nn.CrossEntropyLoss):
            if isinstance(first_label, int):
                logger.info("Labels of Dataset are in the correct form to be processed with " + str(self.loss))
            else:
                self.dataset.target_transform = labelHotEncodedToInt
                logger.warning("Labels of Dataset are transformed to one-hot-encoding "
                            "to be processed with " + str(self.loss))
        elif isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            if isinstance(first_label, torch.Tensor):
                logger.info("Labels of Dataset are in the correct form to be processed with " + str(self.loss))
            else:
                logger.critical("check")
                self.dataset.target_transform = labelIntToHotEncoded
                logger.warning("Labels of Dataset are transformed to one-hot-encoding "
                            "to be processed with " + str(self.loss))



        #
        #
        # # Inspecting the model to find the output size
        # for name, module in self.model.named_modules():
        #     if isinstance(module, torch.nn.Linear):  # Checking for the last Linear layer
        #         last_linear_layer_name, last_linear_layer = name, module
        #
        # logger.critical(type(last_linear_layer.out_features))
        # exit()
        # # Überprüfen Sie, ob die Dimension der Labels mit der erwarteten One-Hot-Encoded-Dimension übereinstimmt
        # if first_label.ndim == 1 and ((first_label == 0) | (first_label == 1)).all():
        #     # Überprüfen Sie, ob genau ein Wert im Label 1 ist
        #     logger.critical("check")
        #     exit()
        #     is_one_hot = (first_label.sum() == 1)
        #     return is_one_hot
        # return False


    def train(self):
        self.prepareDataset()
        self.checkLabelEncoding()
        if self.dataset is None:
            logger.warning("No DatasetConfigs were found in DataHandlerConfig.json")
            return
        dataloader = DataLoader(self.dataset,batch_size=16, shuffle=True, num_workers=2)
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

def labelIntToHotEncoded(y):
    return torch.zeros(1000, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

def labelHotEncodedToInt(y):
    logger.critical(y)
    return torch.argmax(y, dim=0)