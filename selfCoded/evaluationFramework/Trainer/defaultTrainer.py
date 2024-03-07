from .trainer import Trainer
from torch.utils.data import DataLoader

import os
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

        # Variables for snapshot state saver
        self.best_test_loss = float('inf')
        self.best_epoch = -1
        self.epoch_since_improvement = 0
        self.snapshot_model_path = None
        self.snapshot_model_path_raw = None
        self.recovery_epoch = None
        self.snapshot_enabled = False
        self.model_name = "model"
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
        if modelOutputClasses != len(SampleDataset.classes):
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

    def setSnapshotSettings(self, kwargs: dict):
        super().setSnapshotSettings(kwargs)

        if self.snapshotConfig.get('snapshot_enabled') is not None:
            logger.info("Recovery Epoch was set to : " +
                        str(self.snapshotConfig.get('snapshot_enabled')))
            self.snapshot_enabled = self.snapshotConfig.get('snapshot_enabled')
        else:
            self.snapshot_enabled = False
            logger.warning(f"Snapshot is disabled ")
            return

        # sets prefix for model files
        if self.snapshotConfig.get('name') is not None:
            logger.info("Model file prefix was set to : " +
                        str(self.snapshotConfig.get('name')))
            self.model_name = self.snapshotConfig.get('name')
        else:
            logger.warning(f"model name prefix was not set")

        # path where model should be safed
        if self.snapshotConfig.get('snapshot_model_path') is not None:
            logger.info("Snapshot model path was set to : " +
                        self.snapshotConfig.get('snapshot_model_path'))
            self.snapshot_model_path = os.path.join(self.snapshotConfig.get('snapshot_model_path'),
                                                    self.model_name + ".pth")
        else:
            self.snapshot_model_path = os.path.join(self.snapshotConfig.get('snapshot_model_path'),self.model_name)
            logger.warning(f"No snapshot model path was set. Using default snapshot path:" +
                           f" {self.snapshot_model_path + '.pth'}")

        # path where raw model should be safed
        if self.snapshotConfig.get('snapshot_model_path_raw') is not None:
            logger.info("Snapshot model path was set to : " +
                        self.snapshotConfig.get('snapshot_model_path_raw'))
            self.snapshot_model_path_raw = os.path.join(self.snapshotConfig.get('snapshot_model_path_raw'),
                                                        "raw_" + self.model_name+ ".pth")
        else:
            logger.warning(f"No Snapshot path of raw model was set")

        # how many epochs until previouse model gets recovered
        if self.snapshotConfig.get('recovery_epoch') is not None:
            logger.info("Recovery Epoch was set to : " +
                        str(self.snapshotConfig.get('recovery_epoch')))
            self.recovery_epoch = self.snapshotConfig.get('recovery_epoch')
        else:
            self.recovery_epoch = 5
            logger.warning(f"No recovery epoch was defined in TrainerConfig.json. Recovery Epoch set to " +
            f": {self.recovery_epoch}")





    # NOTE: here comes the trainer for mnist
    # TODO: make it more general with train and test, no redundant dataloader initialization and testset
    def train(self, test=False): #model, device, train_loader, optimizer, epoch):
        self.model.train()
        self.preTrainingChecks()
        dataloader = self.createDataLoader(self.dataset)
        if test is True:
            self.prepareDataset(testset=True)
            test_loader = self.createDataLoader(self.testset)
            test_loader.shuffle = False
        for epo in range(self.epoch):
            for batch_idx, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epo} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

            if test is True:
                self.test(test_loader, snapshot_enabled=self.snapshot_enabled, current_epoch=epo)




    def test(self, test_loader, snapshot_enabled=False,current_epoch=None):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loader = test_loader
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += self.loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
        if snapshot_enabled is True:
            self.createSnapshot(test_loss, current_epoch)
    def createSnapshot(self, val_loss, epoch):
        if epoch > self.recovery_epoch:
            # Prüfen, ob es eine Verbesserung gibt
            if val_loss < self.best_test_loss:
                self.best_test_loss = val_loss
                self.best_epoch = epoch
                self.epoch_since_improvement = 0
                # Speichern des besten Checkpoints
                logger.info(f"New best Model will be safed to: {self.snapshot_model_path}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, self.snapshot_model_path)
                if self.snapshot_model_path_raw is not None:
                    logger.info(f"New raw best Model will be safed to: {self.snapshot_model_path_raw}")
                    torch.save(self.model.state_dict(),self.snapshot_model_path_raw)
            else:
                # Wenn sich das Modell in die falsche Richtung entwickelt
                if os.path.exists(self.snapshot_model_path):
                    self.epoch_since_improvement += 1
                    if self.epoch_since_improvement >= self.recovery_epoch:
                        logger.info(f'Keine Verbesserung seit {self.recovery_epoch} Epochen, lade besten Checkpoint von Epoche {self.best_epoch+1}')
                        # Laden des besten Checkpoints
                        checkpoint = torch.load(self.snapshot_model_path)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.epoch_since_improvement = 0  # Reset der Patience nach dem Zurückladen
                else:
                    logger.info(f"New best Model will be safed to: {self.snapshot_model_path}")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                    }, self.snapshot_model_path)
                    if self.snapshot_model_path_raw is not None:
                        logger.info(f"New raw best Model will be safed to: {self.snapshot_model_path_raw}")
                        torch.save(self.model.state_dict(),self.snapshot_model_path_raw)

    # NOTE: this method is for saving a model state with a log-message
    def export_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        logger.info('Model state_dict saved to %s', model_path)

        # , onnx_path=None
        # if onnx_path is not None:
        #     assert input_shape is not None, 'input_shape must be specified to export onnx model'
        #     # input info needed
        #     if device is None:
        #         device = torch.device('cpu')
        #     input_data = torch.Tensor(*input_shape)
        #     torch.onnx.export(self._model_to_prune, input_data.to(device), onnx_path)
        #     _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

    # TODO: this method needs to be extended to meet the criteria of general trainer
    # def train(self, test = False):
    #     self.preTrainingChecks()
    #     dataloader = self.createDataLoader(self.dataset)
    #     self.model.train()
    #     for batch, (X, y) in enumerate(dataloader):
    #
    #         # remove existing settings
    #         self.optimizer.zero_grad()
    #
    #         # Compute prediction and loss
    #         pred = self.model(X)
    #         loss = self.loss(pred, y)
    #
    #         # Backpropagation
    #         loss.backward()
    #
    #         return
    #         # Apply optimization with gradients
    #         self.optimizer.step()
    #
    #         if batch % 2 == 0:
    #             loss, current = loss.item(), batch * len(X)
    #             print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
    #
    #
    #     if test is True:
    #         self.test()
    #
    #     pass



    # TODO: needs to be adapted for generalization
    # def test(self):
    #     self.prepareDataset(testset=True)
    #     if self.testset is None:
    #         logger.warning("No DatasetConfigs were found in DataHandlerConfig.json")
    #         return
    #     self.checkLabelEncoding(self.testset)
    #     # TODO: evtl. andere Configs für Dataloader bei tests
    #     dataloader = self.createDataLoader(self.testset)
    #     self.model.eval()
    #     for batch, (X, y) in enumerate(dataloader):
    #
    #         # Compute prediction and loss
    #         pred = self.model(X)
    #         loss = self.loss(pred, y)
    #
    #         if batch % 2 == 0:
    #             loss, current = loss.item(), batch * len(X)
    #             print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
    #             break
    #     pass



def labelIntToHotEncoded(y,size):
    return torch.zeros(size, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

def labelHotEncodedToInt(y):
    return torch.argmax(y, dim=0).item()


# count = 0
# for batch, labels in dataloader:
#     count +=1
#     if count == 2:
#         break
#     with torch.no_grad():
#         if len(batch) == 1:
#             # Step 4: Use the model and print the predicted category
#             prediction = Model(batch.clone().detach()).squeeze(0).softmax(0)
#             class_id = prediction.argmax().item()
#             score = prediction[class_id].item()
#             category_name = _weights.meta["categories"][class_id]
#             print(f"{category_name}: {100 * score:.1f}%")
#         else:
#             # Assuming 'Model' is your model instance and 'batch' is your input batch of tensors
#             predictions = Model(batch.clone().detach())
#             dataset_image = DataHandler.preprocessBackwardsBatched(batch)
#             for i in dataset_image:
#                 i.show()
#
#             softmax_predictions = torch.softmax(predictions, dim=1)  # Apply softmax on the correct dimension
#
#             for i, prediction in enumerate(softmax_predictions):
#                 class_id = prediction.argmax().item()
#                 score = prediction[class_id].item()
#                 category_name = _weights.meta["categories"][class_id]
#                 print(f"Tensor {labels[i]}: {category_name}: {100 * score:.1f}%")



