# A few code snippets which can be useful in the future

### Test of the reference seperation between layer list for W,U,Z
```
for i in self.pruningLayers:
logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")

        for i in list_U:
            logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")

        for i in list_Z:
            logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")

        self.testZCopy()
        logger.critical(f"Difference W-Z: {(self.pruningLayers[0].W-list_Z[0].Z).sum()}")
        for module_name, module in self.model.named_modules():
            if module_name == "model.conv1":
                logger.critical(f"Difference Model-Z: {(module.weight.data-list_Z[0].Z).sum()}")
```

### Test of influnece of modification of LayerInfo

```
self.list_Z[0].Z = self.list_Z[0].Z*mask
        self.list_W[0].W = self.list_W[0].W*mask
        logger.critical(f"Difference after prune W-Z:")
        logger.critical(self.list_W[0].W - self.list_Z[0].Z)

        for i in self.list_W:
            logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")

        for i in self.list_U:
            logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")

        for i in self.list_Z:
            logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")

        #self.testZCopy()
        logger.critical(f"Difference W-Z: {(self.list_W[0].W-self.list_Z[0].Z).sum()}")
        for module_name, module in self.model.named_modules():
            if module_name == "model.conv1":
                logger.critical(f"Difference Model-Z: {(module.weight.data-self.list_Z[0].Z).sum()}")
```

### Test old of Gradients in Layer-Instances

```
    # def testGradientModification(self):
    #     for name, param in self.model.named_parameters():
    #         if name == "model.conv1.weight":
    #             before = copy.deepcopy(param.data)
    #             for module_name, module in self.model.named_modules():
    #                 if module_name == "model.conv1":
    #                     test_obj = LayerInfoGrad(name, module, param)
    #                     test_obj.module.weight.data += 0.1
    #                     logger.critical(torch.unique(before - test_obj.module.weight.data))
    #                     logger.critical(test_obj.grad.shape)
    #                     logger.critical(test_obj.module.weight.data.shape)
    #                     return
```

### Train method OLD

```
# def train(self, test = False):
    #     self.preTrainingChecks()
    #     dataloader = self.createDataLoader(self.dataset)
    #     self.model.train()
    #
    #     count = 0
    #     for i in range(self.epoch):
    #         for batch, (X, y) in enumerate(dataloader):
    #
    #             if count == self.main_iterations:
    #                 return
    #
    #             # remove existing settings
    #             self.optimizer.zero_grad()
    #
    #             # Compute prediction and loss
    #             pred = self.model(X)
    #             loss = self.loss(pred, y)
    #
    #             # Backpropagation
    #             loss.backward()
    #
    #             # here should the logic of admm cycle be located
    #             self.admmFilter()
    #
    #             # Apply optimization with gradients
    #             self.optimizer.step()
    #
    #             if batch % 2 == 0:
    #                 loss, current = loss.item(), batch * len(X)
    #                 print(f"Epoch number: {i}")
    #                 print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
    #
    #             # if it hits main_iterations count it will end the admm training
    #             count += 1
    #
    #
    #     if test is True:
    #         self.test()
    #
    #     pass
```