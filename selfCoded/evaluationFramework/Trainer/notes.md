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