
from enum import Enum, auto

import logging
logger = logging.getLogger(__name__)

class ADMMVariable(Enum):
    '''
    input for Layerclass to choose which variables should be initialized/ removed
    '''
    W = auto()
    Z = auto()
    U = auto()

class LayerInfo:
    '''
    This class is a wrapper for a layer which should be modifyed with admm.
    Wrapper is the same for every class. Depending on the enum value other variables will be named
    For example W, U and Z (for ADMM)
    '''
    def __init__(self, name, module, param, sparsity):
        self.module = module
        self.name = name
        self.param = param
        self.type = type(module).__name__
        self.sparsity = sparsity

        self.U = None
        self.Z = None
        self.state = None

    def set_admm_vars(self, ADMM_ENUM: Enum):
        # Reset variables to None before initialization
        self.U = None
        self.Z = None
        self.state = ADMM_ENUM

        if ADMM_ENUM == ADMMVariable.W:
            logger.info(f"Layer was set as Weight Layer 'W' with Weight and Gradient")
        elif ADMM_ENUM == ADMMVariable.U:
            self.U = self.module.weight.data.clone().detach().zero_()
            logger.info(f"Layer was set as Dual Variable Layer 'U' with same shape as weights")
        elif ADMM_ENUM == ADMMVariable.Z:
            self.Z = self.module.weight.data.clone().detach()
            logger.info(f"Layer was set as Auxiliary Variable Layer 'Z' copy of Weights")

    # W and dW properties
    @property
    def W(self):
        if self.state == ADMMVariable.W:
            return self.module.weight.data
        else:
            logger.warning(f"GET not possible instance is not configured as Weight Layer: {self}")
            return None

    @W.setter
    def W(self, value):
        if self.state == ADMMVariable.W:
            self.module.weight.data = value
        else:
            logger.warning(f"SET not possible instance is not configured as Weight Layer: {self}")

    @property
    def dW(self):
        if self.state == ADMMVariable.W:
            return self.param.grad
        else:
            logger.warning(f"GET not possible instance is not configured as Weight Layer: {self}")
            return None

    @dW.setter
    def dW(self, value):
        if self.state == ADMMVariable.W:
            self.param.grad = value
        else:
            logger.warning(f"SET not possible instance is not configured as Weight Layer: {self}")
