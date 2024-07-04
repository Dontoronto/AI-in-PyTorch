import torchvision.transforms as transforms
import torch

class AdversarialModelWrapper:
    '''
    Interface for a model (classifier).

    Besides the required methods below, a model should do a reasonable job of
    providing easy access to internals to make white box attacks easier. For
    example, a model using TensorFlow might want to provide access to the input
    tensor placeholder and the tensor representing the logits output of the
    classifier.
    '''
    def __init__(self, model, transform_function):

        self._model = model
        self.cuda_enabled = False
        try:
            device = next(self._model.parameters()).device
            if device.type == 'cuda':
                self.cuda_enabled = True
                torch.set_default_device('cuda')
                print(f"Device= {device}")
        except Exception:
            self.cuda_enabled = False
            print("Failed to set device automatically, please try set_device() manually.")

        self.transform = transform_function

    def set_transformer(self, transformer):
        self.transform = transformer


    def classify(self, x):
        '''
        Returns the label for the input x (as a Python integer).
        '''

        x_in = self.transform(x).detach()
        if len(x_in.shape) < 4:
            x_in = x_in.unsqueeze(0)
        if self.cuda_enabled:
            x_in = x_in.to('cuda')
        prediction = self._model(x_in).squeeze(0)
        class_id = prediction.argmax().item()
        return class_id