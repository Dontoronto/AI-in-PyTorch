import robustml
import torchvision.transforms as transforms

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
        self._transform = transform_function

    def classify(self, x):
        '''
        Returns the label for the input x (as a Python integer).
        '''
        x_in = self._transform(x)
        prediction = self._model(x_in).squeeze(0)
        class_id = prediction.argmax().item()
        return class_id