import robustml
import sys
import numpy as np
import torchattacks
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image

import robustml_utils
import adversialModelWrapper

# TODO: schauen wie ich die Backwards Transformation Methode von Datahandler nachbauen kann als mock
# TODO: datahandler muss später um die funktion erweitert werden damit das automatisch generiert wird
# TODO: datahandler braucht get Methoden für die Transform function- und Backwards Transform funktionen
# TODO: dataset pfade müssen übergeben werden. schauen ob evtl. Parser zum einsatz kommt oder es anders
#       geladen werden muss
# TODO: auf numpy konvertierungen achten
# TODO: testrun machen ohne save funktion, es muss aber backwards transformieren können
# TODO: save funktion implementieren und schauen wie das läuft
# TODO: dynamisch tests einstellen können. Damit ich beliebige Anzahl von tests laufen lasen kann
# TODO: methode die für gewählten Index einmal das bild anzeigt welches daraus entstanden ist
# TODO: abgeänderte Skripte behalten ohne das pip package zu verwenden.

class AdversarialAttacker(robustml.attack.Attack):
    def __init__(self, model, transform_batched_function,
                 backwards_transform_function,
                 save_adversarial_images=False,
                 *kwargs):
        '''

        :param model:
        :param transform_function:
        :param backwards_transform_function: muss in numpy ausgeben können
        :param save_adversarial_images:
        :param kwargs:
        '''

        self._model = model

        self.transform = transform_batched_function

        # TODO: Wird evtl. nicht immer gebraucht und soll extern gesetzt werden
        self.backwards_transform_function = backwards_transform_function

        self.save_images_flag = save_adversarial_images

        self.adversarialModel = model

        self.dataset_provider = None

        self.threat_model = None

    def createAdversarialEvaluationModel(self):
        self.adversarialModel = adversialModelWrapper.AdversarialModelWrapper(self._model,
                                                                              self.transform)

    def setProvider(self, dataset_provider):
        self.dataset_provider = dataset_provider

    def getDatasetProvider(self):
        return self.dataset_provider

    def setThreatModel(self, threat_model):
        self.threat_model = threat_model

    def getThreatModel(self):
        return self.threat_model


    def evaluate(self, start, end):
        if start is not None and not (0 <= start < len(self.dataset_provider)):
            raise ValueError('start value out of range')
        if end is not None and not (0 <= end <= len(self.dataset_provider)):
            raise ValueError('end value out of range')

        robustml_utils.evaluate.evaluate(
            self._model,
            self,
            self.getDatasetProvider(),
            start,
            end,
            deterministic=False,
            debug=True
        )



    def run(self, x, y, target=None):

        # #x with shape (32, 32, 3) to tensor with shape (1,3,32,32)
        x_in = self.transform(x)


        #y as int to tensor with shape (1),  1 -> tensor([1])
        # with dtype=torch.uint8 war die itemsize 1 und DeepFool hat nicht funktioniert weil fs komisch geschnitten
        # wurde, mit torch.int64 funktioniert es
        y_label = torch.tensor([y], dtype=torch.int64)

        # y auch noch von numpy in torch umwandeln
        #attack = torchattacks.PGD(self._model, eps=self._epsilon, alpha=self._alpha, steps=self._max_steps, random_start=True)
        attack = torchattacks.DeepFool(self._model, steps=self._max_steps, overshoot=0.01)
        print(x_in.shape)
        print(y_label.shape)
        adv_images = attack(x_in, y_label)

        adv_numpy_img = self.backwards_transform_function(adv_images, numpy_original_shape_flag=True)

        if self._showImages == True:
            plotten_real = np.copy(x)
            plot_real_img = Image.fromarray((plotten_real * 255).astype('uint8'))
            plot_real_img.show("real")

            plot_img = Image.fromarray((np.copy(adv_numpy_img) * 255).astype('uint8'))
            plot_img.show("adversarial")


        return adv_numpy_img

class NullAttack(robustml.attack.Attack):
    def run(self, x, y, target):
        return x
#%%
