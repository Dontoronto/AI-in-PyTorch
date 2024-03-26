import robustml
import sys
import numpy as np
import torchattacks
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image

class Cifar10PGD(robustml.attack.Attack):
    def __init__(self, model, epsilon, alpha, max_steps=100, show_images=False):
        self._model = model
        self._epsilon = epsilon
        self._alpha = alpha
        self._max_steps = max_steps
        self._showImages = show_images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def run(self, x, y, target=None):

        # #x with shape (32, 32, 3) to tensor with shape (1,3,32,32)
        x_in = self.transform(x)
        x_in = x_in.unsqueeze(0)


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

        adv_numpy_img = adv_images.clone().detach().squeeze(0).permute((1,2,0)).numpy()
        adv_numpy_img = adv_numpy_img/2+0.5

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
