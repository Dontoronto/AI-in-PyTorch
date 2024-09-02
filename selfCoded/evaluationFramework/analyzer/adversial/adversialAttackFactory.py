import torchattacks

import numpy as np
import torch
from torch import nn


class BPDAattack(object):
    def __init__(self, model=None, defense=None, device=None, epsilon=None, learning_rate=0.5,
                 max_iterations=100, clip_min=0, clip_max=1):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.defense = defense
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        # self.device = device
        try:
            self.device = next(self.model.parameters()).device
            if device.type == 'cuda':
                # self.cuda_enabled = True
                torch.set_default_device('cuda')
                print(f"Device= {device}")
        except Exception:
            # self.cuda_enabled = False
            print("Failed to set device automatically, please try set_device() manually.")

    def attack(self, x, y):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.

        """

        adv = x.detach().clone().to(self.device)

        lower = np.clip(x.detach().cpu().numpy() - self.epsilon, self.clip_min, self.clip_max)
        upper = np.clip(x.detach().cpu().numpy() + self.epsilon, self.clip_min, self.clip_max)

        for i in range(self.MAX_ITERATIONS):
            #adv_purified = self.defense(adv)
            adv_purified = adv.detach()
            adv_purified.requires_grad_()
            adv_purified.retain_grad()

            scores = self.model(adv_purified)
            loss = self.loss_fn(scores, y)
            loss.backward()

            grad_sign = adv_purified.grad.data.sign()

            # early stop, only for batch_size = 1
            # p = torch.argmax(F.softmax(scores), 1)
            # if y != p:
            #     break

            adv += self.LEARNING_RATE * grad_sign

            adv_img = np.clip(adv.detach().cpu().numpy(), lower, upper)
            adv = torch.Tensor(adv_img).to(self.device)
        return adv

    def __call__(self, x, y):
        return self.attack(x, y)

class AdversarialAttackerFactory:
    @staticmethod
    def create_attacker(model, typ, **kwargs):
        if typ == "DeepFool":
            return torchattacks.DeepFool(model, **kwargs)
        elif typ == "PGD":
            return torchattacks.PGD(model, **kwargs)
        elif typ == "JSMA":
            return torchattacks.JSMA(model, **kwargs)
        elif typ == "OnePixel":
            return torchattacks.OnePixel(model, **kwargs)
        elif typ == "SparseFool":
            return torchattacks.SparseFool(model, **kwargs)
        elif typ == "TIFGSM":
            return torchattacks.TIFGSM(model, **kwargs)
        elif typ == "AutoAttack":
            return torchattacks.AutoAttack(model, **kwargs)
        elif typ == "DIFGSM":
            return torchattacks.DIFGSM(model, **kwargs)
        elif typ == "MIFGSM":
            return torchattacks.MIFGSM(model, **kwargs)
        elif typ == "RFGSM":
            return torchattacks.RFGSM(model, **kwargs)
        elif typ == "EOTPGD":
            return torchattacks.EOTPGD(model, **kwargs)
        elif typ == "APGD_CE":
            return torchattacks.APGD(model, **kwargs)
        elif typ == "APGD_DLR":
            return torchattacks.APGD(model, **kwargs)
        elif typ == "APGDT":
            return torchattacks.APGDT(model, **kwargs)
        elif typ == "Jitter":
            return torchattacks.Jitter(model, **kwargs)
        elif typ == "CW":
            return torchattacks.CW(model, **kwargs)
        elif typ == "Square":
            return torchattacks.Square(model, **kwargs)
        elif typ == "BPDA":
            return BPDAattack(model, **kwargs)
        else:
            raise ValueError("Unknown attack type")
