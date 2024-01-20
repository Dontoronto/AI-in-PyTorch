import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

import torchattacks

from torchvision import models
from utils import get_imagenet_data, get_accuracy

images, labels = get_imagenet_data()
print('[Data loaded]')

#device = "cuda"

model = models.resnet18(pretrained=True).to().eval()
acc = get_accuracy(model, [(images.to(), labels.to())])
print('[Model loaded]')
print('Acc: %2.2f %%'%(acc))

from torchattacks import PGD

atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(atk)

adv_images = atk(images, labels)

from utils import imshow, get_pred
idx = 0
pre = get_pred(model, adv_images[idx:idx+1])
imshow(adv_images[idx:idx+1], title="True:%d, Pre:%d"%(labels[idx], pre))