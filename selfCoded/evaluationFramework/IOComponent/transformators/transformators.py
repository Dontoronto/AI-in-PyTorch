from torch import functional as F
import torch
from torch import nn, Tensor
import torchvision.transforms as T


def imagenet_transformer(image_flag=True) -> T.Compose:
    if image_flag is True:
        transformator = T.Compose([
           T.Resize(256, interpolation=T.InterpolationMode("bilinear"), antialias=True),
           T.CenterCrop( 224),
           T.ToTensor(),
           T.ConvertImageDtype( torch.float),
           T.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transformator

def adv_imagenet_transformer():
    transformator = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transformator



def mnist_transformer() -> T.Compose:
    transformator = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype( torch.float),
        T.Normalize(mean=[0.1307], std=[0.3081])
    ])

    return transformator

def imagenet_cpu_transformer(img: Tensor) -> Tensor:
    img = F.resize(img, 256, interpolation='bilinear', antialias=True)
    img = F.center_crop(img, 224)
    if not isinstance(img, Tensor):
        img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float32)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img