import torchattacks
from torchvision.io import read_image
import torch
from torchvision.models import resnet101, ResNet101_Weights
import sys, os
sys.path.append(os.getcwd())
from PIL import Image
import torchvision.transforms as T
import numpy as np

print(os.path.exists("testImages/balloon.jpeg"))

#img = read_image("testImages/balloon.jpeg")
#img = Image.open("testImages/desk.jpeg")
#img = Image.open("testImages/balloon.jpeg")
#img = Image.open("testImages/tisch_v2.jpeg")
#img = Image.open("testImages/adv_desk.jpeg")
img = Image.open("testImages/tischShow.jpeg")

np_image1 = np.array(img)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])
img = preprocess(img)

#img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# Step 1: Initialize model with the best available weights
weights = ResNet101_Weights.IMAGENET1K_V1
model = resnet101(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch.clone().detach()).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

temp = torch.topk(prediction,3)[1]
print(temp) #tensor([532, 739, 896])



_, idx = torch.max(prediction, dim=0)

# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
#cam_extractor = SmoothGradCAMpp(model)

with SmoothGradCAMpp(model, target_layer='layer4') as cam_extractor:
    print(img.shape)
    print(batch.shape)
    # Preprocess your data and feed it to the model
    out = model(batch.clone().detach()).squeeze(0).softmax(0)

    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()



#%%
