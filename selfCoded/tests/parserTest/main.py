import torchattacks
from torchvision.io import read_image
import torch
from torchvision.models import resnet101, ResNet101_Weights
import sys, os
sys.path.append(os.getcwd())
from PIL import Image
import torchvision.transforms as T
import numpy as np

print(os.path.exists("testImages/tisch_v2.jpeg"))

#img = read_image("testImages/balloon.jpeg")
#img = Image.open("testImages/desk.jpeg")
#img = Image.open("testImages/balloon.jpeg")
img = Image.open("testImages/tisch_v2.jpeg")

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


attack = torchattacks.DeepFool(model, steps=1, overshoot=0.01)
adv_images = attack(batch.clone().detach(),idx.unsqueeze(0))
#print(adv_images)

prediction = model(adv_images).squeeze(0).softmax(0)
#print(prediction)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

numpy_image = adv_images.squeeze(0).permute(1, 2, 0).numpy()

# Convert the NumPy array to a PIL image
image = Image.fromarray((numpy_image * 255).astype('uint8'))
image.save("testImages/adv_desk.jpeg")


