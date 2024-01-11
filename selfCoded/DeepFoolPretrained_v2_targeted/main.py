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
# img = Image.open("testImages/tisch_v2.jpeg")
img = Image.open("testImages/waterSnake.jpeg")


np_image1 = np.array(img)
# preprocess = T.Compose([
#     #T.Resize(256),
#     T.Resize((224,224)),
#     #T.CenterCrop(224),
#     T.ToTensor(),
# ])
# img = preprocess(img)

## ONLY NECESSARY TO CHECK IF IMAGE IS RIGHT FORMATTED
# numpy_image = img.permute(1, 2, 0).numpy()
# # Convert the NumPy array to a PIL image
# image = Image.fromarray((numpy_image * 255).astype('uint8'))
# image.save("testImages/pre_adv_pic.jpeg")

#img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# Step 1: Initialize model with the best available weights
weights = ResNet101_Weights.IMAGENET1K_V1
model = resnet101(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

#pre model picture
# numpy_image = batch.squeeze(0).permute(1, 2, 0).numpy()
# # Convert the NumPy array to a PIL image
# image = Image.fromarray((numpy_image * 255).astype('uint8'))
# image.save("testImages/pre_adv_pic.jpeg")


# Step 4: Use the model and print the predicted category
prediction = model(batch.clone().detach()).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

temp = torch.topk(prediction,15)[1]
print(temp) #tensor([532, 739, 896])



_, idx = torch.max(prediction, dim=0)
idx = idx.unsqueeze(0)


attack = torchattacks.DeepFool(model, steps=10, overshoot=0.02)
attack.set_mode_targeted_by_function(target_map_function=lambda images, idx:(idx-idx+44)) #(idx+1)%3
adv_images = attack(batch.clone().detach(),idx)

prediction = model(adv_images).squeeze(0).softmax(0)

##check new prediction
temp = torch.topk(prediction,15)[1]
print(temp)

class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

numpy_image = adv_images.squeeze(0).permute(1, 2, 0).numpy()

# Convert the NumPy array to a PIL image
image = Image.fromarray((numpy_image * 255).astype('uint8'))
image.save("testImages/adv_pic.jpeg")


