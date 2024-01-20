#from torchvision.io import read_image
import torch
from torchvision.models import resnet101, ResNet101_Weights
import sys, os
sys.path.append(os.getcwd())
from PIL import Image
import torchvision.transforms as T
import numpy as np

from HeritatePreTrainedModel import DeepFoolHeritage



print(os.path.exists("testImages/balloon.jpeg"))

#img = read_image("testImages/balloon.jpeg")
#img = Image.open("testImages/desk.jpeg")
img = Image.open("testImages/adv_desk.jpeg")
#img.show("Picture")
#just for comparison
np_image1 = np.array(img)
#img = Image.open("testImages/balloon.jpeg")
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])
#img = T.ToTensor()(img)
img = preprocess(img)
#print(img.shape)


# # Step 1: Initialize model with the best available weights
# weights = ResNet101_Weights.IMAGENET1K_V1
# model = resnet101(weights=weights.transforms(antialias=None))
# model.eval()
#
# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()
#
# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)
#
# print(batch)
weights = ResNet101_Weights.IMAGENET1K_V1

model = DeepFoolHeritage(weights)

# Step 4: Use the model and print the predicted category
#with redirect_stdout(sys.stdout):
_ = model.forward(img.clone().detach().unsqueeze(0))
prediction = _.squeeze(0).softmax(0)

class_ids = torch.topk(prediction, k=5).indices
#class_id = prediction.argmax().item()
print("Model Results:")
for id in class_ids:
    score = prediction[id].item()
    category_name = weights.meta["categories"][id]
    print(f"{category_name}: {100 * score:.1f}%")




## Just in Case we want to save the weights of the model and
## apply them afterwards
#torch.save(D.state_dict(), 'CPUDiscriminator_batch.pth')
#D.load_state_dict(saved_model)


### Here Starts the attack
from torchattacks.attacks.deepfool import DeepFool



DeepFool = DeepFool(model=model,steps=500, overshoot=0.06)
#print(torch.unique(img))
adv_images = DeepFool(img.unsqueeze(0).clone().detach(), prediction).clone().detach()
#print(torch.unique(img))
adv_images_Comparison = adv_images.squeeze(0)

numpy_image = adv_images.squeeze(0).permute(1, 2, 0).numpy()

# Convert the NumPy array to a PIL image
image = Image.fromarray((numpy_image * 255).astype('uint8'))

# Display the image
#image.save("testImages/adv_desk.jpeg")
#image.show()

adv_prediction = model(adv_images).squeeze(0).softmax(0)
adv_class_ids = torch.topk(adv_prediction, k=5).indices
#adv_class_id = adv_prediction.argmax().item()
print("Adversial Results:")
for id in adv_class_ids:
    adv_score = adv_prediction[id].item()
    adv_category_name = weights.meta["categories"][id]
    print(f"{adv_category_name}: {100 * adv_score:.1f}%")

# # Convert images to NumPy arrays
#
# np_image2 = np.array(image)
#
# # Calculate the absolute difference between the images
# difference = np.abs(np_image1 - np_image2)
diff = img - adv_images_Comparison
#print(torch.unique(diff))
#
# # Create a PIL image from the difference array
# diff_image = Image.fromarray(difference.astype('uint8'))
# print(difference.argmax())
#
# # Display the difference image
# diff_image.show()



#%%
