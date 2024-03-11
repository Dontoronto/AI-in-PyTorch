import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from torchvision import transforms
from PIL import Image

def saliencyMap(img):
    image = img.convert(mode='RGB')
    image = T.Compose([T.ToTensor()])(image)

    # Set the requires_grad_ to the image for retrieving gradients
    image.requires_grad_()

    # Retrieve output from the image
    output = model(image)

# Catch the output
output_idx = output.argmax()
output_max = output[0, output_idx]

# Do backpropagation to get the derivative of the output based on the image
output_max.backward()

# Retireve the saliency map and also pick the maximum value from channels on each pixel.
# In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
saliency, _ = torch.max(X.grad.data.abs(), dim=1)
saliency = saliency.reshape(224, 224)

# Reshape the image
image = image.reshape(-1, 224, 224)

# Visualize the image and the saliency map
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
ax[0].axis('off')
ax[1].imshow(saliency.cpu(), cmap='hot')
ax[1].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()