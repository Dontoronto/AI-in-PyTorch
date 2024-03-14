import torch
import torchvision.transforms as T



@staticmethod
def saliency_map(model, original_image, single_batch):
    '''
    :param model: model to test
    :param original_image: PIL image type
    :param single_batch: Tensor type batched shape (1,channel,width,height)
    :return img, saliency: first is an image of tensor format,
                        second returning value "saliency" is a tensor with 1 channel i think
                        with values which points of the image are responsible for classification
                        the most
    '''
    model.eval()
    width, height = original_image.size
    img = T.Compose([T.ToTensor()])(original_image)

    # Set the requires_grad_ to the image for retrieving gradients
    single_batch.requires_grad_()

    # Retrieve output from the image
    output = model(single_batch)

    # Catch the output
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()

    # Retireve the saliency map and also pick the maximum value from channels on each pixel.
    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
    saliency, _ = torch.max(single_batch.grad.data.abs(), dim=1)
    saliency = saliency.reshape(width, height)

    model.eval()

    return img, saliency