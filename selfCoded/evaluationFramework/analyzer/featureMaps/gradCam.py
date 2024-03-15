from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
import torchvision.transforms as T



def gradCamLayer(model, original_image, single_batch, target_layer):
    '''
    :param model: model to test
    :param original_image: PIL Image type
    :param single_batch: Tensor type batched shape (1,channel,width,height)
    :param target_layer: "string of layer name
    :return img, result: first is an image of tensor format with 3 channels,
                        second returning value is the result PIL image wiht
                        overlay and original image combined
    '''

    image = original_image.copy()
    image = image.convert(mode='RGB')
    image = T.Compose([T.ToTensor()])(image)
    img = T.Compose([T.ToTensor()])(original_image.copy())
    with SmoothGradCAMpp(model, target_layer=target_layer) as cam_extractor:
        # Preprocess your data and feed it to the model
        model.eval()
        out = model(single_batch).squeeze(0).softmax(0)

        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        # Resize the CAM and overlay it
        result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    return img, result
