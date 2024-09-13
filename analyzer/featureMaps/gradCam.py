import copy

import torch
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
import torchvision.transforms as T

from ..evaluationMapsStrategy import EvaluationMapsStrategy



class GradCAM(EvaluationMapsStrategy):

    def analyse(self, model, original_image, single_batch, **kwargs):
        '''
        :param model: model to test
        :param original_image: PIL Image type
        :param single_batch: Tensor type batched shape (1,channel,width,height)
        :param target_layer: "string of layer name
        :return img, result: first is an image of tensor format with 3 channels,
                            second returning value is the result PIL image wiht
                            overlay and original image combined
        '''
        label = None
        target_layer = kwargs.get('target_layer')
        label = kwargs.get('label')
        if target_layer is None:
            return


        # Define the transformation pipeline once
        transform = T.Compose([T.ToTensor()])

        # Convert and transform the original image once
        image = transform(original_image.convert('RGB'))

        # Apply the same transformation to the original image copy for img
        img = transform(original_image.copy().convert('RGB'))

        with SmoothGradCAMpp(model, target_layer=target_layer, std = 1e-15) as cam_extractor:

            batch = single_batch.detach().clone()
            # Preprocess your data and feed it to the model
            # out = model(batch).squeeze(0).softmax(0)
            out = model(batch).squeeze(0)

            # Retrieve the CAM by passing the class index and the model output
            if label:
                activation_map = cam_extractor(label, out)
            else:
                activation_map = cam_extractor(out.argmax().item(), out)
            # activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

        return img, result

