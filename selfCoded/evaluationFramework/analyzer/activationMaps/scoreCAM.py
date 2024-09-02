import copy
import torch
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import ScoreCAM as _ScoreCAM
import torchvision.transforms as T

from ..evaluationMapsStrategy import EvaluationMapsStrategy


class ScoreCAM(EvaluationMapsStrategy):
    def analyse(self, model, original_image, single_batch, **kwargs):
        '''
        Github Source: https://github.com/frgfm/torch-cam
        :param model: model to test
        :param original_image: PIL Image type
        :param single_batch: Tensor type batched shape (1,channel,width,height)
        :param target_layer: string of layer name
        :return img, result: first is an image of tensor format with 3 channels,
                            second returning value is the result PIL image with
                            overlay and original image combined
        '''
        target_layer = kwargs.get('target_layer')
        if target_layer is None:
            return

        # image = original_image.copy()
        # image = image.convert(mode='RGB')
        # image = T.Compose([T.ToTensor()])(image)
        # img = T.Compose([T.ToTensor()])(original_image.copy())

        # Define the transformation pipeline once
        transform = T.Compose([T.ToTensor()])

        # Convert and transform the original image once
        image = transform(original_image.convert('RGB'))

        # Apply the same transformation to the original image copy for img
        img = transform(original_image.copy().convert('RGB'))

        with torch.no_grad():
            cam_extractor = _ScoreCAM(model, target_layer=target_layer, batch_size=single_batch.shape[0])

            batch = single_batch.detach().clone()
            # Preprocess your data and feed it to the model
            out = model(batch).squeeze(0).softmax(0)

            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(class_idx=out.squeeze(0).argmax().item())

            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

        return img, result
