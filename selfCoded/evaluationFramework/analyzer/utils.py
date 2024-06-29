import os
import shutil

import torch
import logging
import torchvision.transforms as T


logger = logging.getLogger(__name__)

def weight_export(model, layer_name, path):

    for name, module in model.named_modules():
        #name == layer_name and

        if (hasattr(module, 'weight') and
                module.weight is not None and module.weight.requires_grad):

            # Access the weight tensor directly
            weight_tensor = module.weight.data

            torch.save(weight_tensor, path)
            #print(f"shape of layers = {weight_tensor.shape}")

            return


def create_directory(file_path):
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
        logger.warning(f"created directory at {file_path}")
    else:
        logger.critical(f"could not create directory at {file_path}")

def copy_directory(source_dir, destination_dir):
    try:
        # Check if the source directory exists
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory '{source_dir}' does not exist.")
            return

        # Check if the destination directory already exists
        if os.path.exists(destination_dir):
            logger.warning(f"Destination directory '{destination_dir}' already exists.")
            return

        # Copy the directory tree
        shutil.copytree(source_dir, destination_dir)
        logger.warning(f"Directory '{source_dir}' has been copied to '{destination_dir}'.")
    except Exception as e:
        logger.critical(f"An error occurred while copying the directory: {e}")

def adjust_transformer(transformer):
    transform_steps = []
    crop_size = None

    # Überprüfen, ob CenterCrop im Transformer enthalten ist und entferne es
    for t in transformer.transforms:
        if isinstance(t, T.CenterCrop):
            crop_size = t.size
        else:
            transform_steps.append(t)

    # Füge einen angepassten Resize hinzu, wenn CenterCrop gefunden wurde
    if crop_size is not None:
        transform_steps.insert(0, T.Resize(crop_size, interpolation=T.InterpolationMode("bilinear"), antialias=True))

    new_transformer = T.Compose(transform_steps)
    return new_transformer




