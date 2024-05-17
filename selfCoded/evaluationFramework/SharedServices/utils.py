from PIL import Image
import os, sys
import shutil

import logging
logger = logging.getLogger(__name__)

def singleton(class_):
    """
    This function is used as Decorator to modifies a class to be a singleton
    -> Only one instance of the class have to exist
    """
    instances = {}
    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance


def pil_loader(path: str) -> Image.Image:
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()  # Load the image into memory to access its mode attribute

        # Check the image mode
        if img.mode == 'L':
            # Image is grayscale
            return img  # Return as is
        elif img.mode == 'RGB':
            # Image is RGB
            return img  # Return as is
        else:
            # If the image has a different mode, convert it to RGB for consistency
            return img.convert("RGB")


# Function to copy a directory
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