import os
import shutil
import sys

import torch
import logging
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import random
import pandas as pd
import csv


logger = logging.getLogger(__name__)


from contextlib import contextmanager
from io import StringIO

@contextmanager
def capture_output_to_file(file_path):
    # Save the original stdout
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        yield
        # Get the output
        output = sys.stdout.getvalue()
        # Write the output to the specified file
        with open(file_path, 'w') as f:
            f.write(output)
        # Write the output to the terminal
        original_stdout.write(output)
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout

def create_missing_folders(base_path, folder_classes):
    # Ensure the base path exists
    if not os.path.exists(base_path):
        print(f"The specified base path '{base_path}' does not exist.")
        return

    # Get all existing folder names in the base path
    existing_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()]

    # Convert folder names to integers
    existing_numbers = sorted(int(folder) for folder in existing_folders if 0 <= int(folder) <= 999)

    # Create missing folders
    for number in range(folder_classes):
        if number not in existing_numbers:
            new_folder_path = os.path.join(base_path, str(number))
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")

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

def subsample(dataset, num_samples_per_class, batch_size, cuda_enabled=False):


    # # Klassenindizes sammeln
    # class_indices = {}
    # for idx, class_idx in enumerate(dataset.targets):
    #     if class_idx not in class_indices:
    #         class_indices[class_idx] = []
    #     class_indices[class_idx].append(idx)


    # Klassenindizes sammeln
    class_indices = {}
    for idx, target in enumerate(dataset.targets):
        if isinstance(target, torch.Tensor):
            target = target.item()  # Konvertiere von Tensor zu int
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(idx)

    # Stichprobe pro Klasse ziehen
    selected_indices = []
    for class_idx, indices in class_indices.items():
        sampled_indices = random.sample(indices, min(num_samples_per_class, len(indices)))
        selected_indices.extend(sampled_indices)

    # Subset-Dataset und DataLoader erstellen
    subset_dataset = Subset(dataset, selected_indices)
    if cuda_enabled is True:
        subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    return subset_loader




def save_dataframe_to_csv(df, file_path, index=False, na_rep='NA', sep=';', encoding='utf-8', columns=None, header=True, quoting=csv.QUOTE_MINIMAL):
    """
    Save a Pandas DataFrame to a CSV file with specified options.

    Parameters:
    - df: The DataFrame to save.
    - file_path: The file path where the CSV will be saved.
    - index: Whether to include the DataFrame's index in the CSV file.
    - na_rep: Representation for missing values.
    - sep: The delimiter to use in the CSV file.
    - encoding: The encoding for the CSV file.
    - columns: Specific columns to save.
    - header: Whether to include the header row in the CSV file.
    - quoting: Control the quoting behavior.
    """
    df.to_csv(
        file_path,
        index=index,
        na_rep=na_rep,
        sep=sep,
        encoding=encoding,
        columns=columns,
        header=header,
        quoting=quoting
    )

def load_dataframe_from_csv(file_path, sep=';',encoding='utf-8', na_values='NA'):
    """
    Load a Pandas DataFrame from a CSV file with specified options.

    Parameters:
    - file_path: The file path of the CSV file to load.
    - encoding: The encoding of the CSV file.
    - na_values: Additional strings to recognize as NA/NaN.

    Returns:
    - df: The loaded DataFrame.
    """
    df = pd.read_csv(
        file_path,
        sep=sep,
        encoding=encoding,
        na_values=na_values
    )
    return df

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images).to('cuda')
    labels = torch.tensor(labels).to('cuda')
    return images, labels






