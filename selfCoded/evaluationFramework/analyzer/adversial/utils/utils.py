import logging
import os
import shutil

logger = logging.getLogger(__name__)

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

def save_list_to_file(lst, path):
    """
    Saves a list of integers to a file, one integer per line.

    Parameters:
    lst (list): List of integers to save.
    path (str): The file path where the list will be saved.
    """
    try:
        with open(path, 'w') as file:
            for item in lst:
                file.write(f"{item}\n")
        print(f"List saved successfully to {path}")
    except Exception as e:
        print(f"An error occurred while saving the list: {e}")

def load_list_from_file(path):
    """
    Loads a list of integers from a file.

    Parameters:
    path (str): The file path from which the list will be loaded.

    Returns:
    list: The list of integers loaded from the file.
    """
    try:
        with open(path, 'r') as file:
            lst = [int(line.strip()) for line in file]
        print(f"List loaded successfully from {path}")
        return lst
    except Exception as e:
        print(f"An error occurred while loading the list: {e}")
        return None

# Example usage:
# save_list_to_file([1, 2, 3, 4, 5], 'path/to/your/file.txt')
# loaded_list = load_list_from_file('path/to/your/file.txt')
# print(loaded_list)
