import os
import torch


def save_tensors_to_files(tensor, folder='output'):
    """
    Saves each 3x3 tensor in the given 126x3x3 tensor to individual text files.
    :param tensor: A 126x3x3 PyTorch tensor.
    :param folder: The folder where text files will be saved.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    file_name = f"{folder}/tensor.txt"
    with open(file_name, 'w') as file:
        #Iterate and save each 3x3 tensor
        for i, matrix in enumerate(tensor):
            for row in matrix:
                file.write(' '.join(map(str, row.tolist())) + '\n')
            file.write('\n')


def count_unique_tensors(tensor_list):
    '''
    returns the amount of unique tensors in a list
    :param tensor_list: list of tensors
    :return: lenght of the unique tensors in the list
    '''
    unique_tensors = []
    for tensor in tensor_list:
        if not any(torch.all(torch.eq(tensor, unique_tensor)) for unique_tensor in unique_tensors):
            unique_tensors.append(tensor)
    return len(unique_tensors)