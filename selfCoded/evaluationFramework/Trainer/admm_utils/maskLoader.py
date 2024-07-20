import torch

def load_pruning_mask_csv(csv_path):

    # for conv layer with int64 values
    tensor_list = []
    current_tensor = []
    temp_tensors = []

    # for fully connected layer with bool values
    tensor_bool_list = []

    # for conv layer with bool values
    tensor_conv_bool_list = []
    current_conv_bool_tensor = []
    temp_conv_bool_tensors = []

    # general
    final_list = []

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tensor:
                    # Convert current_tensor to a PyTorch tensor and add it to the list
                    tensor_list.append(torch.tensor(current_tensor, dtype=torch.int64))
                    current_tensor = []
                    final_list.append(torch.stack(tensor_list))
                    tensor_list = []

                if current_conv_bool_tensor:
                    # Convert current_tensor to a PyTorch tensor and add it to the list
                    tensor_conv_bool_list.append(torch.tensor(current_conv_bool_tensor, dtype=torch.bool))
                    current_conv_bool_tensor = []
                    final_list.append(torch.stack(tensor_conv_bool_list))
                    tensor_conv_bool_list = []

                if tensor_bool_list:
                    #tensor_bool_list.append(torch.tensor(current_bool_tensor, dtype=torch.bool))
                    #current_bool_tensor = []
                    final_list.append(torch.stack(tensor_bool_list))
                    tensor_bool_list = []

            elif line == 'Tensor':
                continue
            elif '[' not in line or ']' not in line:

                fc_bool_list = list(map(lambda x: True if x=='True' else False, line.split(',')))
                tensor_bool_list.append(torch.tensor(fc_bool_list, dtype=torch.bool))
            else:

                if "True" in line or "False" in line:
                    if current_conv_bool_tensor:
                        # Convert current_tensor to a PyTorch tensor and add it to the list
                        tensor_conv_bool_list.append(torch.tensor(current_conv_bool_tensor, dtype=torch.bool))
                        current_conv_bool_tensor = []

                    # Parse the line to get individual 3x3 tensors
                    row_tensors = line.split('],')
                    for tensor_str in row_tensors:
                        tensor_str = tensor_str.strip(' []')  # Remove brackets and extra spaces
                        if tensor_str:  # Only process non-empty strings
                            tensor = list(map(lambda x: True if x=='True' else False, tensor_str.split(',')))# for x in tensor_str.split('], [').split(',')
                            #tensor = [list(map(int, x.split(','))) for x in tensor_str.split('], [')]
                            temp_conv_bool_tensors.append(tensor)

                            # Check if we have a complete 3x3 tensor
                            if len(temp_conv_bool_tensors) == 3:
                                current_conv_bool_tensor.append(temp_conv_bool_tensors)
                                temp_conv_bool_tensors = []

                else:
                    # TODO: same logic as below just for bool values inside the same structured lists of strings
                    if current_tensor:
                        # Convert current_tensor to a PyTorch tensor and add it to the list
                        tensor_list.append(torch.tensor(current_tensor, dtype=torch.int64))
                        current_tensor = []

                    # Parse the line to get individual 3x3 tensors
                    row_tensors = line.split('],')
                    for tensor_str in row_tensors:
                        tensor_str = tensor_str.strip(' []')  # Remove brackets and extra spaces
                        if tensor_str:  # Only process non-empty strings
                            tensor = list(map(int, tensor_str.split(',')))# for x in tensor_str.split('], [').split(',')
                            #tensor = [list(map(int, x.split(','))) for x in tensor_str.split('], [')]
                            temp_tensors.append(tensor)

                            # Check if we have a complete 3x3 tensor
                            if len(temp_tensors) == 3:
                                current_tensor.append(temp_tensors)
                                temp_tensors = []

        # Append the last tensor if exists
        if current_tensor:
            tensor_list.append(torch.tensor(current_tensor, dtype=torch.int64))
            final_list.append(torch.stack(tensor_list))
        elif current_conv_bool_tensor:
            tensor_conv_bool_list.append(torch.tensor(current_conv_bool_tensor, dtype=torch.bool))
            final_list.append(torch.stack(tensor_conv_bool_list))
        elif tensor_bool_list:
            final_list.append(torch.stack(tensor_bool_list))

    return final_list