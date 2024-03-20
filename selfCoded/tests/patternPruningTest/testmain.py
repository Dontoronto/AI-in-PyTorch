# testmain.py

import torch
import sys, os
sys.path.append(os.getcwd())

import patternManager

def main():

    tensor = torch.load('test_tensor.pt')
    tensor_v2 = torch.randn(6, 1, 3, 3)
    tensor_v3 = torch.randn(16, 6, 3, 3)

    tensor_list = [tensor, tensor_v2, tensor_v3]

    ptManager = patternManager.PatternManager()

    ptManager.assign_patterns_to_tensors(tensor_list)

    for i in range(120):
        ptManager.update_pattern_assignments(tensor_list)

    temp_masks = ptManager.get_pattern_masks()

    print(sys.getsizeof(temp_masks))




if __name__ == '__main__':
    main()
#%%
