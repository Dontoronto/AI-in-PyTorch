# testmain.py

import torch
import sys, os
sys.path.append(os.getcwd())

import patternManager

def main():

    #tensor = torch.load('test_tensor.pt')
    tensor = torch.randn(1, 1, 3, 3)
    tensor_v2 = torch.randn(1, 6, 3, 3)
    tensor_v3 = torch.randn(16, 6, 3, 3)

    tensor_list = [tensor, tensor_v2, tensor_v3]

    ptManager = patternManager.PatternManager()

    ptManager.assign_patterns_to_tensors(tensor_list)
    # print(ptManager.pattern_counts)
    # print(ptManager.abs_impact_patterns)
    # print(ptManager.avg_impact_patterns)

    for i in range(130):
        ptManager.update_pattern_assignments(tensor_list, min_amount_indices=12)

    temp_masks = ptManager.get_pattern_masks()

    # print("-------")
    # print(sys.getsizeof(temp_masks))
    # print(ptManager.pattern_counts)
    # print(ptManager.abs_impact_patterns)
    # print(ptManager.avg_impact_patterns)

    # print(ptManager.pattern_counts)
    # print(ptManager.abs_impact_patterns)
    # print(ptManager.avg_impact_patterns)




if __name__ == '__main__':
    main()
#%%
