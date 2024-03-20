import torch
import sys

import patternManager

def main():

    tensor = torch.load('test_tensor.pt')
    tensor_v2 = torch.randn(6, 1, 3, 3)
    tensor_v3 = torch.randn(16, 6, 3, 3)

    tensor_list = [tensor, tensor_v2, tensor_v3]

    ptManager = patternManager.PatternManager()

    ptManager.assign_patterns_to_tensors(tensor_list)

    ptManager.count_pattern_assignments()


    #print(ptManager.get_single_pattern_mask(layer_index=2).shape)
    #print(ptManager.count_pattern_assignments())
    # print(ptManager.get_pattern_masks())
    #print(ptManager.convert_to_tensors()[2].shape)


    print('-------')
    for i in range(120):
        #low_index = ptManager.reduce_available_patterns()
        ptManager.update_pattern_assignments(tensor_list)

    print(ptManager.available_patterns_indices)
    print(ptManager.tensor_assignments)
    print(ptManager.pattern_counts)

    print(type(ptManager.get_single_pattern_mask(0)[0]))

    print('----------')
    temp_masks = ptManager.get_pattern_masks()
    print(sys.getsizeof(temp_masks))
    print(ptManager.get_pattern_masks()[2].shape)


    #print(low_index)



    pass



if __name__ == '__main__':
    main()
#%%
