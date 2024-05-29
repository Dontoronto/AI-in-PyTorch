import torch


tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

print("dim=0 tests:")
unbind_sample = torch.unbind(tensor)
print(unbind_sample)
print("-----------")

test_1, test_2, test_3 = torch.unbind(tensor)

print(test_1)
print(test_2)
print(test_3)

print("-----")
print("dim=1 tests:")

test_1, test_2, test_3 = torch.unbind(tensor, dim=1)

print(test_1)
print(test_2)
print(test_3)

