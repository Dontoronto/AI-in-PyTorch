import torch


a = torch.zeros((1,1,28,28),dtype=torch.float32)
b = torch.ones((1,1,28,28), dtype=torch.float32)

diff = torch.norm(a-b,'fro')
print(diff)

#%%
