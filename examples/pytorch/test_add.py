import torch

a = torch.rand(3)
a = a.cuda()
a = a + 1
