import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(threshold=np.inf)


# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# # target = torch.empty(3, dtype=torch.long).random_(9)
# target = torch.LongTensor([-1,1,2])
# print(target)
# output =torch.nn.functional.cross_entropy(input, target)
# print(output)
# # output.backward()
# print(output.backward())




a=np.asarray([[1,1,1],[2,2,2],[3,3,3]])
a=torch.Tensor(a)
print(a)
a=a+1
print(a)



