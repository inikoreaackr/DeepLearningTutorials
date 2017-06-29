from __future__ import print_function # http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
import torch

# Construct a 5x3 matrix, uninitialized:

x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# Get its size
print(x.size())

# addition syntax1
y = torch.rand(5, 3)
print(x + y)

# addition syntax2
print(torch.add(x, y))

# addition syntax3
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# addtion synatx4
y.add_(x)
print(y)

# Numpy 처럼 쓸 수 있
print(x[:, 1])


# Numpy Bridge
a = torch.ones(5)
print(a)

# This is a numpy bridge
b = a.numpy()
print(b)


print("# From Numpy to torch tensor")
# From Numpy to torch tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# from CPU -> GPU  # like theano
# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y