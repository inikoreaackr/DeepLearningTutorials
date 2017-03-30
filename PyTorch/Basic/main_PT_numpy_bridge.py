import torch
import numpy as np

# Numpy Bridge
# Converting a torch Tensor to a numpy array and vice versa is a breeze.
# The torch Tensor and numpy array will share their underlying memory locations,
# and changing one will change the other.

a = torch.ones(5)
print(a)

b = a.numpy() # Share same location but it is a numpy object
print(b)


a.add_(1)
print(a)
print(b)    # see how the numpy array changed in value


# Converting numpy Array to torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)  # see how changing the np array changed the torch Tensor automatically

