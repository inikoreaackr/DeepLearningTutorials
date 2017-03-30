import torch
from torch.autograd import Variable

print('# Create a variable:')
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

print('# Do an operation of var')
y = x + 2
print(y)

print('# Find its creator')
print(y.creator)

z = y * y * 3
out = z.mean()

print(z, out) # 여러개 출력가능

# http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
print('# Gradients')
out.backward() # is equivalent to doing out.backward(torch.Tensor([1.0]))
print(x.grad)


x = torch.randn(3)
x = Variable(x, requires_grad=True)
print(x)
y = x * 2
#print(y)
#print(y.data)

while y.data.norm() < 1000: #default L2 norm
    y = y * 2

print(y.data.norm())

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
