import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):

    def __init__(self):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in here
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5) # Filter Shape 지정할 수 있는지..?
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # it's the forward function that defines the network structure
    # we're accepting only a single input in here, but if you want,
    # feel free to use more
    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))

        # in your model definition you can go full crazy and use arbitrary
        # python code to define your model structure
        # all these are perfectly legal, and will be handled correctly
        # by autograd:
        # if x.gt(0) > x.numel() / 2:
        #      ...
        #
        # you can even do a loop and reuse the same module inside it
        # modules no longer hold ephemeral state, so you can use them
        # multiple times during your forward pass
        # while x.norm(2) < 10:
        #    x = self.conv1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net = MNISTConvNet()
print(net)

# input
input = Variable(torch.randn(1, 1, 28, 28))
out = net(input)
print(out.size())

# Define a dummy target
target = Variable(torch.LongTensor([3])) # 3
print("target")
print(target)
print("out")
print(out)
loss_fn = nn.CrossEntropyLoss()  # LogSoftmax + ClassNLL Loss
err = loss_fn(out, target)
err.backward()

print(err)
print(net.conv1.weight.grad.size())
print('')
print("net.conv1.weight!!")
print(net.conv1.weight) #바로 Weight 볼수 있음!!
print("net.conv1.weight[0][0][0][0]!!")
print(net.conv1.weight[0][0][0][0]) #바로 Weight 볼수 있음!!
# print("net.conv1.weight[0][0][0][0] Change!!")
# net.conv1.weight[0][0][0][0] = 1 # Fail... Is there any way to change value in weight matrix?
# print(net.conv1.weight[0][0][0][0]) #바로 Weight 볼수 있음!!

print(net.conv1.weight.data.norm())  # norm of the weight
print(net.conv1.weight.grad.data.norm())  # norm of the gradients

# Hooks!!
# You can register a function on a Module or a Variable.
# The hook can be a forward hook or a backward hook.

def print_norm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

net.conv2.register_forward_hook(print_norm)

#out = net(input)

def print_grad_norm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())

net.conv2.register_backward_hook(print_grad_norm)

out = net(input)
err = loss_fn(out, target)
print('')
err.backward()