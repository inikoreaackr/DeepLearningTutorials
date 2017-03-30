# Define the neural network that has some learnable parameters (or weights)
# Iterate over a dataset of inputs
# Process input through the network
# Compute the loss (how far is the output from being correct)
# Propagate gradients back into the network’s parameters
# Update the weights of the network, typically using a simple update rule: weight = weight + learning_rate * gradient

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# torch.nn only supports mini-batches
# The entire torch.nn package only supports inputs
# that are a mini-batch of samples, and not a single sample.

# nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolutional kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # an affine operation: y = Wx + b # 이건 이미지가 5x5 라는 가정이 있는건데..?
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x (batch?, channel, row, col)
        print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) # Max pooling over a (2,2) window
        print(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        print(x.size())
        x = x.view(-1, self.num_flat_features(x)) # Returns a new tensor with the same data but different size.
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x) # 왜 softmax 안하지? 그냥 linear regression이네 (pre activation)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension // 0번째 인덱스가 batch dim인가? 그냥 추측!
        print(size)
        num_features = 1
        for s in size:
            num_features *= s # tensor가 사각형이니 전체 element개수 구하려면 곱해줘야
        return num_features

net = Net()
print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())

input = Variable(torch.randn(1, 1, 32, 32)) # 32 - 5 + 1 = 28, 28 / 2 = 14, 14 - 5 + 1 = 10, 10 / 2 = 5 -> 5*5 탄생
#print(input)
out = net(input)
print(out)

# net.zero_grad()
# out.backward(torch.randn(1, 10))
# print(input.grad)

output = net(input)
target = Variable(torch.range(1, 10))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.creator)
print(loss.creator.previous_functions[0][0])  # Linear
print(loss.creator.previous_functions[0][0].previous_functions[0][0])  # ReLU

net.zero_grad()  # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# learning_rate = 0.01
# for f in net.parameters(): # torch.optim package안에 다양한 optimization 함수가 있음
#     f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
