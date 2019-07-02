## web
## https://github.com/chenyuntc/pytorch-book/blob/master/chapter2-%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8/chapter2:%20PyTorch%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8.ipynb


from __future__ import print_function
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


print(t.__version__)

x = t.Tensor(5,3)
x = t.Tensor([[1,2],[3,4]])
print(x)

x = t.rand(5,3)
print(x)
print(x.size())
print(x.size(0))

y = t.rand(5, 3)
print(x + y)

print(t.add(x,y))

result = t.Tensor(5,3)
t.add(x,y, out = result)

print(result)
print(result[:,1])


a = np.ones(5)
b = t.from_numpy(a)
print(a)
print(b)
scalar = b[0]
print(scalar)
print(scalar.size())

tensor = t.tensor([3, 4])
old_tensor = tensor
new_tensor = t.tensor(old_tensor)
new_tensor[0] = 1111
print(old_tensor, new_tensor)

#device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#x = x.to(device)
#y = y.to(device)
#z = x+y
#print(z)

x = t.ones(2, 2, requires_grad=True)
print(x)

y = x.sum()
print(y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

params = list(net.parameters())
print(len(params))

for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

input = t.randn(1, 1, 32, 32)
print(input)
out = net(input)
print(out.size())

output = net(input)
target = t.arange(0, 10).view(1, 10)
criterion = nn.MSELoss()
print(output)
print(target)
loss = criterion(output, target.float())
print(loss)








