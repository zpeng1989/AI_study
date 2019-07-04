# -*- coding: utf-8 -*

import torch as t
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from collections import OrderedDict
from torch import optim


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))
    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)



layer = Linear(4,3)
input = t.randn(2,4)
print(input)
output = layer(input)
print(output)

for name, parameter in layer.named_parameters():
    print(name, parameter)



class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)

perceptron = Perceptron(3,4,1)
for name, param in perceptron.named_parameters():
    print(name, param.size())



to_tensor = ToTensor()
to_pil = ToPILImage()
lena = Image.open('index.png')
#lena.show()

input = to_tensor(lena).unsqueeze(0)

kernel = t.ones(3,3)/-9.
kernel[1][1] = 1
conv = nn.Conv2d(1,1,(3,3),1, bias = False)
conv.weight.data = kernel.view(1,1,3,3)

out = conv(input)
#to_pil(out.data.squeeze(0)).show()


input = t.randn(2,3)
linear = nn.Linear(3,4)
h = linear(input)
print(h)

bn = nn.BatchNorm1d(4)
bn.weight.data = t.ones(4)*4
bn.bias.data = t.zeros(4)

bn_out = bn(h)
print(bn_out.mean(0))
print(bn_out.var(0, unbiased = False))

dropout = nn.Dropout(0.5)
o = dropout(bn_out)
print(o)

relu = nn.ReLU(inplace = True)
input = t.randn(2,3)
print(input)
output = relu(input)
print(output)



###################################


net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3,3,3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

net2 = nn.Sequential(
        nn.Conv2d(3,3,3),
        nn.BatchNorm2d(3),
        nn.ReLU()
        )

net3 = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3,3,3)),
        ('bn1', nn.BatchNorm2d(3)),
        ('relu1', nn.ReLU())
        ]))

print('net1', net1)
print('net2', net2)
print('net3', net3)


print(net1.conv, net2[0], net3.conv1)


input = t.rand(1,3,4,4)
output = net1(input)
print(output)
output = net2(input)
print(output)
output = net3(input)
print(output)

output = net3.relu1(net1.batchnorm(net1.conv(input)))
print(output)

### RNN
print('++++++++++++++++++++++++++++')
print('----------  RNN  -----------')
print('____________________________')

t.manual_seed(1000)
input = t.randn(2,3,4)
lstm = nn.LSTM(4,3,1)
h0 = t.randn(1,3,3)
c0 = t.randn(1,3,3)
out, hn = lstm(input, (h0, c0))
print(input)
print(out)


lstm = nn.LSTMCell(4,3)
hx = t.randn(3,3)
cx = t.randn(3,3)
out = []
for i_ in input:
    hx, cx = lstm(i_, (hx,cx))
    out.append(hx)
t.stack(out)



embedding = nn.Embedding(4,5)
embedding.weight.data = t.arange(0,20).view(4,5)
print(embedding)

input = t.arange(3,0,-1).long()
output = embedding(input)
print(output)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
                        nn.Conv2d(3, 6, 5),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(6, 16, 5),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2)
                        )
        self.classifier = nn.Sequential(
                        nn.Linear(16*5*5, 120),
                        nn.ReLU(),
                        nn.Linear(120, 84),
                        nn.ReLU(),
                        nn.Linear(84, 10)
                        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        return x

net = Net()
print(net)


optimizer = optim.SGD(params = net.parameters(), lr = 1)
optimizer.zero_grad()

input = t.randn(1, 3, 32, 32)
print(input)
output = net(input)
output.backward(output)

optimizer.step()

optimizer = optim.SGD([
                {'params': net.features.parameters()},
                {'params': net.classifier.parameters(),'lr':1e-2}
            ], lr = 1e-5)

print(optimizer)



input = t.randn(2, 3)
model = nn.Linear(3, 4)
output1 = model(input)
output2 = nn.functional.linear(input, model.weight, model.bias)
print(output1 == output2)

b = nn.functional.relu(input)
b2 = nn.ReLU()(input)
print(b == b2)

from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.pool(F.relu(self.conv1(x)), 2)
        x = F.pool(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


from torch.nn import init
linear = nn.Linear(3, 4)
t.manual_seed(1)
print(init.xavier_normal_(linear.weight))

import math
t.manual_seed(1)

std = math.sqrt(2)/math.sqrt(7)

linear.weight.data.normal_(0, std)

for name, params in net.named_parameters():
    if name.find('linear')!= -1:
        params[0]
        params[1]
    elif name.find('conv')!= -1:
        pass
    elif name.find('norm')!= -1:
        pass


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.param1 = nn.Parameter(t.rand(3,3))
        self.submodel1 = nn.Linear(3,4)
    def forward(self, input):
        x = self.paraml.mm(input)
        x = self.submodel(x)
        return x

net = Net()
print(net)

print(net._parameters)
print(net._modules)
#print(net.param1)
print(net.param1)

for name, param in net.named_parameters():
    print(name, param.size())

for name, submodel in net.named_modules():
    print(name, submodel)

bn = nn.BatchNorm1d(2)
input = t.rand(3, 2)
output = bn(input)
print(bn._buffers)

input = t.arange(0,12).view(3,4)
model = nn.Dropout()
print('sssssssssssssssssssssssss')
print(model(input.float()))

model.training = False
print(model(input))


## save


t.save(net.state_dict(), 'net.pth')

net2 = Net()
net2.load_state_dict(t.load('net.pth'))

t.save(net, 'net_all.pth')
net2 = t.load('net_all.pth')
print(net2)



from torch import nn
import torch as t
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchanne1,outchanne1, stride = 1, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                        nn.Conv2d(inchanne1, outchanne1, 3, stride, 1, bias = False),
                        nn.BatchNorm2d(outchanne1),
                        nn.ReLU(inplace = True),
                        nn.Conv2d(outchanne1, outchanne1, 3, 1, 1, bias = False),
                        nn.BatchNorm2d(outchanne1)
                        )
        self.right = shortcut
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(3,2,1))
        self.layer1 = self._make_layer(64, 64, 3,stride = 1)
        self.layer2 = self._make_layer(64, 128, 4, stride = 2)
        self.layer3 = self._make_layer(128, 256, 6, stride = 2)
        self.layer4 = self._make_layer(256, 512, 3, stride = 2)
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, inchanne1, outchanne1, block_num, stride = 1):
        shortcut = nn.Sequential(
                nn.Conv2d(inchanne1, outchanne1, 1, stride, bias = False),
                nn.BatchNorm2d(outchanne1))
        layers = []
        layers.append(ResidualBlock(inchanne1, outchanne1, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchanne1, outchanne1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = ResNet()
input = t.randn(1,3,224, 224)
print(model)
o = model(input)
print(o)













