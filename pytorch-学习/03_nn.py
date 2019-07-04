# -*- coding: utf-8 -*

import torch as t
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage



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






