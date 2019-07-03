from __future__ import print_function
import torch as t

print(t.__version__)

a = t.Tensor(2, 3)
print(a)

b = t.Tensor([[1,2,3],[4,5,6]])
print(b)

print(b.tolist())

b_size = b.size()
print(b_size)
print(b.numel())


c = t.Tensor(b_size)
d = t.Tensor((2,3))
print(c, d)
print(c.shape)
print(t.randn(2,3))
print(t.randperm(5))
print(t.eye(3,3,dtype = t.int))

a = t.arange(0, 6)
print(a.view(2,3))
b = a.view(-1, 2)
print(b.shape)

a = t.randn(3, 4)
print(a)
print(a[0])
print(a[:,0])
print(a[0][2])
print(a[0,-1])
print(a[:2])

x = t.arange(0, 27).view(3,3,3)
print(x)

print(x[[1,2],[1,2],[2,0]])


b = t.ones(2, 3)
print(b.sum(dim = 0, keepdim = True))
print(b.sum(dim = 0, keepdim = False))
print(b.sum(dim = 1))

a = t.arange(0, 6).view(2, 3)
print(a)
a.cumsum(dim = 1)


a = t.linspace(0,15,6).view(2,3)

b = t.linspace(15,0,6).view(2,3)

print(a>b)
print(a[a>b])

t.max(a)

print(b)
print(t.max(b, dim = 1))

import numpy as np
a = np.ones([2,3], dtype = np.float32)
print(a)

b1 = t.from_numpy(a)
print(b)

b2 = t.Tensor(a)
print(b)

a[0,1] = 100

print(b1)
print(b2)


a = t.arange(0, 6)
print(a.storage())

b = a.view(2,3)
print(b.storage())

import torch as t
#%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
device = t.device('cpu')

t.manual_seed(1000)

def get_fake_data(batch_size = 8):
    x = t.rand(batch_size, 1, device = device)*5
    y = x*2 + 3 + t.randn(batch_size, 1, device = device)
    return x, y

x, y = get_fake_data(batch_size = 16)
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())

print(x)
print(y)

w = t.rand(1,1).to(device)
b = t.zeros(1,1).to(device)

lr = 0.2

for ii in range(500):
    x, y = get_fake_data(batch_size = 40)
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5*(y_pred - y)**2
    loss = loss.mean()
    dloss = 1
    dy_pred = dloss * (y_pred - y)
    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    w.sub_(lr*dw)
    b.sub_(lr*db)

    if ii%50 == 0:
        display.clear_output(wait=True)
        x = t.arange(0, 40).view(-1, 1)
        x = t.arange(0, 40).view(-1, 1)
        plt.plot(x.cpu().numpy(), y.cpu().numpy())
        x2, y2 = get_fake_data(batch_size=32) 
        plt.scatter(x2.numpy(), y2.numpy())
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.show()

print('w:',w.item(), 'b:',b.item())














