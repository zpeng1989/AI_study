from __future__ import print_function
import torch as t
from torch.autograd import Function
import numpy as np


a = t.randn(3, 4, requires_grad = True)
b = t.zeros(3, 4).requires_grad_()

c = a.add(b)

print(c)

d = c.sum()
d.backward()
print(d)

print(d.requires_grad)
print(a.grad)

def f(x):
    y = x**2*t.exp(x)
    return y

def gradf(x):
    dx = 2*x*t.exp(x) + x**2*t.exp(x)
    return dx

x = t.randn(3,4, requires_grad = True)
y = f(x)

print(y)

y.backward(t.ones(y.size()))
print(x.grad)
gradf(x)

class MultiplyAdd(Function):
    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w,x)
        output = w * x + b
        return output
    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b


x = t.ones(1)
w = t.rand(1, requires_grad = True)
b = t.rand(1, requires_grad = True)

z = MultiplyAdd.apply(w,x,b)

print(z.backward())

print(x.grad, w.grad, b.grad)

###################################

print('++++++++++++++++++++++++++++++++++++++++')

x = t.tensor([5.], requires_grad = True)
y = x**2
grad_x = t.autograd.grad(y, x, create_graph = True)
print(grad_x)

grad_grad_x = t.autograd.grad(grad_x[0], x)

print(grad_grad_x)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        output = 1/(1+t.exp(-x))
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_x = output*(1 - output)* grad_output
        return grad_x


test_input = t.randn(3, 4, requires_grad = True)
t.autograd.gradcheck(Sigmoid.apply, (test_input, ), eps = 1e-3)


t.manual_seed(1000)

def get_fake_data(batch_size = 8):
    x = t.rand(batch_size, 1) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1)
    return x, y

x, y = get_fake_data()

w = t.rand(1, 1, requires_grad = True)
b = t.zeros(1, 1, requires_grad = True)
losses = np.zeros(500)
lr = 0.005

for ii in range(500):
    x, y = get_fake_data(batch_size = 32)
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    losses[ii] = loss.item()
    print(loss.backward())
    print(w.grad.data)
    print(b.grad.data)
    w.data.sub_(lr*w.grad.data)
    b.data.sub_(lr*b.grad.data)
    w.grad.data.zero_()
    b.grad.data.zero_()


print(w.item(), b.item())












