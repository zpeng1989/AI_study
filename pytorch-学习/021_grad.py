from __future__ import print_function
import torch as t
from torch.autograd import Function

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



