from __future__ import print_function
import torch as t

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


