# -*- coding:utf-8 -*-
# 作者：chy
# 联系方式：1945942166@qq.com

#动态计算图
#修改版1.0
from torchviz import make_dot
import torch

torch.manual_seed(42)

x = torch.rand(100,1)
epsilon = torch.randn(100,1)
y = 3*x+1+0.1*epsilon

x_train = x[:80]
y_train = y[:80]

EPOCHS = 2000
LR = 0.1

k = torch.randn(1,requires_grad=True,dtype = torch.float)
b = torch.randn(1,requires_grad=True,dtype = torch.float)
#参数需要进行记录梯度requires_grad=True，才能进行梯度下降

print(f'The initial values of k,b is {k},{b}')

# y_hat = k * x_train + b
# error = y_train - y_hat
# loss = (error ** 2).mean()
#
# g = make_dot(y_hat)
#
# g.view()

import torch.optim as optim
optimizer = optim.SGD([k,b],lr=LR)
#使用优化器代替下面的手动更新代码
# with torch.no_grad():
#     k -= k.grad*LR
#     b -= b.grad*LR

# optimizer.zero_grad()
#使用优化器zero_grad()方法代替之前逐个对变量的梯度清零

for epoch in range(EPOCHS):
    y_hat = k*x_train+b
    error = y_train-y_hat
    loss = (error ** 2).mean()

    loss.backward() #梯度计算

    optimizer.step()    #梯度下降

    optimizer.zero_grad()   #梯度清零

print(f'The final values of k,b is {k},{b}')