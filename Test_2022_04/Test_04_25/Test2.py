# -*- coding:utf-8 -*-
# 作者：chy
# 联系方式：1945942166@qq.com

#使用pytorch完后线性回归

import torch

torch.manual_seed(42)

x = torch.rand(100,1)
epsilon = torch.randn(100,1)


y = 3*x+1+0.1*epsilon   #k=3  b=1
x_train,x_valid = x[:80],x[80:]
y_train,y_valid = y[:80],y[80:]

EPOCHS = 2000
LR = 0.1

k = torch.randn(1,requires_grad=True,dtype = torch.float)
b = torch.randn(1,requires_grad=True,dtype = torch.float)
#参数需要进行记录梯度requires_grad=True，才能进行梯度下降

print(f'The initial values of k,b is {k},{b}')

for epoch in range(EPOCHS):
    y_hat = k*x_train+b
    error = y_train-y_hat
    loss = (error**2).mean()


    loss.backward() #自动进行梯度下降

    with torch.no_grad():   #上下文管理器,别包裹的程序段梯度不会被跟踪，提高运行速度减少对内存的消耗
        k -=k.grad*LR
        b -=b.grad*LR

    b.grad.zero_()  #变量梯度清零
    k.grad.zero_()

print(f'The final values of k,b is {k},{b}')
