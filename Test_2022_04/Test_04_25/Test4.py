# -*- coding:utf-8 -*-
# 作者：chy
# 联系方式：1945942166@qq.com
#快捷键“Crtl + Shift + F10”来运行python代码
#快捷键“Crtl + Shift + F9”来调试python代码

#损失函数以及nn.Module
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.optim as optim


# #定义MSE损失函数
# loss_fn = nn.MSELoss(reduction='mean')
#
# y = torch.tensor([1.0,2.0])
# y_hat = torch.tensor([1.2,1.9])
#
# loss = loss_fn(y,y_hat)
# print(loss)

#使用Torch内置的MSE来代替手动损失函数代码
#loss = loss_fn(y_train,y_hat)

# class myLinearRegerssion(nn.Module):
#     def __init__(self):
#         super(myLinearRegerssion, self).__init__()
#         self.k = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
#         self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
#
#     def forward(self,x):
#         return (self.k*x+self.b)

# #实例化
# model = myLinearRegerssion()
#
# #参数
# print(list(model.parameters()))
#
# print(model.state_dict()) #返回当前参数值,效果跟上面一样

# #修改版2.0
# torch.manual_seed(42)
#
# x = torch.rand(100,1)
# epsilon = torch.randn(100,1)
# y = 3*x+1+0.1*epsilon
#
# x_train = x[:80]
# y_train = y[:80]
#
# EPOCHS = 2000
# LR = 0.1
#
# model = myLinearRegerssion()    #实例化
#
# optimizer = optim.SGD(model.parameters(),lr=LR) #model.parameters()替换k,b
#
# loss_fn = nn.MSELoss(reduction='mean')  #定义损失函数
#
# model.train()   #声明此代码是用来进行训练的，并不是真的模型训练
#
# for epoch in range(EPOCHS):
#     y_hat = model(x_train)
#
#     loss = loss_fn(y_train,y_hat)
#
#     loss.backward() #梯度计算
#
#     optimizer.step()    #梯度下降
#
#     optimizer.zero_grad()   #梯度清零
#
# print(model.state_dict())

##修改版3.0
##利用Pytorch内置的参数代替自己定义的参数k,b

# linear = nn.Linear(1,1)
# print(linear)
# print(linear.state_dict())

#因此可以简化Class的定义

class myLinearRegerssion(nn.Module):
    def __init__(self):
        super(myLinearRegerssion, self).__init__()
        self.linear = nn.Linear(1,1)
        # self.k = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self,x):
        return self.linear(x)
        # return (self.k*x+self.b)
torch.manual_seed(42)

x = torch.rand(100,1)
epsilon = torch.randn(100,1)
y = 3*x+1+0.1*epsilon

x_train = x[:80]
y_train = y[:80]

EPOCHS = 2000
LR = 0.1

model = myLinearRegerssion()    #实例化

optimizer = optim.SGD(model.parameters(),lr=LR) #model.parameters()替换k,b

loss_fn = nn.MSELoss(reduction='mean')  #定义损失函数

model.train()   #声明此代码是用来进行训练的，并不是真的模型训练

##关于pytorch的内容还有很多，例如GPU的使用，DataLoader等
##另外训练过程的循环中的五个语句是固定的模式，使用泛函编程可以进一步简化
#先定义一个高阶函数make_train_step
#该函数的输入是optimizer,model.loss_fn,输出是个函数，其内部的函数定于具体的训练的5步

# 1.计算预测值
# 2.计算损失
# 3.计算梯度
# 4.权重跟新
# 5.梯度清零

def make_train_step(optimizer,model,loss_fn):
    def train_step(x,y):
        y_hat = model(x)
        y_hat = model(x)
        loss = loss_fn(y,y_hat)
        loss.backward() #梯度计算
        optimizer.step()    #梯度下降
        optimizer.zero_grad()   #梯度清零
        return loss.item()
    return train_step
# trainstep = make_train_step(optimizer,model,loss_fn)

losses = []

trainstep = make_train_step(optimizer,model,loss_fn)

for i in range(EPOCHS):
    loss = trainstep(x_train,y_train)
    losses.append(loss)
print(model.state_dict())
print(losses)




