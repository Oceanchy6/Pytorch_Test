# -*- coding:utf-8 -*-
# 作者：chy
# 联系方式：1945942166@qq.com
# Pytorch_Test




##之前的测试都是所有数据放在一起处理，处理完才更新参数权重，称为梯度下降（GD）,当数据量较大时，速度太慢。于是有了只用一个样本就更新权重的
##随机梯度下降（SGD),同时为了提升稳定性，迷你批梯度下降(mini-batch Gradient Descent)很常用

##其中DataLoader可以生成mini_batch,其中Dataset是基础

##学会定义自己的Dataset类很重要，它必须包含三个方法：

# __init__(self，x,y)
# __len__(self)
# __geyitem__(self.idx)   读取特定编号(idx)的数据


import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader






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

#修改版4.0
# 定义Dataset
class ChyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


training_data = ChyDataSet(x_train, y_train)

# 定义Dataloader,Batch Size一般为2的nc次方

train_loader = DataLoader(dataset=training_data, batch_size=16, shuffle=True)
# 注意：训练集一般为了梯度下降表现更好，一般shuffle=True，意为随机抽取，而验证集则没有这类要求

#注意：与批梯度下降相比，mini_batch GD要多一层循环

model.train()

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

training_step = make_train_step(optimizer,model,loss_fn)

for epoch in range(EPOCHS):
    mini_batch_losses = []
    for x_batch,y_batch in train_loader:    #多的一层循环
        loss = training_step(x_batch,y_batch)
        mini_batch_losses.append(loss)

print(model.state_dict())
print(mini_batch_losses)














