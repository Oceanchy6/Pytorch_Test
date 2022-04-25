# -*- coding:utf-8 -*-
# 作者：chy
# 联系方式：1945942166@qq.com

#使用numpy徒手完成线性回归
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)   #将随机数的种子设定为42

x = np.random.rand(100,1)
epsilon = np.random.randn(100,1)
y = 3*x+1+0.1*epsilon   #k=3  b=1

x_train,x_valid = x[:80],x[80:]
y_train,y_valid = y[:80],y[80:]

# #数据可视化
# f,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))
# ax1.set_title('training')
# ax1.scatter(x_train,y_train,c = 'blue')
#
# ax2.set_title('validation')
# ax2.scatter(x_valid,y_valid,c = 'green')
# plt.show()

#定义超参数——迭代次数和学习率

EPOCHS = 2000
LR = 0.1

#随机生成一个k和b

k = np.random.randn(1)
b = np.random.randn(1)

for epoch in range(EPOCHS):
    y_hat = k*x_train+b #预测值
    error = y_train-y_hat   #误差
    loss = (error**2).mean() #均方损失函数 MSE

    k_grad =-(x_train*error).mean()  #斜率的梯度
    b_grad = -error.mean() #偏执的梯度

    #梯度下降
    k = k - k_grad* LR
    b = b-b_grad*LR
print(k,b)

# #使用sklearn的线性回归验证咋们手工计算出来的结果
#
# from sklearn.linear_model import LinearRegression
#
# model = LinearRegression()
# model.fit(x_train,y_train)
# print(model.coef_[0],model.intercept_)