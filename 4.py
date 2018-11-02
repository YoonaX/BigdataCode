# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# 机器学习的普通线性模型、岭回归模型、lasso模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 模型效果评估
from sklearn.metrics import  r2_score

# 导入机器学习相关的数据集
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split



data = np.loadtxt("D:\本科数据挖掘\课件 (1)\housing.data")
table, predict = data[:, :-1], data[:, -1]
# table_train, table_test, predict_train, predict_test = train_test_split(table, predict, test_size=0.2, random_state=42)
table_train = table[:480]
table_test = table[480: 600]
predict_train = predict[:480]
predict_test = predict[480: 600]
# print(X_train)
# Y_train = target[:480]
# X_train = data[:480]
# print(X_train)
# Y_train = target[:480]



# 从datasets模块中导入boston当家数据
boston = datasets.load_boston()
data = boston.data
target = boston.target

# 训练数据
X_train = data[:480]
print(X_train)
Y_train = target[:480]
print(Y_train)

# 测试数据
x_test = data[480: 600]
y_true = target[480: 600]

line = LinearRegression()
ridge = Ridge()
lasso = Lasso()


#
line.fit(X_train, Y_train)
ridge.fit(X_train, Y_train)
lasso.fit(X_train, Y_train)

line_y_pre = line.predict(x_test)
ridge_y_pre = ridge.predict(x_test)
lasso_y_pre = lasso.predict(x_test)

plt.plot(y_true, label='True')
plt.plot(line_y_pre, label='Line', color='r')
plt.plot(ridge_y_pre, label='Ridge', color='y')
# plt.plot(lasso_y_pre, label='Lasso')
plt.legend()
plt.show()


# line.fit(table_train, predict_train)
# ridge.fit(table_train, predict_train)
# lasso.fit(table_train, predict_train)
#
# line_y_pre = line.predict(table_test)
# ridge_y_pre = ridge.predict(table_test)
# lasso_y_pre = lasso.predict(table_test)
#
# plt.plot(predict_train, label='True')
# plt.plot(line_y_pre, label='Line', color='r')
# plt.plot(ridge_y_pre, label='Ridge', color='y')
# # plt.plot(lasso_y_pre, label='Lasso')
# plt.legend()
# plt.show()