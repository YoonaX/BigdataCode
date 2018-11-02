# -*- coding: utf-8 -*-

from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import pydotplus


def outlook_type(s):
    it = {'sunny': 1, 'overcast': 2, 'rainy': 3}
    return it[s]


def temperature(s):
    it = {'hot': 1, 'mild': 2, 'cool': 3}
    return it[s]


def humidity(s):
    it = {'high': 1, 'normal': 0}
    return it[s]


def windy(s):
    it = {'TRUE': 1, 'FALSE': 0}
    return it[s]


def play_types(s):
    it = {'yes': 1, 'no': 0}
    return it[s]


play_feature_E = ('outlook', 'temperature', 'humidity', 'windy')
play_class = ('yes', 'no')

# 1、读入数据，并将原始数据中的数据转换为数字形式
# data = np.loadtxt("play.txt", delimiter=" ", dtype=str, converters={0: outlook_type, 1: temperature, 2: humidity,
#                                                                              3: windy, 4: play_types})
data = np.loadtxt("play.txt", delimiter=" ", dtype=float)

x, y = data[:, :-1], data[:, -1]

print(x, y)

# 2、拆分训练数据于测试数据，为了进行交叉验证
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# 3、使用Gini作为划分标准，对决策树进行训练
Gini = tree.DecisionTreeClassifier(criterion='gini')
print(Gini)
Gini.fit(x_train, y_train)

# 4、把决策树结构写入文件
dot_data = tree.export_graphviz(Gini, out_file="tree.dot", feature_names=play_feature_E, class_names=play_class, filed=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
print(Gini.feature_importances_)

# 5、使用训练数据预测，预测结果完全正确
answer = Gini.predict(x_train)
y_train = y_train.reshape(-1)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

# 6、对测试数据进行预测，准确率较低，说明过拟合
answer = Gini.predict(x_test)
y_test = y_test.reshape(-1)
print(answer)
print(y_test)
print(np.mean(answer == y_test))


# DecisionTreeClassfier()
