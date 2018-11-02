# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import precision_recall_curve  # 准确率与召回率
import numpy as np
# import graphviz

import os

os.environ["PATH"] += os.pathsep + "E:\\Grapvz2.38\\bin"


def get_data():
    file_path = "D:\\本科数据挖掘\\课件 (1)\\iris.txt"
    data = np.loadtxt(file_path, delimiter=",", dtype=str)
    print(data)
    # data = pd.read_excel(file_path)
    # loandata = pd.DataFrame(data)
    # ncol = (len(loandata.keys()))
    # print(ncol)
    # # l = list(data.head(0))  #获取表头
    # # print(l)
    #
    # feature1 = []
    # for i in range(ncol - 1):
    #     feature1.append("feature" + str(i))
    # print(feature1)
    # iris_x = data.iloc[1:, :ncol - 1]  # 此处有冒号，不显示最后一列
    # iris_y = data.iloc[1:, ncol - 1]  # 此处没有冒号，直接定位

    iris_x = data[:, :-1]  # 此处有冒号，不显示最后一列
    iris_y = data[:, -1]  # 此处没有冒号，直接定位

    '''计算到底有几个类别'''
    from collections import Counter
    counter = Counter(iris_y)
    con = len(counter)
    print(counter.keys())
    class_names = []
    for i in range(con):
        class_names.append(list(counter.keys())[i])
    x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y)
    print(x_train)
    print(y_test)
    # return x_train, x_test, y_train, y_test

# def dtfit(x_train, x_test, y_train, y_test):

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    predict_data = clf.predict(x_test)
    predict_proba = clf.predict_proba(x_test)
    from sklearn import metrics
    # Do classification task,
    # then get the ground truth and the predict label named y_true and y_pred
    classify_report = metrics.classification_report(y_test, clf.predict(x_test))
    confusion_matrix = metrics.confusion_matrix(y_train, clf.predict(x_train))
    overall_accuracy = metrics.accuracy_score(y_train, clf.predict(x_train))
    acc_for_each_class = metrics.precision_score(y_train, clf.predict(x_train), average=None)
    overall_accuracy = np.mean(acc_for_each_class)
    print(classify_report)

    import pydotplus
    feature = ['spal_length', 'speal_width', 'petal_length', 'petal_width']
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature, filled=True, rounded=True,
                                    special_characters=True, precision=4)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("workiris1.pdf")
    return classify_report



if __name__ == "__main__":
    x = get_data()
    print(x)
    # dtfit(x_train, x_test, y_train, y_test)