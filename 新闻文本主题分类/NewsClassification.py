#!/usr/bin/env python3.7
# -*-coding:utf-8 -*-
# @Time  : 2020-10-29 10:27
# Author : yuwenqi
"""新闻数据的模型训练及测试"""
import numpy as np
# 加载训练集
train_data = np.load("train_lda.npz")
train_text = train_data['data']
train_label = train_data['label']

# 加载验证集
val_data = np.load("val_lda.npz")
val_text = val_data['data']
val_label = val_data['label']

# 加载测试集
test_data = np.load("test_lda.npz")
test_text = test_data['data']
test_label = test_data['label']

# 打印各数据集和标签的维度
print(train_text.shape, train_label.shape, val_text.shape, val_label.shape, test_text.shape, test_label.shape)

from sklearn.externals import joblib
# 加载向量化模型和降维模型
model_vect = joblib.load("model_vect")
model_reduce = joblib.load("model_reduce")


# 在该数据集上应用KNN算法，并搜索最佳超参数K
from sklearn.neighbors import KNeighborsClassifier
import  matplotlib.pyplot as plt
# 设定K的取值范围
k_range = range(1, 31)
# 模型在验证集上的准确率数组
acc_knn = []
# 让K依次取1到30之间的一个整数
for k in k_range:
    # 在K为定值时取得模型
    knn = KNeighborsClassifier(n_neighbors=k)
    # 在训练集上训练
    knn.fit(train_text, train_label)
    # 得到模型在验证集上的准确率值
    score_knn = knn.score(val_text, val_label)
    # 放入准确率数组
    acc_knn.append(score_knn)
plt.plot(k_range, acc_knn)
plt.xlabel('Value of K')
plt.ylabel('accuracy')
plt.show()
print('best super-param of K:', k_range[np.argmax(acc_knn)])


# 在该数据集上应用决策树算法，并搜索最佳超参数max_depth
from sklearn.tree import DecisionTreeClassifier
# 树深度数组
depth_dt = [10, 20, 50, 100, 200]
# 准确率数组
acc_dt = []
# 树深度取以上五个值
for i in depth_dt:
    # 取得深度为i的决策树模型
    dt = DecisionTreeClassifier(max_depth=i)
    # 在训练集上训练
    dt.fit(train_text, train_label)
    # 得到模型在验证集上的准确率值
    score_dt = dt.score(val_text, val_label)
    # 放入准确率数组
    acc_dt.append(score_dt)
plt.plot(depth_dt, acc_dt)
plt.xlabel('depth of tree')
plt.ylabel('Acc')
plt.show()
print('best super-param of max_depth:', depth_dt[np.argmax(acc_dt)])


# 在该数据集上应用随机森林算法，并搜索最佳超参数max_depth
from sklearn.ensemble import RandomForestClassifier
# rf深度数组
depth_rf = [10, 20, 50, 100, 200]
# 准确率数组
acc_rf = []
# 树深度取以上五个值
for j in depth_rf:
    # 取得深度为j的RF模型
    rf = RandomForestClassifier(max_depth=j)
    # 在训练集上训练
    rf.fit(train_text, train_label)
    # 得到模型在验证集上的准确率值
    score_rf = rf.score(val_text, val_label)
    # 放入准确率数组
    acc_rf.append(score_rf)
plt.plot(depth_rf, acc_rf)
plt.xlabel('Depth')
plt.ylabel('Acc')
plt.show()
print('best super-param of max_depth:', depth_rf[np.argmax(acc_rf)])


# 在该数据集上应用随机森林算法，并搜索最佳超参数n_estimators
# 树棵树数组
num_rf = [20, 50, 100, 200, 300, 500]
# 准确率数组
acc_rf1 = []
# 树棵树取以上六个值
for n in num_rf:
    # 取得棵树为n的rf模型
    rf1 = RandomForestClassifier(n_estimators=n)
    # 在训练集上训练
    rf1.fit(train_text, train_label)
    # 得到模型在验证集上的准确率值
    score_rf1 = rf1.score(val_text, val_label)
    # 放入准确率数组
    acc_rf1.append(score_rf1)
plt.plot(num_rf, acc_rf1)
plt.xlabel('number of trees')
plt.ylabel('Acc')
plt.show()
print('best super-param of n_estimators:', num_rf[np.argmax(acc_rf1)])