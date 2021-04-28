#!/usr/bin/env python3.7
# -*-coding:utf-8 -*-
# @Time  : 2020-10-31 11:24
# Author : yuwenqi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC

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

"""在该数据集上应用多种分类算法并比较精度和速度，从而得出较好的分类模型"""
# 模型列表
models1 = [KNeighborsClassifier(n_neighbors=10),
           DecisionTreeClassifier(max_depth=50),
           RandomForestClassifier(max_depth=50, n_estimators=100),
           AdaBoostClassifier(),
           LogisticRegression(),
           GaussianNB(),
           MultinomialNB(),
           SVC()]

import time

# 在该数据集上依次应用各个模型，在训练集上训练，在测试集上测试，并记录时间和精度
for model in models1:
    model_name = model.__class__.__name__
    t1 = time.time()
    model.fit(train_text, train_label)
    train_score = model.score(train_text, train_label)
    t2 = time.time()
    test_score = model.score(test_text, test_label)
    t3 = time.time()
    print('{} 模型训练耗时：{}'.format(model_name, t2 - t1))
    print('{} 模型测试耗时: {}'.format(model_name, t3 - t2))
    print('{} 模型训练测试总耗时: {}'.format(model_name, t3 - t1))
    print('{} 模型训练集精度: {}'.format(model_name, train_score))
    print('{} 模型测试集精度: {}'.format(model_name, test_score))

# 对KNN模型、决策树模型、随机森林模型、逻辑回归模型取不同的参数，再次比较精度和速度
# 模型列表
models2 = [KNeighborsClassifier(n_neighbors=5),
           DecisionTreeClassifier(max_depth=25),
           RandomForestClassifier(max_depth=25, n_estimators=50),
           LogisticRegression(C=0.1)]

# 在该数据集上依次应用各个模型，在训练集上训练，在测试集上测试，并记录时间和精度
for model in models2:
    model_name = model.__class__.__name__
    t1 = time.time()
    model.fit(train_text, train_label)
    train_score = model.score(train_text, train_label)
    t2 = time.time()
    test_score = model.score(test_text, test_label)
    t3 = time.time()
    print('{} 模型训练耗时: {}'.format(model_name, t2 - t1))
    print('{} 模型测试耗时: {}'.format(model_name, t3 - t2))
    print('{} 模型训练测试总耗时: {}'.format(model_name, t3 - t1))
    print('{} 模型训练集精度: {}'.format(model_name, train_score))
    print('{} 模型测试集精度: {}'.format(model_name, test_score))


"""保存训练好的各个分类模型"""
from sklearn.externals import joblib
# 保存训练好的各个分类模型
joblib.dump(models2[0], "KNN")
joblib.dump(models2[1], "DT")
joblib.dump(models2[2], "RF")
joblib.dump(models2[3], "LR")
joblib.dump(models1[3], "Ada")
joblib.dump(models1[5], "GNB")
joblib.dump(models1[6], "MNB")
joblib.dump(models1[7], "SVC")