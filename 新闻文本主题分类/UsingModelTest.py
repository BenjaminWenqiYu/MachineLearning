#!/usr/bin/env python3.7
# -*-coding:utf-8 -*-
# @Time  : 2020-10-31 13:41
# Author : yuwenqi
"""模拟实际场景，对两篇新闻进行主题预测"""
import re
import jieba

r1 = "[a-zA-Z0-9\.\!\/_,$%^*()+\"\'\[\]]+|[+——！，。？、：；;《》“”~@#￥%……&*（）]+|[\的\在\了]"

# 创建停用词词汇列表
stop_word_list = set([line.strip() for line in open("stop_words.txt", "r", encoding="utf-8-sig")])


# 去除停用词
def rm_stopwords(words, stop_word_list):
    words_list = list(words)
    for i in range(words_list.__len__())[::-1]:
        word = words_list[i]
        if word in stop_word_list:
            # 去除停用词
            words_list.pop(i)
        elif len(word) == 1:
            # 去除单个字符
            words_list.pop(i)
        elif word == " ":
            # 去除空字符
            words_list.pop(i)
    return words_list


def readData(dir):
    """
    读取新闻文本，分词并去掉停用词
    :param dir: 新闻文本的具体路径
    :return: 分词且去掉停用词后的新闻文本
    """
    content = ""
    file = open(dir, "r", encoding="utf-8-sig")
    # 把文本读入content，变成一行
    for line in file.readlines():
        content += line.strip()
    print(content)
    # 过滤特殊符号、标点和数字等
    content_re = re.sub(r1, "", content)
    # jieba分词
    content_cut = jieba.lcut(content_re)
    # 去掉停用词
    content_rm_stopwords = rm_stopwords(content_cut, stop_word_list)
    return content_rm_stopwords


# 把测试文本放入预测数据列表
predict_data = []
data1 = readData("test1.txt")
predict_data.append(" ".join(data1))
data2 = readData("test2.txt")
predict_data.append(" ".join(data2))
print(predict_data)

from sklearn.externals import joblib

# 加载向量化模型和降维模型
model_vect = joblib.load("model_vect")
model_reduce = joblib.load("model_reduce")
# 文本向量化
predict_WordsVect = model_vect.transform(predict_data)
print(predict_WordsVect.shape)
# 文本降维
predict_DimReduc = model_reduce.transform(predict_WordsVect)
print(predict_DimReduc.shape)

# 在预测数据集上应用多种分类算法,预测新闻主题 from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC

# 模型列表
models = [joblib.load("KNN"),
          joblib.load("DT"),
          joblib.load("RF"),
          joblib.load("Ada"),
          joblib.load("LR"),
          joblib.load("GNB"),
          joblib.load("MNB"),
          joblib.load("SVC")]
# 在该数据集上依次应用各个模型
for model in models:
    model_name = model.__class__.__name__
    predicted_subject = model.predict(predict_DimReduc)
    print('{}:'.format(model_name))
    for number in predicted_subject:
        print(Number2Label[number])

import pyecharts.options as opts
from pyecharts.charts import WordCloud

lda = LatentDirichletAllocation(n_components=10)
data_List, label_List = loadDataSet("cnews.train.txt")
train_vect = model_vect.fit_transform(data_List)
lda.fit_transform(train_vect)
w = lda.components_
for num, topic in enumerate(w):
    print("length of topic:", len(topic))
    idxs = topic.argsort()[::-1]
    words = [vect_method.get_feature_names()[i] for i in idxs[:100]]
    data = [(i, f"{int(s*1000)}") for i, s in zip(words, topic[idxs[:100]])]
    (
        WordCloud()
            .add(series_name="热点分析", data_pair=data, word_size_range=[6, 66])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="热点分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render(f"wordcloud-{num}.html"))
