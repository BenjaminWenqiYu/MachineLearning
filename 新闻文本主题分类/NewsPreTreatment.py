#!/usr/bin/env python3.7
# -*-coding:utf-8 -*-
# @Time  : 2020-10-28 11:26
# Author : yuwenqi
"""新闻数据预处理过程：加载数据、数据清洗、文本向量化、数据降维、数据保存"""
from sklearn.feature_extraction.text import CountVectorizer
# 获取CountVectorizer,用于文本向量化
vect_method = CountVectorizer()

from sklearn.decomposition import LatentDirichletAllocation
# 获取LatentDirichletAllocation,用于文本降维；隐变量设置为64，即主题
de_method = LatentDirichletAllocation(n_components=64)

# 创建过滤特殊符号、标点和数字的正则字符串
import re
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

# 读取文本文件
def loadDataSet(textdir):
    # 文本集
    data_list = []
    # 标签集
    label_list = []
    # 标签名称集
    label_name_list = []
    # 读取数据
    file = open(textdir, "r", encoding="utf-8-sig", errors="ignore")

    # 读取训练集文本，并把标签和新闻文本分别存入相应的数组
    # 依次读取每行
    for line in file.readlines():
        # 删除行首或行尾的空白符，并按制表符区分标签和新闻文本
        content = line.strip().split("\t")
        # 过滤特殊符号、标点和数字等
        content_re = re.sub(r1, "", content[1])
        # jieba分词
        import jieba
        content_cut = jieba.lcut(content_re)
        # 去掉停用词
        content_rm_stopwords = rm_stopwords(content_cut, stop_word_list)
        # 生成文本集和标签集
        data_list.append(" ".join(content_rm_stopwords))
        label_name_list.append(content[0])

    # 打印分词后的第一个文本
    print(data_list[0])

    # 关闭文件
    file.close()

    # 生成标签集字典
    label_dict = dict(zip(set(label_name_list), range(len(set(label_name_list)))))
    # 打印标签集字典
    print(label_dict)

    # 生成数字化的标签集
    import numpy as np
    label_list = np.array([label_dict[itr] for itr in label_name_list])
    # 打印数字化的标签集
    print(label_list)

    # 返回文本集和标签集
    return data_list, label_list


# 训练集文本向量化
def train_wordsVect(data_list):
    # 在data_list上进行文本向量化的操作，训练且转换
    text_wordsVect = vect_method.fit_transform(data_list)
    print("向量化完成")
    # 打印训练集文本向量化后的维度
    print(text_wordsVect.shape)
    # 返回向量化后的训练集文本
    return text_wordsVect


# 训练集文本降维
def train_dim_reduce(text_words_vect):
    # 给向量化的文本降维，训练且转换
    text_dim_reduce = de_method.fit_transform(text_words_vect)
    print("降维完成")
    # 打印训练集文本经预处理后的向量维度
    print(text_dim_reduce.shape)
    # 返回降维后的训练集文本向量
    return text_dim_reduce


# 读取训练集文本文件并进行预处理
def train_text_pretreatment(textdir):
    import time
    # 记录起始时间
    t1 = time.time()
    # 创建数据集和标签
    data_list, label_list = loadDataSet(textdir)
    t2 = time.time()
    # 文本向量化
    text_wordsvect = train_wordsVect(data_list)
    t3 = time.time()
    # 文本降维
    text_dim_reduce = train_dim_reduce(text_wordsvect)
    t4 = time.time()
    # 计算文本预处理所耗时间
    print('文本预处理总耗时：{}'.format(t4 - t1))
    print('读取数据耗时：{}'.format(t2 - t1))
    print('文本向量化耗时: {}'.format(t3 - t2))
    print('文本降维耗时: {}'.format(t4 - t3))
    # 返回标签和预处理后的数据集
    return text_dim_reduce, label_list


# 测试集/验证集文本向量化
def wordsVect(data_list):
    # 在data_list上进行文本向量化的操作，仅转换
    text_words_vect = vect_method.transform(data_list)
    print('向量化完成')
    # 打印测试集/验证集文本经预处理后的向量维度
    print(text_words_vect.shape)
    # 返回向量化后的测试集/验证集文本
    return text_words_vect


# 测试集/验证集文本降维
def dim_reduce(text_words_vect):
    # 给向量化的文本降维，仅转换
    text_dim_reduce = de_method.transform(text_words_vect)
    print('降维完成')
    # 打印测试集/验证集文本经预处理后的向量维度
    print(text_dim_reduce.shape)
    # 返回降维后的训练集文本向量
    return text_dim_reduce


# 读取测试集/验证集文本文件并进行预处理
def text_pretreatment(textdir):
    import time
    # 记录起始时间
    t1 = time.time()
    # 创建数据集和标签
    data_list, label_list = loadDataSet(textdir)
    t2 = time.time()
    # 文本向量化
    text_words_vect = wordsVect(data_list)
    t3 = time.time()
    # 文本降维
    text_dim_reduce = dim_reduce(text_words_vect)
    t4 = time.time()
    # 计算文本预处理所耗时间
    print('文本预处理总耗时: {}'.format(t4 - t1))
    print('读取数据耗时: {}'.format(t2 - t1))
    print('文本向量化耗时: {}'.format(t3 - t2))
    print('文本降维耗时: {}'.format(t4 - t3))
    # 返回标签和预处理后的数据集
    return text_dim_reduce, label_list


import numpy as np
# 对训练集文本进行预处理
train_text_dim_reduce, train_label_list = train_text_pretreatment("cnews.train.txt")
# 保存降维后的训练集向量
np.savez("train_lda.npz", data=train_text_dim_reduce, label=train_label_list)

# 对验证集文本进行预处理
val_text_dim_reduce, val_label_list = text_pretreatment("cnews.val.txt")
# 保存降维后的验证集向量
np.savez('val_lda.npz', data=val_text_dim_reduce, label=val_label_list)

# 对测试集文本进行预处理
test_text_dim_reduce, test_label_list = text_pretreatment("cnews.test.txt")
# 保存降维后的测试集向量
np.savez('test_lda.npz', data=test_text_dim_reduce, label=test_label_list)

from sklearn.externals import joblib
# 保存向量化模型和降维模型
joblib.dump(vect_method, 'model_vect')
joblib.dump(de_method, 'model_reduce')
