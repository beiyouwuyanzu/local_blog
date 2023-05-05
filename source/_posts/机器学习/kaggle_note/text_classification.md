---
title: text_classification
date: 2021-12-13 21:47:55
categories: nlp
---
> 原文链接: https://www.kaggle.com/revathiprakash/jigsaw-ensemble-lb-0-859/notebook

## 阅读笔记

### sklearn库
1. sklearn.ensemble 集成学习的常用库
    + bagging
    + 随机森林
    + adaboost 主要侧重训练前面学习器学错的部分
    + Gradient Tree Boosting, GBDT
    + Histogram-Based Gradient Boosting, 对标LightGBM. 相对GBDT, 适合于大数量样本(上万), 可以自己补充缺失值

2. sklearn.feature_extraction.text
    - 从文本构建向量的库
    - CountVectorizer
        - 相当于把词汇列表的个数展开成向量的维数, 然后统计每个词汇出现的次数
    - HashingVectorizer
        - 优点, 不储存状态, 节省内存, 快
      - 缺点, 没法反解词语. 没有tf-idf特征
    - TfidfVectorizer
        - 将原始文档转化成TF-IDF矩阵
        - 体现每个词在文档中的重要性程度
        

### pandas
1. 某一列的数据分布, 可以直接看hist分布图
    - df_.y.hist(bins = 100)
2. 枚举型的数据, 可以直接用value_counts()看分布统计
    - tmp_df['y'].value_counts()