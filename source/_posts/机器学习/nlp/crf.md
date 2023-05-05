---
title: CRF笔记
date: 2021-12-17 21:04:50
mathjax: true
---

> 参考: https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541

## 背景
- CRF用于切词, 序列标注等. 
- 普通的分类模型是对单个字词的分类任务, CRF考虑了前后文的联系, 对当前的片段进行分类


### 概述
- 条件随机场是一种判别模型, 用于预测序列. 它使用标签的上下文信息, 从而增加了模型进行预测的信息量.

以下分三个方面:
1. 判别模型和生成模型
2. 条件随机场的数学表达
3. crf和HMM的区别

#### 判别模型和生成模型
- 判别模型, 给出某一个概率的分类. 比如二分类, A的概率是0.7, B类的概率是0.3. 非A则B
- 生成模型, 是基于概率分布. 基于A类的概率分布得到的概率是0.9, B类概率分布得到的结果是0.6, 各项的概率之和大于1
- 判别模型: 贝叶斯; 判别模型: 逻辑回归

#### 条件随机场的数学表达
条件分布建模如下:
$$\hat { y } = \operatorname { argmax } _ { y } P ( y | x )$$

构建特征函数, 特征函数依赖如下输入:
1. 输入序列X
2. 预测点的位置i
3. 前一个位置i-1的label Li-1
4. 当前位置i的label Li

最终构建的特征函数如下:
$$f ( X , i , l _ { i - 1 } , l _ { i } )$$

特征函数的目的是表示数据序列的某些特征. 
比如f (X, i, L{i - 1}, L{i} ) = 1 if L{i - 1} == 名词, else 0 if L{i} ==  动词 
同样的, f (X, i, L{i - 1}, L{i} ) = 1 if L{i - 1} == 动词, else 0 if L{i} == 形容词

每个特征函数都基于上一个词和当前词的标签, 要么是0要么是1. 位了构建条件场, 我们给每个特征函数分配一个权重, 算法将学习这些权重
$$P ( y , X , \lambda ) = \frac { 1 } { Z ( X ) } \operatorname { exp } \{ \sum _ { i = 1 } ^ { n } \sum _ { j } \lambda _ { j } f _ { i } ( X , i , y _ { i - 1 } , y _ { i } ) \}$$

$$Z ( x ) = \sum _ { y ^ { \prime } \in y } \sum _ { i = 1 } ^ { n } \sum _ { j } \lambda _ { j } f _ { i } ( X , i , y _ { i - 1 } ^ { \prime } , y _ { i } ^ { \prime } )$$


接下来位了获得估计参数 （λ）, 使用最大似然估计. 为了后续求偏导方便计算, 先取对数然后求负数.
$$\begin{array}{c}
L(y, X, \lambda)=-\log \left\{\prod_{k=1}^{m} P\left(y^{k} \mid x^{k}, \lambda\right)\right\} \\
=-\sum_{k=1}^{m} \log \left[\frac{1}{Z\left(x_{m}\right)} \exp \left\{\sum_{i=1}^{n} \sum_{j} \lambda_{j} f_{j}\left(X^{m}, i, y_{i-1}^{k}, y_{i}^{k}\right)\right]\right.
\end{array}$$

这个就是损失函数, 我们的目标是 求其最小值时候的λ值. 通过对λ求偏导, 得到:

$$\frac{\partial L(X, y, \lambda)}{\partial \lambda}=\frac{-1}{m} \sum_{k=1}^{m} F_{j}\left(y^{k}, x^{k}\right)+\sum_{k=1}^{m} p\left(y \mid x^{k}, \lambda\right) F_{j}\left(y, x^{k}\right)$$

$$\text { Where: } F_{j}(y, x)=\sum_{i=1}^{n} f_{i}\left(X, i, y_{i-1}, y_{i}\right)$$

然后每次使用 当前的梯度下降去更新当前的λ值.
$$\lambda=\lambda+\alpha\left[\sum_{k=1}^{m} F_{j}\left(y^{k}, x^{k}\right)+\sum_{k=1}^{m} p\left(y \mid x^{k}, \lambda\right) F_{j}\left(y, x^{k}\right)\right]$$

### 总结
使用条件随机场的过程: 先定义所需的特征函数, 然后初始化权重为随机值, 然后迭代应用梯度下降方法, 直到参数值收敛.这里为λ.
CRF与逻辑回归类似, 都是用条件概率分布, 但是CRF在特征函数里引入序列信息拓展了算法

### CRF和HMM的区别
HMM是生成式的, 通过对联合概率分布建模来输出. CRF不依赖独立性假设. HMM可以看作是条件随机场的一种特殊情况. 使用了常数转移概率. HMM是基于朴素贝叶斯的, 朴素贝叶斯可以从逻辑回归中推导出来, CRF 就是从逻辑回归中推导出来的.
