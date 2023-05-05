---
title: asr评估
date: 2021-12-15 22:14:26
tags: asr
mathjax: true
---

> 论文 《 Meaning Error Rate: ASR domain-specific metric framework》


# 目标
### 构建一个指标模型, 考虑不同场景、不同词识别错误的不同损失, 用来评估ASR识别质量

### 背景
- 当前已有的ASR评估指标:
    - WER: 字错误率
    - CER: 字符错误率
    - 间接评估法
- 当前指标缺陷:
    - 无法体现出词语的重要性: 不同的词语, 不同的语境被识别错造成的损失不一样
    

### 指标定义
$\alpha ( s , d )$ 定义为
$\alpha _ { i } ( s , d ) = ( \alpha _ { i } ^ { s } , \alpha _ { i } ^ { d } ) , i = 1 \ldots n _ { \alpha }$
代表两个序列之间的映射(s->d), 其中部分可能为空

#### WER定义
$$\operatorname { WER } _ { \text { orig } } ( s , d ) = \operatorname { min } _ { \alpha \in A ( s , d ) } \frac { \sum _ { i = 1 } ^ { | \alpha | } 1 ( \alpha _ { i } ^ { s } \neq \alpha _ { i } ^ { d } ) } { | s | }$$
意思是两个序列对齐的最小diff率

#### WER改进版
$$\operatorname { gWER } ( s , d , h ) = \operatorname { min } _ { \alpha \in A ( s , d ) } \frac { 1 } { \operatorname { max } ( | s | , | d | ) } \sum _ { i = 1 } ^ { | \alpha | } h ( \alpha _ { i } ^ { s } , \alpha _ { i } ^ { d } )$$
相对上面的那个, 原来的两个词不相等就算错, 现在的由一个h(x)函数进行打分, 用来决定不相等的严重程度


### 意义损失
- 不同知识领域标准不同
- 基于人的标注来提供样本
    - 由人来进行打分判断有没有损失意义

#### 意义损失公式
    
$$\operatorname { MERa } ( s , d , \theta ) = \sigma ( \theta _ { 0 } + gWER ( s , d , h _ { \theta _ { 1 } } ) ) )$$

- gWER就是上面的WER改进版 在特定领域的意义损失
- 𝜎(x) 就是Sigmoid 函数用来转化成概率.

#### 交叉熵
$$\left. \begin{array} { l }{ l ( s , d , j , \theta ) = 1 ( j > 0 ) \operatorname { log } ( M E R a ( s , d , \theta ) ) + }\\{ 1 ( j < 0 ) \operatorname { log } ( 1 - \operatorname { MER } a ( s , d , \theta ) ) } \end{array} \right.$$

#### 目标函数
$$\hat { \theta } = \operatorname { arg } \operatorname { max } _ { \theta } \sum _ { ( s , d ) \in X } \frac { 1 } { | J _ { ( s , d ) } | } \sum _ { j _ { l } \in J _ { ( s , d ) } } l ( s , d , j _ { l } , \theta )$$
- 目标函数确定
- 未知变量只有$h _ { \theta _ { 1 } } ( \alpha _ { i } ^ { s } , \alpha _ { i } ^ { d } )$
- 现在就是要做的就是假设一个h(x), 然后求目标函数最小的时候的h(x)参数的最小值
- 论文里h(x)选用了线性模型
$$h _ { \theta _ { 1 } } ( \alpha _ { i } ) = \theta _ { 1 } ^ { T } \psi ( \alpha _ { i } )$$
意思就是每个ai都赋予一个错误程度系数

### 优化算法
- EM优化算法

### 缺点
- 样本需要人标 



