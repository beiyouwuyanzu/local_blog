---
title: 大模型优化方法Toolformer
date: 2023-09-04 23:48:31
---

## Toolformer
> Toolformer是由Meta于2023年2月的论文《Toolformer: Language Models Can Teach Themselves to Use Tools》中提出的一种解决LLM，尤其是10b以下规模的LLM能力局限的一种方法。
> https://arxiv.org/pdf/2302.04761.pdf

实验表明，该方法能使小规模（5b, 20b）语言模型在事实陈述、多语言、数学计算等方面接近或显著超过大规模（60b, 175b）语言模型。使用该方法Finetune后的LLM能自动在输入的合适位置插入请求外部资源服务（搜索引擎、计算器、翻译、问答）的API，并根据API返回内容，影响后续生成结果。


![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308171552670.png)

1. 提示模板：对每一个C集合中的x，使用如下几个人工设计的提示模板，输入模型M，每个提示类型，生成k条含有API Call的$x'_i, i\in\{1,...,m\}$。实践中k=5，m=5。（翻译API Call的k=20, m=10）

### 构造样本方法
问答,计算,翻译,维基百科
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308171554346.png)

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308171555511.png)

计算API Call对后续生成结果是否有帮助是通过定义一个交叉熵的loss函数，并计算由于调用API Call引起的loss函数差值是否超过阈值$\tau_f$，实践中$\tau_f=1.0$。
$$L_i(\mathbf{z})=-\sum_{j=i}^nw_{j-i}\cdot\log p_M(x_j\mid\mathbf{z},x_{1:j-1})$$
插入API Call后的文本和对应调用后loss差值示例，可以看到会有一些正例被误过滤。
$$\begin{aligned}L_i^+&=L_i(\mathsf{e}(c_i,r_i))\\L_i^-&=\min\left(L_i(\varepsilon),L_i(\mathsf{e}(c_i,\varepsilon))\right)\end{aligned}$$

#### 为什么用交叉熵
推测: 因为这里API生成的是一个文本序列, 交叉熵是比较这个结果文本序列和模型预期生成的差别, 如果差别太大就不适合这个模型去学习
#### 这样做为什么有效
如果调用API和使用结果有用, 则会对预测下一个token有帮助, 也就是loss减小.于此对应的就是和不使用apiCALL和不返回结果去做对比

### 优势与局限
* Toolformer方法能显著改善（5b, 20b）参数规模的语言模型在时效性、虚假陈述、数学计算等方面的局限，使之追平或大幅超越千亿参数的LLM。使用C*对模型M finetuning 产生M*的过程只需要1台8卡A100 40GB节点。
* 由于需要finetuning，不能灵活调整或扩展不同类型的API Call。


|方案|支持非GPT模型|无需调优算力|可频繁灵活扩展|支持全本地部署|使用门槛低|是否开源|品质高|
|--|--|--|--|--|--|--|--|
|langchain|Y|Y|Y|Y|Y|Y|N具有一定不确定性|
|Toolformer|Y|N|N|Y|N|Y|Y|
|ChatGPT Plugins|N|Y|Y但依赖OpenAI|N|NA|NA|Y|