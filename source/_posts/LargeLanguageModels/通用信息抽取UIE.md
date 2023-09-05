---
title: 通用信息抽取UIE
date: 2023-09-05 23:24:27
---

## 统一信息抽取任务输出结构的结构化抽取语言SEL

用的模型还是T5
> https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/information_extraction/DuUIE/run_seq2struct.py#L39
> 
> https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/t5#transformers.T5ForConditionalGeneration



---
## 任务介绍

> https://zhuanlan.zhihu.com/p/495315026


作者发现四种信息抽取任务的目标都可以拆解成两个原子操作：

1. Spotting：指在输入的原句中找到目标信息片段，比如说实体识别中某个类型的实体，事件抽取中的触发词和论元，他们都是原句中的片段。
2. Associating：指找出Spotting输出的信息片段之间的关系，比如关系抽取中两个实体之间的关系，或事件抽取中论元和触发词之间的关系。


而每个信息抽取任务都可以用这两个原子操作去完成，因此作者设计了结构化抽取语言SEL可以对这两个原子操作进行表示，不同的任务只要组合不同的原子操作对应结构即可统一表示。

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011714715.png)

## 指导模型做特定任务的结构化模式指导器SSI
有了SEL语法，模型统一了不同任务的输出结构，但是当我们输入一个句子后，如何让模型去做我们想要的任务呢？因此作者提出了SSI(Structural Schema Instructor)，是一种基于Schema的prompt。当我们输入句子时，在句子前面拼接上对应的Prompt，即可让模型做对应的任务。

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011716668.png)

以下用 $s$表示SSI，用$x$ 
 表示需要输入的原始句子，UIE表示UIE模型，它由transformer的Encoder和Decoder组成，形式化定义如式(1)：
 ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011717952.png)
 
 输出 $y$就是采用SEL语法描述的结构化数据，其中 $s \oplus x$表示如式(2)：
 ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011719874.png)
 
 详细来说，首先将 $s \oplus x$输入至Encoder，得到每一个token的隐层表示，形式化表示如式(3)
 ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011719048.png)
 
 接下来使用隐层表示在Decoder端生成目标结构化信息，表示如式(4)所示：
 ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011719464.png)
 
 ## 预训练任务
 ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011720819.png)
 
 ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308011720440.png)