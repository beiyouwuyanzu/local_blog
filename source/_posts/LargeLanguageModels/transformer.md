---
title: transformer
date: 2023-05-08 21:04:50
mathjax: true
---


## 如何理解transformer中decoder的k, v来自encoder的输出
```python
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory_bank, src_mask=None, tgt_mask=None, memory_mask=None):
        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](output, memory_bank, src_mask=src_mask, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask)
        return output
```

> 其中，memory_bank 即为编码器的输出，而 src_mask、tgt_mask 和 memory_mask 分别为编码器输入序列、解码器输入序列和编码器输出序列的掩码矩阵。

在Transformer中，Decoder的Query向量来自上一个Decoder层的输出，而Key和Value向量则来自Encoder的输出。因为Encoder中的Key和Value表示的是源语言的信息，而Decoder需要根据当前的Query向量和Encoder的Key和Value向量计算Attention权重，从而得到当前位置的上下文信息。在这个过程中，由于Encoder的Key和Value表示的都是源语言的信息，因此它们的含义是相同的，因此可以将它们看作相等的。

在实践中，Decoder的Key和Value向量是通过复制Encoder的输出得到的，即$K=V=X$，其中$X$是Encoder的输出。这种方法既可以减少模型参数量，也可以提高模型的泛化能力。因为在训练过程中，Encoder和Decoder使用的都是相同的权重，因此它们的表示能力是相同的。这样做可以避免在训练中过度拟合训练数据，同时提高模型在测试数据上的性能。

## self attention和target attention
在Transformer中，self-attention和target-attention都是一种注意力机制，用于计算输入序列中每个位置对其他位置的重要性权重。

Self-attention也称为intra-attention，是在Encoder和Decoder中使用的一种注意力机制。它用于计算输入序列中每个位置与其他位置的相似度，并根据相似度计算每个位置与其他位置的权重。具体地，对于每个位置，self-attention通过计算该位置与所有其他位置的相似度来得到一个与输入序列长度相同的权重向量，用于对所有其他位置的表示进行加权平均。

Target-attention也称为inter-attention，是在Decoder中使用的一种注意力机制。它用于计算Decoder中的每个位置对Encoder中所有位置的重要性，并根据重要性计算每个位置与Encoder中所有位置的权重。具体地，对于每个Decoder位置，target-attention通过计算该位置与Encoder中所有位置的相似度来得到一个与Encoder输入序列长度相同的权重向量，用于对Encoder中所有位置的表示进行加权平均。

虽然self-attention和target-attention的计算方式略有不同，但它们都遵循了相同的基本原则：通过计算每个位置与其他位置的相似度，来为每个位置分配一个权重向量，用于加权平均其他位置的表示。这种注意力机制在Transformer中的应用，大大提高了模型的表达能力和泛化能力，成为了当前自然语言处理领域中最为流行的模型之一。

## transformer的位置编码
- 公式
$$
PE_{(pos,2i)}=sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos,2i+1)}=cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/mGSYD.png)
让我们尝试理解计算位置嵌入的公式的“sin”部分
这里的“pos”指的是“word”在序列中的位置。 P0指的是第一个词的位置embedding； “d”表示单词/标记嵌入的大小。在本例中 d=5。最后，“i”指的是嵌入的 5 个单独维度中的每一个（即 0、1、2、3、4）
虽然“d”是固定的，但“pos”和“i”是变化的
- pos
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/uxfb4.png)

如果我们绘制一条 sin 曲线并改变“pos”（在 x 轴上），您将在 y 轴上获得不同的位置值。因此，具有不同位置的词将具有不同的位置嵌入值。
但是有一个问题。由于“sin”曲线在一定间隔内重复，您可以在上图中看到 P0 和 P6 具有相同的位置嵌入值，尽管它们位于两个非常不同的位置。这就是等式中“i”部分发挥作用的地方。
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/Fhc4M.png)

如果你改变上面等式中的“i”，你会得到一堆频率不同的曲线。读取不同频率的位置嵌入值，得出 P0 和 P6 在不同嵌入维度下的不同值。
## 为什么位置编码要区分奇数位置和偶数位置
如果使用一个正弦函数，对于每个位置，它的位置向量中的i值都是不同的，因此仍然能够区分不同位置。实际上，这种方法也被用于一些Transformer的实现中。使用两个不同的函数来编码位置向量，是一种较为常见的实现方法，但不是唯一的实现方式。

但是这样做会导致位置编码向量之间的相似度随着位置之间的距离增加而减小
对于一个序列中的任意两个位置$i$和$j$，如果它们之间的距离$d=|i-j|$大于一个周期的长度$2\pi$，那么位置编码向量$PE_i$和$PE_j$中对应的正弦函数值会是一样的，即：

$$
\begin{aligned}
PE_{i,k} &= \sin\left(\frac{i}{10000^{2k/d_{model}}}\right) \\
PE_{j,k} &= \sin\left(\frac{j}{10000^{2k/d_{model}}}\right) \\
&= \sin\left(\frac{i+d}{10000^{2k/d_{model}}}\right) \\
&= \sin\left(\frac{i}{10000^{2k/d_{model}}}\right)
\end{aligned}
$$

这就使得$PE_i$和$PE_j$之间的相似度会随着位置之间的距离增加而减小。

为了避免这个问题，Transformer中使用了正弦函数和余弦函数的组合来编码位置信息，因为余弦函数的周期与正弦函数不同，使得每个位置的位置向量更加独特，从而提高了模型对位置信息的理解能力。同时，使用不同的函数来编码奇数位置和偶数位置的位置向量，也进一步增加了位置向量之间的差异性，有助于提高模型的性能。

> 另一种证明
> 在 Transformer 中，位置编码是通过将正弦和余弦函数作用于位置信息来实现的。如果只使用正弦函数，那么对于任意两个位置 i 和 j，它们的位置编码向量之间的相似度可以表示为：

$$\begin{aligned} \text{sim}(PE_i, PE_j) &= \cos(\text{PE}_i, \text{PE}_j) \\ &= \cos((i+1)\times f(k), (j+1)\times f(k)) \\ &= \cos(i\times f(k) + f(k), j\times f(k) + f(k)) \\ &= \cos(i\times f(k))\cos(f(k)) + \sin(i\times f(k))\sin(f(k))\cos(j\times f(k))\cos(f(k)) + \sin(j\times f(k))\sin(f(k)) \end{aligned} $$

其中，f(k) 是一个常数，它控制了正弦和余弦函数的周期。从上式中可以看出，当 i 和 j 的差距较大时，它们的位置编码向量之间的相似度会随着差距的增加而减小。这会导致模型难以学习到不同位置之间的关系。

