---
title: LoRA
date: 2023-05-08 21:04:50
mathjax: true
---

# LoRA
## 优缺点
* 推理的时候$WX=W_0X+BAX=(W_0+BA)X$，因此可以直接合并权重矩阵减少推理时间和空间开销，在切换下游任务的时候减去$BA$并加上新的$B^`A^`$即可
* 优点：大大减少了下游任务的可训练参数，因为不需要保存大部分参数的优化器状态甚至梯度，降低了GPU显存要求；可以共享预训练模型参数，在不同的下游任务只需要更换低秩矩阵即可；通过简单线性设计，在部署的时候低秩矩阵可以和预训练权重矩阵合并，没有额外的推理开销；可以和其他的微调方法相结合。
* 缺点：在推理的时候通过将$BA$和$W_0$权重矩阵合并来避免推理延迟，但当同一batch中包含使用不同$BA$的下游任务时，将无法work，虽然可以通过不合并$BA$和$W_0$并根据不同的下游任务选择$BA$，但这又会带来推理延迟。
## 常规微调期间的训练过程
假设 W 表示给定神经网络层中的权重矩阵。然后，使用常规反向传播，我们可以获得权重更新 ΔW，它通常计算为损失乘以学习率的负梯度：
ΔW = α ( -∇ LW).

然后，当我们有 ΔW 时，我们可以更新原始权重如下：W' = W + ΔW。如下图所示（为简单起见，省略了偏置向量）：

或者，我们可以将权重更新矩阵分开并按如下方式计算输出：h = W x + ΔW x，

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305151515391.png)

因此，当我们在神经网络中训练完全连接（即“密集”）层时，如上所示，权重矩阵通常具有完整的rank，这是一个技术术语，意思是矩阵没有任何线性依赖（即“冗余”）行或列。相反，对于完整rank，低rank意味着矩阵具有冗余的行或列。

因此，根据Aghajanyan等人的说法，虽然预训练模型的权重在预训练任务上具有完整的rank，但LoRA作者指出，预训练的大型语言模型在适应新任务时具有较低的“内在维度”。

低内在维度意味着数据可以通过低维空间有效地表示或近似，同时保留其大部分基本信息或结构。换句话说，这意味着我们可以将适应任务的新权重矩阵分解为低维（较小）矩阵，而不会丢失太多重要信息。

例如，假设 ΔW 是 A × B 权重矩阵的权重更新。然后，我们可以将权重更新矩阵分解为两个较小的矩阵：ΔW = W A W B ，其中 W A 是 A × r 维矩阵，W B 是 r × B 维矩阵。在这里，我们保持原始权重 W 冻结，只训练新矩阵 W A 和 W B 。简而言之， 这就是 LoRA 方法， 如下图所示.

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305151517089.png)
------



"Low-Rank Adaptation of Large Language Models" 的原理是通过对大型语言模型进行低秩适应来减少参数数量，并提高模型的泛化能力和训练效率。

大型语言模型通常具有数十亿个参数，而这些参数需要大量的计算资源进行训练和推断，且容易产生过拟合等问题。因此，减少语言模型的参数数量是非常有必要的。而低秩矩阵分解可以将一个高维矩阵表示成若干个低秩矩阵的和，从而降低矩阵的复杂度，减少参数数量。

具体地，对于大型语言模型的参数矩阵$W \in \mathbb{R}^{d_1 \times d_2}$，可以将其分解为两个低秩矩阵的乘积形式，即$W = UV^\top$，其中$U \in \mathbb{R}^{d_1 \times r}$和$V \in \mathbb{R}^{d_2 \times r}$均为低秩矩阵，$r$为低秩因子的数量。

在训练过程中，可以使用低秩矩阵对原始语言模型进行适应，从而得到一个新的语言模型。具体地，可以通过在原始语言模型参数$\theta$上加上一个低秩矩阵$\Delta$，得到一个新的参数$\tilde{\theta} = \theta + \Delta$。然后，可以使用随机梯度下降等优化方法对新的语言模型进行训练，并在适当的时候停止训练以防止过拟合。

低秩适应可以使得新的语言模型参数数量大大减少，同时能够保持模型的拟合能力，提高模型的泛化能力和训练效率。

## 代码实现
以下是在Transformer模型中使用LoRA进行低秩适应的PyTorch代码实现：

```python
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class LoRATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, rank):
        super(LoRATransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 低秩适应参数
        self.rank = rank
        self.U = Parameter(torch.Tensor(d_model, rank))
        self.V = Parameter(torch.Tensor(rank, d_model))
        self.init_parameters()
        
    def init_parameters(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 原始self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        
        # 低秩适应
        U, V = self.U, self.V
        src2 = torch.matmul(src, U)
        src2 = torch.matmul(src2, V)
        src = src + self.dropout(src2)
        
        # 原始feedforward
        src2 = self.linear1(src)
        src2 = nn.functional.relu(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout(src2)
        
        return src
```

上述代码中，我们定义了一个名为`LoRATransformerLayer`的新的Transformer层。其中，我们使用了原始Transformer层的self-attention和feedforward模块，同时在self-attention之后加入了LoRA的低秩适应模块。

在低秩适应模块中，我们使用了两个可训练的参数矩阵$U$和$V$，分别表示输入和输出特征的低秩表示。在forward方法中，我们首先进行了原始的self-attention计算，并使用dropout进行正则化；然后，通过将输入特征与$U$相乘，将其低秩表示为一个较小的中间特征；最后，将中间特征与$V$相乘得到低秩表示的输出特征。最终，我们将输出特征与输入特征相加，并使用dropout进行正则化，得到低秩适应后的输出。