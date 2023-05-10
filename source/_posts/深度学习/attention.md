---
title: 注意力机制
date: 2023-05-08 21:04:50
mathjax: true
---

# 注意力机制
## 概述
注意力机制（Attention Mechanism）是一种在深度学习中广泛使用的技术，它允许模型在处理输入序列时，能够更加准确地关注与输出有关的信息，提高模型的性能

假设我们有一个输入序列 $x_1, x_2, ..., x_n$ 和一个输出序列 $y_1, y_2, ..., y_m$。为了计算每个输出 $y_i$ 和所有输入 $x_j$ 之间的关系，我们引入一个注意力向量 $a_i \in \mathbb{R}^n$，其中 $a_{i,j}$ 表示 $y_i$ 对 $x_j$ 的注意力权重。

现在，我们定义每个输出 $y_i$ 为输入序列的加权和，权重由对应的注意力向量 $a_i$ 来确定：

$$
y_i = \sum_{j=1}^{n} a_{i,j} x_j
$$

在实现注意力机制时，最常见的方法是使用点积注意力（Dot-Product Attention）。在点积注意力中，我们首先将输入和输出映射到低维空间（通常是一个固定大小的向量空间），然后通过点积来计算注意力权重。具体地，对于每个输出 $y_i$，我们计算其对输入 $x_j$ 的注意力权重 $a_{i,j}$ 如下：

$$
a_{i,j} = \frac{\exp\left(\mathrm{score}(y_i, x_j)\right)}{\sum_{k=1}^{n} \exp\left(\mathrm{score}(y_i, x_k)\right)}
$$

其中，$\mathrm{score}(y_i, x_j)$ 是一个衡量 $y_i$ 和 $x_j$ 之间相似度的函数，常见的有点积、加性、和双线性等。点积注意力中的 $\mathrm{score}(y_i, x_j)$ 定义为两个向量的点积：

$$
\mathrm{score}(y_i, x_j) = y_i \cdot x_j
$$

通过以上公式，我们可以计算出每个输出 $y_i$ 对应的注意力权重 $a_i$，然后使用加权和来计算 $y_i$。需要注意的是，在实际应用中，我们通常还会对注意力权重进行一些调整，例如添加一个缩放因子、使用多头注意力等，以提高模型的性能和鲁棒性。

## 代码实现
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(DotProductAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        
    def forward(self, query, key, value, mask=None):
        Q = self.query(query)  # shape: (batch_size, hidden_dim)
        K = self.key(key)  # shape: (batch_size, seq_len, hidden_dim)
        V = self.value(value)  # shape: (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)).squeeze(1)  # shape: (batch_size, seq_len)
        
        # Mask out padding positions
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)  # shape: (batch_size, seq_len)
        attn_values = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)  # shape: (batch_size, hidden_dim)
        
        return attn_values, attn_weights
```


在上面的代码中，`DotProductAttention` 类封装了点积注意力机制的实现。在构造函数中，我们定义了三个线性层 `query`、`key` 和 `value`，分别用于将输入序列映射到隐藏空间中。在 `forward` 方法中，我们将输入的查询向量 `query`、键向量 `key` 和值向量 `value` 作为输入，然后分别将它们通过线性层映射到隐藏空间中，并计算它们之间的点积注意力得分。

在计算注意力得分时，我们使用了 PyTorch 中的 `bmm` 函数，它可以高效地进行批矩阵乘法。具体地，我们将查询向量 $Q$ 与键向量 $K$ 的转置相乘，得到每个查询向量对键向量的注意力得分。

接下来，我们可以使用 Softmax 函数将注意力得分转换为注意力权重，并将它们应用到值向量上，得到最终的注意力向量。需要注意的是，在实际使用中，我们通常还需要对注意力权重进行缩放、加入残差连接等操作，以提高模型的性能和稳定性。

## 传统RNN模型中注意力机制的实现

在传统的 GRU 模型中，可以使用 Bahdanau 注意力机制来增强模型的表现力和泛化能力。Bahdanau 注意力机制的公式形式如下： $$\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k=1}^T \exp(e_{i,k})}$$ $$c_i = \sum_{j=1}^T \alpha_{i,j}h_j$$ 其中，$T$ 表示输入序列的长度，$h_j$ 表示编码器在时间步 $j$ 的隐藏状态，$c_i$ 表示在时间步 $i$ 的上下文向量，$\alpha_{i,j}$ 表示在时间步 $i$，编码器的隐藏状态 $h_j$ 对上下文向量的贡献权重，$e_{i,j}$ 是一个可学习的向量，用于计算注意力权重。 在具体实现时，可以使用一个全连接层或线性层来计算 $e_{i,j}$，然后使用 softmax 函数将 $e_{i,j}$ 转化为注意力权重 $\alpha_{i,j}$。最终的上下文向量 $c_i$ 则是编码器的隐藏状态 $h_j$ 的加权和。


下面是一个示例代码，展示了如何在传统的 GRU 模型中使用注意力机制：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Define the bidirectional GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True)
        
        # Define the attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)
        
        # Define the output layer
        self.output = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, input):
        # Pass the input sequence through the bidirectional GRU layer
        outputs, hidden = self.gru(input)  # outputs shape: (seq_len, batch_size, hidden_dim * 2)
        
        # Compute attention weights
        weights = self.attention(outputs.transpose(0, 1))  # shape: (batch_size, seq_len, 1)
        weights = F.softmax(weights, dim=1)  # shape: (batch_size, seq_len, 1)
        
        # Compute the weighted sum of the GRU outputs
        context = torch.bmm(outputs.transpose(0, 1), weights)  # shape: (batch_size, hidden_dim * 2, 1)
        context = context.squeeze(2)  # shape: (batch_size, hidden_dim * 2)
        
        # Pass the context vector through the output layer
        output = self.output(context)  # shape: (batch_size, output_dim)
        
        return output
```

在上面的代码中，我们定义了一个名为 `GRUAttention` 的类，它继承自 PyTorch 中的 `nn.Module` 类。在构造函数中，我们首先定义了一个双向 GRU 层，用于对输入序列进行编码。然后，我们定义了一个线性层，用于计算注意力权重，并定义了一个输出层，用于将上下文向量映射到输出空间中。

在前向传递过程中，我们将输入序列作为输入传递给双向 GRU 层，并获取每个时间步的输出。然后，我们将这些输出传递给注意力层，用于计算注意力权重。接着，我们使用注意力权重将 GRU 输出加权求和，得到上下文向量。最后，我们将上下文向量作为输入传递给输出层，得到最终的输出结果。